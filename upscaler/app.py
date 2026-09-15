import ctypes
import glob
import grp
import io
import logging
import os
import stat
import subprocess
import time
import traceback

import numpy as np
from flask import Flask, jsonify, request, send_file
from openvino.runtime import Core, PartialShape
from PIL import Image

MODEL_XML = "/models/single-image-super-resolution-1032.xml"

# The model's 4x scale factor is fixed by its architecture (its
# pixel-shuffle upsampling depth is baked in, not derived from input
# shape) - reshaping only changes the working resolution, not the
# scale. So to go from a 1920x1080 source to an exact 4K output, we
# reshape the model to accept the full 1080p source untouched
# (preserving all real detail) and let it run at its native 4x,
# producing a raw 7680x4320 result that we then resize down to the
# exact 4K target. This uses real source detail rather than
# pre-shrinking it, at the cost of ~16x the compute of the model's
# published reference benchmark (which used a 480x270 input) -
# expect inference times in the low seconds, not sub-second.
SRC_W, SRC_H = 1920, 1080
NATIVE_W, NATIVE_H = SRC_W * 4, SRC_H * 4
TARGET_W, TARGET_H = 3840, 2160

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upscaler")

app = Flask(__name__)

def gpu_diagnostics(core: Core) -> dict:
    """Collect enough context to debug a GPU-not-found failure over HTTP,
    since the add-on container may not be reachable via docker exec."""
    diag = {}
    try:
        diag["available_devices"] = core.available_devices
    except Exception as exc:
        diag["available_devices_error"] = str(exc)

    diag["process"] = {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "groups": [grp.getgrgid(g).gr_name for g in os.getgroups()],
    }

    dri_path = "/dev/dri"
    if os.path.isdir(dri_path):
        entries = {}
        for name in os.listdir(dri_path):
            full = os.path.join(dri_path, name)
            entry = {}
            try:
                st = os.stat(full)
                entry["mode"] = oct(stat.S_IMODE(st.st_mode))
                entry["uid"] = st.st_uid
                entry["gid"] = st.st_gid
                if not stat.S_ISDIR(st.st_mode):
                    try:
                        entry["group_name"] = grp.getgrgid(st.st_gid).gr_name
                    except KeyError:
                        entry["group_name"] = None
                        entry["group_name_note"] = "gid not mapped in this container's /etc/group"
            except Exception as exc:
                entry["stat_error"] = str(exc)

            if name in ("card0", "renderD128"):
                # stat() only needs the parent dir's search bit; actually
                # opening the device is what a cgroup device-controller
                # restriction would block, even for root, so test that too.
                try:
                    fd = os.open(full, os.O_RDWR)
                    os.close(fd)
                    entry["open_rdwr"] = "ok"
                except OSError as exc:
                    entry["open_rdwr_error"] = f"{exc.strerror} (errno {exc.errno})"

                pci_dir = f"/sys/class/drm/{name}/device"
                for field in ("vendor", "device", "uevent"):
                    field_path = os.path.join(pci_dir, field)
                    try:
                        with open(field_path) as f:
                            entry[f"sysfs_{field}"] = f.read().strip()
                    except Exception as exc:
                        entry[f"sysfs_{field}_error"] = str(exc)

            entries[name] = entry
        diag["dev_dri_contents"] = entries
    else:
        diag["dev_dri_contents"] = None
        diag["dev_dri_error"] = f"{dri_path} does not exist in this container"

    icd_files = glob.glob("/etc/OpenCL/vendors/*")
    icds = {}
    for icd_path in icd_files:
        try:
            with open(icd_path) as f:
                so_name = f.read().strip()
        except Exception as exc:
            icds[icd_path] = {"read_error": str(exc)}
            continue
        entry = {"so_name": so_name}
        try:
            ctypes.CDLL(so_name)
            entry["dlopen"] = "ok"
        except OSError as exc:
            entry["dlopen_error"] = str(exc)
        icds[icd_path] = entry
    diag["opencl_vendor_icds"] = icds

    dpkg_packages = ["intel-igc-core-2", "intel-igc-opencl-2", "intel-opencl-icd", "libigdgmm12"]
    try:
        out = subprocess.run(
            ["dpkg-query", "-W", "-f=${Package}=${Version}\n", *dpkg_packages],
            capture_output=True,
            text=True,
            timeout=5,
        )
        diag["installed_driver_packages"] = out.stdout.strip().splitlines()
        if out.stderr.strip():
            diag["installed_driver_packages_stderr"] = out.stderr.strip()
    except Exception as exc:
        diag["installed_driver_packages_error"] = str(exc)

    return diag


# The model is compiled with an explicit device_name="GPU" (never "AUTO"),
# so it never silently falls back to CPU. If GPU compile fails, the add-on
# stays up (rather than crash-looping) and serves the failure + diagnostics
# over /health and /upscale, since the container may not be reachable any
# other way to debug it.
compiled_model = None
input_src = None
input_bicubic = None
output_layer = None
startup_error = None
last_inference = {"device": None, "ms": None}

log.info("Loading model, reshaping to %dx%d source, compiling for device=GPU ...", SRC_W, SRC_H)
core = Core()
try:
    model = core.read_model(MODEL_XML)
    model.reshape(
        {
            model.inputs[0]: PartialShape([1, 3, SRC_H, SRC_W]),
            model.inputs[1]: PartialShape([1, 3, NATIVE_H, NATIVE_W]),
        }
    )
    compiled_model = core.compile_model(model, device_name="GPU")
    input_src = compiled_model.input(0)
    input_bicubic = compiled_model.input(1)
    output_layer = compiled_model.output(0)

    execution_devices = ",".join(compiled_model.get_property("EXECUTION_DEVICES"))
    log.info("Model compiled. EXECUTION_DEVICES=%s", execution_devices)
    last_inference["device"] = execution_devices
except Exception:
    log.error("GPU model compile FAILED:\n%s", traceback.format_exc())
    startup_error = {
        "error": traceback.format_exc(),
        "diagnostics": gpu_diagnostics(core),
    }
    log.error("GPU diagnostics: %s", startup_error["diagnostics"])


def preprocess(image: Image.Image):
    rgb = image.convert("RGB")
    src = rgb.resize((SRC_W, SRC_H), Image.LANCZOS)
    bicubic = src.resize((NATIVE_W, NATIVE_H), Image.BICUBIC)

    src_bgr = np.array(src)[:, :, ::-1]
    bicubic_bgr = np.array(bicubic)[:, :, ::-1]

    src_input = src_bgr.transpose(2, 0, 1)[None].astype(np.float32)
    bicubic_input = bicubic_bgr.transpose(2, 0, 1)[None].astype(np.float32)
    return src_input, bicubic_input


def postprocess(result: np.ndarray) -> Image.Image:
    arr = result[0].transpose(1, 2, 0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[:, :, ::-1]
    native = Image.fromarray(arr)
    return native.resize((TARGET_W, TARGET_H), Image.LANCZOS)


@app.route("/health")
def health():
    if startup_error is not None:
        return jsonify({"status": "gpu_unavailable", **startup_error}), 503
    return jsonify(
        {
            "status": "ok",
            "device": last_inference["device"],
            "last_inference_ms": last_inference["ms"],
        }
    )


@app.route("/upscale", methods=["POST"])
def upscale():
    if startup_error is not None:
        return jsonify({"status": "gpu_unavailable", **startup_error}), 503

    if "file" not in request.files:
        return jsonify({"error": "no file field named 'file'"}), 400

    upload = request.files["file"]
    try:
        image = Image.open(upload.stream)
        image.load()
    except Exception as exc:
        return jsonify({"error": f"invalid image: {exc}"}), 400

    src_input, bicubic_input = preprocess(image)

    start = time.perf_counter()
    result = compiled_model({input_src: src_input, input_bicubic: bicubic_input})[output_layer]
    elapsed_ms = (time.perf_counter() - start) * 1000

    last_inference["ms"] = round(elapsed_ms, 1)
    log.info("upscale request: device=%s time_ms=%.1f", last_inference["device"], elapsed_ms)

    out_image = postprocess(result)
    buf = io.BytesIO()
    out_image.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5300))
    app.run(host="0.0.0.0", port=port, threaded=True)
