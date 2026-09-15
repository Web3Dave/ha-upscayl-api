import io
import logging
import os
import time

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

log.info("Loading model, reshaping to %dx%d source, compiling for device=GPU ...", SRC_W, SRC_H)
core = Core()
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

last_inference = {"device": execution_devices, "ms": None}


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
    return jsonify(
        {
            "status": "ok",
            "device": last_inference["device"],
            "last_inference_ms": last_inference["ms"],
        }
    )


@app.route("/upscale", methods=["POST"])
def upscale():
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
