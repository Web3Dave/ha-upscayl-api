import io
import logging
import os
import time

import numpy as np
from flask import Flask, jsonify, request, send_file
from openvino.runtime import Core
from PIL import Image

MODEL_XML = "/models/single-image-super-resolution-1032.xml"
INPUT_W, INPUT_H = 480, 270
OUTPUT_W, OUTPUT_H = 1920, 1080

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upscaler")

app = Flask(__name__)

log.info("Loading model and compiling for device=GPU ...")
core = Core()
model = core.read_model(MODEL_XML)
compiled_model = core.compile_model(model, device_name="GPU")
input_lr = compiled_model.input(0)
input_bicubic = compiled_model.input(1)
output_layer = compiled_model.output(0)

execution_devices = ",".join(compiled_model.get_property("EXECUTION_DEVICES"))
log.info("Model compiled. EXECUTION_DEVICES=%s", execution_devices)

last_inference = {"device": execution_devices, "ms": None}


def preprocess(image: Image.Image):
    rgb = image.convert("RGB")
    lr = rgb.resize((INPUT_W, INPUT_H), Image.LANCZOS)
    bicubic = lr.resize((OUTPUT_W, OUTPUT_H), Image.BICUBIC)

    lr_bgr = np.array(lr)[:, :, ::-1]
    bicubic_bgr = np.array(bicubic)[:, :, ::-1]

    lr_input = lr_bgr.transpose(2, 0, 1)[None].astype(np.float32)
    bicubic_input = bicubic_bgr.transpose(2, 0, 1)[None].astype(np.float32)
    return lr_input, bicubic_input


def postprocess(result: np.ndarray) -> Image.Image:
    arr = result[0].transpose(1, 2, 0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[:, :, ::-1]
    return Image.fromarray(arr)


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

    lr_input, bicubic_input = preprocess(image)

    start = time.perf_counter()
    result = compiled_model({input_lr: lr_input, input_bicubic: bicubic_input})[output_layer]
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
