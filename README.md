# HA Upscaler Add-on

A Home Assistant OS add-on that exposes a small HTTP API for 4x AI
image super-resolution, running on the Intel iGPU via OpenVINO.

## Model

Uses Intel Open Model Zoo's `single-image-super-resolution-1032`
(FP16), a 4x super-resolution network. Verified live at build time
from the official OpenVINO artifact storage:

```
https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/single-image-super-resolution-1032/FP16/
```

Open Model Zoo itself is archived/folded into newer OpenVINO tooling,
but this specific storage host and path are still live (checked
2026-09-15) and are the same artifact host OMZ always used, so no
substitute model was needed.

This model has **fixed input/output tensor shapes** (not dynamic):

- Input `0` (source image): `270x480` BGR
- Input `1` (bicubic upsample of the source, pre-resized to 4x): `1080x1920` BGR
- Output: `1080x1920` BGR

The API takes care of both resizes server-side, so callers only ever
send one image and receive one image back. Because the model's
internal resolution is fixed, any uploaded image is resized to
480x270 before inference and the output is always 1920x1080 — non
16:9 source images will be stretched to that aspect ratio. This is an
inherent limitation of this specific model, not a bug in the add-on.

## Add-on structure

```
/
├── repository.yaml
└── upscaler/
    ├── config.yaml   # add-on manifest
    ├── Dockerfile    # installs Intel Compute Runtime + OpenVINO, bakes in the model
    ├── run.sh         # reads the configured port from /data/options.json
    └── app.py         # Flask API + OpenVINO inference
```

## Install

1. In Home Assistant: **Settings → Add-ons → Add-on Store → ⋮ → Repositories**,
   add this repository's URL (or add it locally under
   `/addons` on the HA host for a local add-on).
2. Find "OpenVINO Image Upscaler" in the store and install it.
3. Start the add-on. Check the log tab — on startup it logs
   `Model compiled. EXECUTION_DEVICES=GPU`. If the log instead shows a
   traceback about no supported GPU devices, the add-on failed to
   start — see **Verifying GPU acceleration** below.

The add-on requires `/dev/dri` passthrough (already declared in
`config.yaml`, same pattern as this device's working Frigate OpenVINO
GPU detector config) and only targets `amd64`, matching this
device's Intel N100.

## API

### `POST /upscale`

`multipart/form-data` upload, field name `file`, containing the
source image. Returns the 4x-upscaled image as `image/png`.

```bash
curl -F "file=@input.jpg" http://<host>:5300/upscale -o output.png
```

### `GET /health`

```json
{
  "status": "ok",
  "device": "GPU",
  "last_inference_ms": 187.3
}
```

`device` reflects OpenVINO's actual `EXECUTION_DEVICES` property for
the compiled model (queried once at startup, since the model is
compiled with an explicit `device_name="GPU"` — there is no `AUTO`
fallback, so if the GPU isn't usable the add-on fails to start rather
than silently running on CPU). `last_inference_ms` is the wall-clock
time of the most recent `/upscale` inference call, updated on every
request.

## Verifying GPU acceleration

1. Check the add-on log for `EXECUTION_DEVICES=GPU` at startup.
2. Send a test image to `/upscale`, then check `/health` — the
   `device` field should read `GPU` and `last_inference_ms` should be
   in the low hundreds of milliseconds. The published reference
   benchmark for this model is ~250ms on a weaker iGPU than this
   device's UHD Graphics (Xe-LP); multi-second inference times mean
   it silently fell back to CPU and needs debugging (check `/dev/dri`
   passthrough and that the Intel Compute Runtime in the container can
   see the device — the add-on will not start at all if the GPU
   plugin can't enumerate a device, since `device_name="GPU"` is
   explicit).
3. Every `/upscale` request also logs `device=GPU time_ms=...` so you
   can watch acceleration hold up under repeated real traffic, not
   just at startup.

## Configuration

| Option | Default | Description |
|---|---|---|
| `port` | `5300` | Port the API listens on. |
