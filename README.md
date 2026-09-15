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

This model has **fixed input/output tensor shapes** (not dynamic),
and its **4x scale factor is baked into the architecture** — its
pixel-shuffle upsampling depth is a fixed part of the graph, not
derived from the input shape, so reshaping the model changes its
working resolution but never its scale factor.

The add-on is built for a **1920x1080 in → 3840x2160 (4K) out**
contract, which is only a 2x jump. To fit that onto a fixed-4x model,
`app.py` reshapes the model at startup to accept the full 1080p
source untouched — preserving all real source detail rather than
pre-shrinking it — runs it at its native 4x, and resizes the raw
7680x4320 result down to the exact 4K target:

- Input `0` (source image): reshaped to `1920x1080` BGR
- Input `1` (bicubic upsample of the source, pre-resized to native 4x): reshaped to `7680x4320` BGR
- Raw model output: `7680x4320` BGR, resized server-side down to `3840x2160`

The API takes care of every resize (up to the model's fixed input,
and back down to 4K) server-side, so callers only ever send one image
and receive one image back. Any uploaded image is resized to
1920x1080 before inference — non 16:9 source images will be
stretched to that aspect ratio.

**Performance tradeoff:** because the model runs at true 1080p
resolution instead of its originally-benchmarked 480x270, it's
processing ~16x the pixels of the published reference benchmark
(~250ms on a weaker iGPU than this device's). Expect inference times
in the low seconds rather than sub-second, even on GPU — that's the
cost of feeding it real full-resolution detail instead of a
pre-shrunk source. This is a deliberate quality-over-speed choice;
see **Verifying GPU acceleration** below for what a healthy number
looks like at this resolution vs. a CPU-fallback number.

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

**If you change `full_access`, `apparmor`, or `devices` in
`config.yaml` after already having the add-on installed, a
rebuild/update is not enough** — Docker's privileged/device-cgroup
permissions are set when a container is *created*, not on restart or
rebuild. Fully **uninstall** the add-on and **reinstall** it to force
Supervisor to create a fresh container with the new permissions; this
was the actual fix the one time this bit us during development (the
`/health` diagnostics kept reporting `EPERM` on `/dev/dri` with
`full_access: true` already set, across multiple rebuilds and even a
full host reboot, until a real uninstall/reinstall).

The add-on requires `/dev/dri` passthrough (already declared in
`config.yaml`, same pattern as this device's working Frigate OpenVINO
GPU detector config) and only targets `amd64`, matching this
device's Intel N100.

`config.yaml` also sets `apparmor: false`. Supervisor wraps every
add-on in a restrictive AppArmor profile by default, and that profile
blocks `/dev/dri` access even when the device is declared under
`devices:` — without disabling it, the GPU plugin fails to enumerate
any device at all (`RuntimeError: ... no supported devices found`),
which looks identical to a missing `/dev/dri` mount but isn't one.

The `Dockerfile` also skips Debian bookworm's `intel-opencl-icd` apt
package (pinned to driver version 22.43, from late 2022) and instead
installs current Intel Compute Runtime `.deb` releases directly from
GitHub, on a `trixie`-based image (bookworm's glibc is too old for
the current driver release) — worth keeping regardless of the next
point, since it's the correct current driver either way.

`config.yaml` also sets `full_access: true`. On this device (HAOS
18.2), `/health` diagnostics showed `/dev/dri/card0` and `renderD128`
correctly bind-mounted into the container with the right sysfs PCI ID
(`8086:46D1`, confirmed Alder Lake-N) — but actually opening either
device failed with `EPERM`, not `EACCES`. That distinction matters:
`EACCES` would mean a Unix permission-bits problem; `EPERM` on an
`open()` that passes `stat()` fine, even as root, is the signature of
Docker's cgroup device-controller not granting an allow-rule for that
device — a layer `devices:` alone didn't get right on this Supervisor
version, even combined with `apparmor: false`. `full_access: true`
maps to Docker's real `privileged` mode, which bypasses the cgroup
device allowlist entirely instead of relying on that finer-grained
(and apparently broken, here) passthrough path.

## API

### `POST /upscale`

`multipart/form-data` upload, field name `file`, containing the
source image (ideally 1920x1080 — other sizes are stretched to fit).
Returns a 3840x2160 (4K) image as `image/png`.

```bash
curl -F "file=@input.jpg" http://<host>:5300/upscale -o output.png
```

### `GET /health`

```json
{
  "status": "ok",
  "device": "GPU.0",
  "last_inference_ms": 3354.3
}
```

`device` reflects OpenVINO's actual `EXECUTION_DEVICES` property for
the compiled model (queried once at startup, since the model is
compiled with an explicit `device_name="GPU"` — there is no `AUTO`
fallback, so it never silently runs on CPU). `last_inference_ms` is
the wall-clock time of the most recent `/upscale` inference call,
updated on every request.

If the GPU plugin can't enumerate a device at startup, the add-on
does **not** crash-loop — it stays up and both `/health` and
`/upscale` return HTTP 503 with `"status": "gpu_unavailable"`, the
full startup traceback, and a `diagnostics` block (`available_devices`
per OpenVINO, whether `/dev/dri` exists in the container and what's
in it, and what OpenCL ICDs are installed). This exists specifically
so the failure is debuggable with just `curl`, without needing
Docker/host shell access to the add-on's container.

## Verifying GPU acceleration

1. `curl http://<host>:5300/health`. A healthy add-on returns
   `"status": "ok"` with `"device": "GPU"`. A GPU that failed to
   enumerate returns HTTP 503 with `"status": "gpu_unavailable"` and
   a `diagnostics` block — read that first, it tells you exactly what
   failed (e.g. `dev_dri_contents: null` means `/dev/dri` isn't even
   present inside the container — a Supervisor/passthrough problem,
   not something this add-on's code can fix). The add-on log carries
   the same information at startup (`EXECUTION_DEVICES=GPU` on
   success, or the traceback + diagnostics on failure).
   `no supported devices found` has three distinct causes that all
   produce the identical error text — use `diagnostics` to tell them
   apart instead of guessing:
   - `dev_dri_contents: null` — `/dev/dri` isn't present in the
     container at all. Check `apparmor: false` is in `config.yaml`
     (Supervisor's default AppArmor profile blocks `/dev/dri` even
     when it's declared under `devices:`) and that the passthrough
     itself is declared.
   - `dev_dri_contents` lists `card0`/`renderD128` (the device is
     there) but `available_devices` is still `["CPU"]`, and
     `opencl_vendor_icds` shows `/etc/OpenCL/vendors/intel.icd` — the
     device node is reachable but the installed compute-runtime
     driver doesn't recognize this GPU's PCI ID. This is what
     happened on this device's N100: Debian bookworm's stock
     `intel-opencl-icd` package predates Alder Lake-N. Confirm the
     Dockerfile is actually building the current Intel `.deb`
     releases (not silently falling back to an apt package) and that
     the build didn't skip that layer.
   - `opencl_vendor_icds` is empty — no OpenCL ICD installed at all;
     the Dockerfile's driver-install step didn't run or failed.
2. Send a test image to `/upscale`, then check `/health` — the
   `device` field should read `GPU`.
3. `last_inference_ms` is a secondary sanity signal, not the primary
   one: because this add-on runs the model at true 1080p (see
   **Model** above), timings are naturally higher than the model's
   original ~250ms reference benchmark (which used a much smaller
   480x270 input) even on GPU — expect low single-digit seconds.
   What still separates GPU from an accidental CPU fallback is scale,
   not an absolute threshold: CPU on this hardware runs several times
   slower than GPU for the same workload, so a number in the tens of
   seconds despite `device: "GPU"` in `/health` is a strong signal
   something's wrong upstream of the OpenVINO device check itself
   (e.g. GPU present but thermal/power throttled).
4. Every `/upscale` request also logs `device=GPU time_ms=...` so you
   can watch acceleration hold up under repeated real traffic, not
   just at startup.

## Configuration

| Option | Default | Description |
|---|---|---|
| `port` | `5300` | Port the API listens on. |
