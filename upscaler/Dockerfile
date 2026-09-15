FROM python:3.11-slim-trixie

# ocl-icd-libopencl1 is the generic OpenCL loader; curl/jq are for fetching
# the model at build time and reading add-on options at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Intel Compute Runtime (NEO), the actual GPU OpenCL driver the OpenVINO GPU
# plugin depends on. Debian bookworm's "intel-opencl-icd" apt package is
# pinned to 22.43 (late 2022), which predates Alder Lake-N (this device's
# N100) and silently enumerates zero GPU devices on it - OpenVINO then falls
# through to CPU-only, even with device_name="GPU" explicit and /dev/dri
# correctly passed through. Installing current .deb releases directly from
# Intel's GitHub instead, same approach Frigate's own image uses for the
# same hardware.
ARG IGC_VERSION=v2.40.13
ARG IGC_BUILD=2.40.13+22418
ARG NEO_VERSION=26.31.39395.13
ARG GMMLIB_VERSION=22.10.0
RUN set -eux; \
    tmp="$(mktemp -d)"; cd "$tmp"; \
    curl -fsSLO "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-core-2_${IGC_BUILD}_amd64.deb"; \
    curl -fsSLO "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-opencl-2_${IGC_BUILD}_amd64.deb"; \
    curl -fsSLO "https://github.com/intel/compute-runtime/releases/download/${NEO_VERSION}/libigdgmm12_${GMMLIB_VERSION}_amd64.deb"; \
    curl -fsSLO "https://github.com/intel/compute-runtime/releases/download/${NEO_VERSION}/intel-opencl-icd_${NEO_VERSION}-0_amd64.deb"; \
    dpkg -i ./*.deb || apt-get -f install -y --no-install-recommends; \
    dpkg -i ./*.deb; \
    cd /; rm -rf "$tmp"; rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Open Model Zoo single-image-super-resolution-1032 (4x SR, FP16, GPU-friendly).
# Pinned to the 2022.1 model release path on the official OpenVINO storage host.
RUN mkdir -p /models && \
    MODEL_BASE="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/single-image-super-resolution-1032/FP16" && \
    curl -fsSL -o /models/single-image-super-resolution-1032.xml "${MODEL_BASE}/single-image-super-resolution-1032.xml" && \
    curl -fsSL -o /models/single-image-super-resolution-1032.bin "${MODEL_BASE}/single-image-super-resolution-1032.bin"

COPY app.py .
COPY run.sh .
RUN chmod +x run.sh

ENV PORT=5300
EXPOSE 5300

CMD ["./run.sh"]
