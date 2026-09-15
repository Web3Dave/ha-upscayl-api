FROM python:3.11-slim-bookworm

# Intel Compute Runtime (NEO) OpenCL driver used by the OpenVINO GPU plugin,
# plus curl/jq for fetching the model at build time and reading add-on options.
RUN apt-get update && apt-get install -y --no-install-recommends \
        intel-opencl-icd \
        ocl-icd-libopencl1 \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

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
