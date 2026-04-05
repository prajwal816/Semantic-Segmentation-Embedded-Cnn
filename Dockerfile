# x86_64 CUDA-capable dev image. Multi-stage: Python deps + C++ build smoke.
# On Jetson (arm64), use docker/jetson-l4t.Dockerfile instead.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    cmake ninja-build build-essential \
    libopencv-dev \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models/onnx models/checkpoints models/engines data/outputs benchmarks data/calibration

# CI-sized models + ONNX reference check + compile edge binary + headless run
RUN python3 scripts/ci_prepare_models.py \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && ./build/seg_edge_pipeline configs/pipeline_ci.json

CMD ["/bin/bash", "-lc", "echo Image verified: Python + ONNX reference + seg_edge_pipeline build."]
