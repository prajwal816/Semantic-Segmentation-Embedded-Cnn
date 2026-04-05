# x86_64 CUDA-capable dev image (simulates a Jetson-like toolchain layout).
# On Jetson, use an L4T base (see docker/jetson-l4t.Dockerfile) — this image will not run on arm64 as-is.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

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
RUN mkdir -p models/onnx data/outputs benchmarks

# Example build (OpenCV DNN + JSON):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

CMD ["/bin/bash", "-lc", "echo Ready: python training, cmake build, or benchmark scripts."]
