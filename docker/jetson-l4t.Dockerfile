# Jetson (arm64) oriented Dockerfile — pin the nvcr.io tag to your JetPack BSP.
# Flash the device with NVIDIA SDK Manager, then build on-device or via QEMU.
FROM nvcr.io/nvidia/l4t-base:r35.3.1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    cmake build-essential \
    libopencv-dev libopencv-contrib-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models/onnx data/outputs benchmarks

# Prefer JetPack’s prebuilt PyTorch wheel where available; requirements.txt may need adjustment per JP version.

CMD ["/bin/bash", "-lc", "echo Jetson workspace ready."]
