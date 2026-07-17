# videoflow-contrib :: solutions/human_tracking — pose + encode + track people — GPU.
#
# GPU variant (Python 3.12 + CUDA 12.4). Builds on videoflow-base:py3.12-cuda (a CUDA
# -devel image, so nvcc is present to compile detectron2's CUDA kernels). The pose
# (detectron2/PyTorch) and encoder (tensorflow[and-cuda]) nodes run on the GPU; deepsort
# is CPU. Schedule the GPU nodes' pods onto GPU hosts (nvidia.com/gpu + NVIDIA runtime).
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/human_tracking/gpu.Dockerfile -t videoflow-contrib-human-tracking:gpu .
#
# Deploy:
#   videoflow deploy human_tracking.py:build_flow --nats nats://... \
#       --image videoflow-contrib-human-tracking:gpu
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app

# Toolchain + git for the detectron2 source build (nvcc comes from the -devel CUDA base).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 1. CUDA 12.4 PyTorch (must precede the detectron2 build).
RUN uv pip install --system --break-system-packages --no-cache \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch>=2.4' 'torchvision>=0.19'

# 2. CUDA tensorflow (humanencoder) + scipy (deepsort).
COPY solutions/human_tracking/requirements-gpu.txt ./requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

# 3. Build detectron2 from source with CUDA ops enabled.
ENV FORCE_CUDA=1
RUN uv pip install --system --break-system-packages --no-cache \
        'git+https://github.com/facebookresearch/detectron2.git'

# 4. The contrib sub-packages the graph imports. --no-deps: videoflow is already in base.
COPY detectron2 /src/detectron2
COPY tracker_deepsort /src/tracker_deepsort
COPY humanencoder /src/humanencoder
RUN uv pip install --system --break-system-packages --no-cache --no-deps \
        /src/detectron2 /src/tracker_deepsort /src/humanencoder

# 5. The solution graph module, importable as `human_tracking`.
COPY solutions/human_tracking/human_tracking.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
