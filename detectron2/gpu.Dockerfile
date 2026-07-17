# videoflow-contrib :: detectron2 — Detectron2 human-pose estimator (PyTorch) — GPU.
#
# GPU worker image for the new distributed videoflow (Python 3.12 + CUDA 12.4). Builds
# on videoflow-base:py3.12-cuda (a CUDA -devel image, so nvcc is present to compile
# detectron2's CUDA kernels). Schedule the pods onto GPU nodes (nvidia.com/gpu + the
# NVIDIA runtime).
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -f detectron2/gpu.Dockerfile -t videoflow-contrib-detectron2:gpu detectron2/
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app

# Toolchain + git for the source build (nvcc comes from the -devel CUDA base).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 1. CUDA 12.4 PyTorch first — detectron2's build imports torch and matches its CUDA.
RUN uv pip install --system --break-system-packages --no-cache \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch>=2.4' 'torchvision>=0.19'

# 2. Build detectron2 from source with CUDA ops enabled (FORCE_CUDA=1 so the CUDA
#    kernels are compiled even though no GPU is visible during the image build).
ENV FORCE_CUDA=1
RUN uv pip install --system --break-system-packages --no-cache \
        'git+https://github.com/facebookresearch/detectron2.git'

# 3. The videoflow_contrib.detectron2 package. --no-deps: videoflow is already in base.
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
