# videoflow-contrib :: detectron2 — Detectron2 human-pose estimator (PyTorch) — CPU.
#
# CPU worker image for the new distributed videoflow (Python 3.12). Builds on
# videoflow-base:py3.12, installs CPU PyTorch, then compiles detectron2 from source.
# The worker entrypoint (python -m videoflow.worker) is inherited from the base.
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -t videoflow-contrib-detectron2:latest detectron2/
# A GPU build is available: docker build -f detectron2/gpu.Dockerfile -t videoflow-contrib-detectron2:gpu detectron2/
#
# Deploy: pass it as the flow's --image default, or pin it on a node with image=.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app

# detectron2 compiles C++ extensions from source, which need a toolchain and git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 1. CPU PyTorch first — detectron2's build imports torch.
RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch>=2.2' 'torchvision>=0.17'

# 2. Build detectron2 from source (CPU) against the installed torch. Pin a ref for
#    reproducibility if a future main branch breaks against your torch (e.g. a sha).
RUN uv pip install --system --no-cache \
        'git+https://github.com/facebookresearch/detectron2.git'

# 3. The videoflow_contrib.detectron2 package (ships the model configs as package
#    data). --no-deps: videoflow is already present in the base image.
COPY . ./
RUN uv pip install --system --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
