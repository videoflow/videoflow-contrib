# videoflow-contrib :: detectron2 — Detectron2 human-pose estimator (PyTorch).
#
# A deployable worker image for the new distributed videoflow: it builds on
# videoflow-base (framework + broker client + py3.11 interpreter) and adds this
# component's dependencies and package. The worker entrypoint (python -m
# videoflow.worker) is inherited from the base.
#
# Prerequisite: build the base image once (from the videoflow repo root):
#   ./docker/build-images.sh            # produces videoflow-base:py3.11 (and others)
#
# Build this component image (context = this module directory):
#   docker build -t videoflow-contrib-detectron2:latest detectron2/
#
# Deploy: pass it as the flow's --image default, or pin it on a node
#   Detectron2HumanPose(name='...', image='videoflow-contrib-detectron2:latest')
ARG BASE_IMAGE=videoflow-base:py3.11
FROM ${BASE_IMAGE}

WORKDIR /app

# Detectron2 is compiled from source, which needs a C/C++ toolchain and git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 1. Install torch/torchvision FIRST — detectron2's build imports torch.
#    (This is a CPU image. For CUDA, base FROM a CUDA image and install the matching
#     torch+cuXXX wheels and detectron2 build.)
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# 2. Build detectron2 from source against the installed torch. Pin a ref for
#    reproducibility if you need it (e.g. @v0.6 or a commit sha).
RUN uv pip install --system --no-cache \
        'git+https://github.com/facebookresearch/detectron2.git'

# 3. The videoflow_contrib.detectron2 package (ships the model configs as package
#    data). --no-deps: videoflow itself is already present in the base image.
COPY . ./
RUN uv pip install --system --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
