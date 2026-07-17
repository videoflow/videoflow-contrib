# videoflow-contrib :: retinaface — RetinaFace face detector (MXNet + Cython NMS).
#
# A deployable worker image for the new distributed videoflow: it builds on
# videoflow-base (framework + broker client + py3.9 interpreter) and adds this
# component's dependencies and package. The worker entrypoint (python -m
# videoflow.worker) is inherited from the base.
#
# Prerequisite: build the base image once (from the videoflow repo root):
#   ./docker/build-images.sh            # produces videoflow-base:py3.9 (and others)
#
# Build this component image (context = this module directory):
#   docker build -t videoflow-contrib-retinaface:latest retinaface/
#
# Deploy: pass it as the flow's --image default, or pin it on a node
#   RetinafaceDetector(name='...', image='videoflow-contrib-retinaface:latest')
ARG BASE_IMAGE=videoflow-base:py3.9
FROM ${BASE_IMAGE}

WORKDIR /app

# C toolchain for the package's Cython extensions (bbox/anchors/cpu_nms). This is a
# CPU image: the gpu_nms extension is skipped automatically when CUDA is absent. For a
# GPU build, base FROM a CUDA image that also has videoflow installed and mxnet-cuXXX.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Build/runtime deps. Cython + NumPy (NumPy comes from videoflow-base) must be present
# before building the extensions in the next step.
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# The videoflow_contrib.retinaface package; installing it compiles the Cython
# extensions. --no-deps: videoflow itself is already present in the base image.
COPY . ./
RUN uv pip install --system --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
