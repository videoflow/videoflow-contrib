# videoflow-contrib :: solutions/human_tracking — pose + encode + track people.
#
# This solution's graph (human_tracking.py:build_flow) wires together nodes from
# videoflow_contrib.detectron2 (PyTorch), videoflow_contrib.tracker_deepsort and
# videoflow_contrib.humanencoder (TensorFlow), plus several local nodes defined in the
# script. All node classes must be importable in the worker, so this image installs the
# three sub-packages AND the script module (importable as `human_tracking`). Python is
# pinned to 3.11 (detectron2 + TensorFlow-2 Keras-2).
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/human_tracking/Dockerfile -t videoflow-contrib-human-tracking:latest .
#
# Deploy:
#   videoflow deploy human_tracking.py:build_flow --nats nats://... \
#       --image videoflow-contrib-human-tracking:latest
ARG BASE_IMAGE=videoflow-base:py3.11
FROM ${BASE_IMAGE}

WORKDIR /app

# detectron2 is built from source, which needs a C/C++ toolchain and git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 1. torch/torchvision + tensorflow + scipy (torch must precede the detectron2 build).
COPY solutions/human_tracking/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# 2. Build detectron2 from source against the installed torch. This is a CPU image;
#    for CUDA, base FROM a CUDA image and install matching torch+cuXXX wheels.
RUN uv pip install --system --no-cache \
        'git+https://github.com/facebookresearch/detectron2.git'

# 3. The contrib sub-packages the graph imports. --no-deps: videoflow is already in base.
COPY detectron2 /src/detectron2
COPY tracker_deepsort /src/tracker_deepsort
COPY humanencoder /src/humanencoder
RUN uv pip install --system --no-cache --no-deps \
        /src/detectron2 /src/tracker_deepsort /src/humanencoder

# 4. The solution graph module, importable as `human_tracking` so the worker can build
#    its local node classes (FrameIndexSplitter, KeypointsExtractor, ...).
COPY solutions/human_tracking/human_tracking.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
