# videoflow-contrib :: solutions/human_tracking — pose + encode + track people — CPU.
#
# This solution's graph (human_tracking.py:build_flow) wires together nodes from
# videoflow_contrib.detectron2 (PyTorch), videoflow_contrib.tracker_deepsort and
# videoflow_contrib.humanencoder (TensorFlow), plus the glue nodes in
# human_tracking_nodes.py. All node classes must be importable in the worker, so this
# image installs the three sub-packages AND the solution modules (glue nodes appear in
# VF_NODE_CLASS as `human_tracking_nodes.<Class>`). Python 3.12.
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/human_tracking/Dockerfile -t videoflow-contrib-human-tracking:latest .
# A GPU build is available: docker build -f solutions/human_tracking/gpu.Dockerfile -t videoflow-contrib-human-tracking:gpu .
#
# Normally you don't build this by hand — `videoflow deploy human_tracking.py`
# (and `videoflow run-local`) build it, picking gpu.Dockerfile instead when the
# config's `device` is gpu (the template's x-gpu), and run the flow with it. See
# solutions/human_tracking/README.md.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app

# detectron2 is built from source, which needs a C/C++ toolchain and git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 1. CPU PyTorch (must precede the detectron2 build).
RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch>=2.2' 'torchvision>=0.17'

# 2. tensorflow (humanencoder) + scipy (deepsort).
COPY solutions/human_tracking/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# 3. Build detectron2 from source against the installed torch. Its setup.py
#    imports torch, so build isolation must be off (and the build backend's own
#    deps — setuptools/wheel — must already be in the system env).
RUN uv pip install --system --no-cache setuptools wheel && \
    uv pip install --system --no-cache --no-build-isolation \
        'git+https://github.com/facebookresearch/detectron2.git'

# 4. The contrib sub-packages the graph imports. --no-deps: videoflow is already in base.
COPY detectron2 /src/detectron2
COPY tracker_deepsort /src/tracker_deepsort
COPY humanencoder /src/humanencoder
RUN uv pip install --system --no-cache --no-deps \
        /src/detectron2 /src/tracker_deepsort /src/humanencoder

# 5. The solution modules: the graph, the importable glue nodes the worker
#    reconstructs, the config loader, and the prep hook deploy runs.
COPY solutions/human_tracking/human_tracking.py \
     solutions/human_tracking/human_tracking_nodes.py \
     solutions/human_tracking/common.py \
     solutions/human_tracking/prepare.py ./

# 6. Restore the floors the ML stack disturbed, last of all: tensorflow < 2.18
#    needs numpy 1 (the detectron2 install above pulls numpy 2 back in, and a
#    numpy-2 interpreter fails to import tensorflow's compiled modules), and
#    both tensorflow (which pins protobuf<5) and the generated videoflow.v1
#    wire modules (which need >=5.27) are served by the protobuf 5.x line —
#    protobuf 6 removed MessageFactory.GetPrototype, which tensorflow still calls.
RUN uv pip install --system --no-cache 'numpy<2' 'protobuf>=5.27,<6'

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
