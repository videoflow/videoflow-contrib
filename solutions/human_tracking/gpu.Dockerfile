# videoflow-contrib :: solutions/human_tracking — pose + encode + track people — GPU.
#
# GPU variant (Python 3.12 + CUDA 12.4). Builds on videoflow-base:py3.12-cuda (a CUDA
# -devel image, so nvcc is present to compile detectron2's CUDA kernels). The pose
# (detectron2/PyTorch) and encoder (tensorflow[and-cuda]) nodes run on the GPU when the
# config sets `device: gpu`; deepsort is always CPU. Deploy schedules the GPU nodes'
# pods onto GPU hosts (nvidia.com/gpu + NVIDIA runtime).
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/human_tracking/gpu.Dockerfile -t videoflow-contrib-human-tracking:gpu .
#
# Normally built for you: `videoflow deploy human_tracking.py` picks this file
# when the local docker daemon has the NVIDIA runtime.
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

# tensorflow pins protobuf<5, which silently downgrades the base image's protobuf
# below the >=5.27 the generated videoflow.v1 wire modules require (runtime_version)
# — every worker in the image then dies at import. Restore the core floor;
# tensorflow runs fine against the newer runtime.
RUN uv pip install --system --break-system-packages --no-cache 'protobuf>=5.27'

# 3. Build detectron2 from source with CUDA ops enabled. Its setup.py imports
#    torch, so build isolation must be off (and setuptools/wheel must already be
#    in the system env).
ENV FORCE_CUDA=1
RUN uv pip install --system --break-system-packages --no-cache setuptools wheel && \
    uv pip install --system --break-system-packages --no-cache --no-build-isolation \
        'git+https://github.com/facebookresearch/detectron2.git'

# 4. The contrib sub-packages the graph imports. --no-deps: videoflow is already in base.
COPY detectron2 /src/detectron2
COPY tracker_deepsort /src/tracker_deepsort
COPY humanencoder /src/humanencoder
RUN uv pip install --system --break-system-packages --no-cache --no-deps \
        /src/detectron2 /src/tracker_deepsort /src/humanencoder

# 5. The solution modules: the graph, the importable glue nodes the worker
#    reconstructs, the config loader, and the prep hook deploy runs.
COPY solutions/human_tracking/human_tracking.py \
     solutions/human_tracking/human_tracking_nodes.py \
     solutions/human_tracking/common.py \
     solutions/human_tracking/prepare.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
