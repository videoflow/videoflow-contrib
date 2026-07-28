# videoflow-contrib :: solutions/video_captioning — VLM captions → subtitle track — CPU.
#
# This solution's graph (video_captioning.py:build_flow) wires the captioner from
# videoflow_contrib.vlm_caption (Hugging Face transformers) to the glue nodes in
# video_captioning_nodes.py. All node classes must be importable in the worker, so this
# image installs the sub-package AND the solution modules (glue nodes appear in
# VF_NODE_CLASS as `video_captioning_nodes.<Class>`). Python 3.12.
#
# The CPU variant is genuinely slow — a 7B VLM on CPU is minutes per frame. It exists
# so the pipeline is testable without an accelerator; use gpu.Dockerfile for real runs.
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-package):
#   docker build -f solutions/video_captioning/Dockerfile -t videoflow-contrib-video-captioning:latest .
# A GPU build is available: docker build -f solutions/video_captioning/gpu.Dockerfile -t videoflow-contrib-video-captioning:gpu .
#
# Normally you don't build this by hand — `videoflow deploy video_captioning.py`
# builds it (picking gpu.Dockerfile when a GPU is available), loads it into the
# cluster, and runs the flow. See solutions/video_captioning/README.md.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app

# 1. CPU PyTorch, before anything that depends on torch — PyPI's default linux
#    wheel bundles CUDA and would add ~2.5GB this image cannot use.
RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch>=2.4' 'torchvision>=0.19'

# 2. The Hugging Face stack (resolves against the torch installed above).
COPY solutions/video_captioning/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# 3. The contrib sub-package this solution's graph imports. --no-deps: videoflow is
#    already in the base image, and torch is deliberately the CPU build from step 1.
COPY vlm_caption /src/vlm_caption
RUN uv pip install --system --no-cache --no-deps /src/vlm_caption

# 4. The solution modules: the graph, the importable glue nodes the worker
#    reconstructs, the config loader, and the prep hook deploy runs.
COPY solutions/video_captioning/video_captioning.py \
     solutions/video_captioning/video_captioning_nodes.py \
     solutions/video_captioning/common.py \
     solutions/video_captioning/prepare.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
