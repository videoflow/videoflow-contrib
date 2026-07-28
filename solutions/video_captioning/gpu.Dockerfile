# videoflow-contrib :: solutions/video_captioning — VLM captions → subtitle track — GPU.
#
# GPU variant of the video-captioning solution (Python 3.12 + CUDA). The captioner node
# (videoflow_contrib.vlm_caption) loads the model with device_map='auto' and Hugging
# Face shards it across every visible device — which, by the RFC 0003 visibility
# contract, is exactly the `captioner.gpu_count` devices the pod was granted. Deploy
# schedules its pods onto GPU nodes (nvidia.com/gpu + NVIDIA runtime).
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-package):
#   docker build -f solutions/video_captioning/gpu.Dockerfile -t videoflow-contrib-video-captioning:gpu .
#
# Normally built for you: `videoflow deploy video_captioning.py` picks this file
# when the local docker daemon has the NVIDIA runtime.
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app

# 1. CUDA PyTorch from the PyTorch index, before anything that depends on torch,
#    so transformers/accelerate resolve against the GPU build rather than pulling
#    the default CPU wheel over it. Mirrors vlm_caption/gpu.Dockerfile.
RUN uv pip install --system --break-system-packages --no-cache \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. The Hugging Face stack (resolves against the CUDA torch installed above).
COPY solutions/video_captioning/requirements-gpu.txt ./requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

# 3. The contrib sub-package this solution's graph imports. --no-deps: videoflow is
#    already in the base image, and torch is deliberately the cu124 build from step 1.
COPY vlm_caption /src/vlm_caption
RUN uv pip install --system --break-system-packages --no-cache --no-deps /src/vlm_caption

# 4. The solution modules: the graph, the importable glue nodes the worker
#    reconstructs, the config loader, and the prep hook deploy runs.
COPY solutions/video_captioning/video_captioning.py \
     solutions/video_captioning/video_captioning_nodes.py \
     solutions/video_captioning/common.py \
     solutions/video_captioning/prepare.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
