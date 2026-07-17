# videoflow-contrib :: solutions/offside — FIFA-style offside detection (GPU/CUDA).
#
# Build from the videoflow-contrib repo ROOT:
#   docker build -f solutions/offside/gpu.Dockerfile -t videoflow-contrib-offside:gpu .
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 1. CUDA PyTorch + the GPU ML stack (onnxruntime-gpu).
RUN uv pip install --system --break-system-packages --no-cache \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124
COPY solutions/offside/requirements-gpu.txt ./
RUN uv pip install --system --break-system-packages --no-cache -r requirements-gpu.txt

# 2. The nine contrib sub-packages (--no-deps).
COPY synced_video_reader /src/synced_video_reader
COPY soccer_detector /src/soccer_detector
COPY tracker_botsort /src/tracker_botsort
COPY pose_topdown /src/pose_topdown
COPY team_classifier /src/team_classifier
COPY pitch_calib /src/pitch_calib
COPY multiview_fuser /src/multiview_fuser
COPY offside_engine /src/offside_engine
COPY offside_visualizer /src/offside_visualizer
RUN uv pip install --system --break-system-packages --no-cache --no-deps \
        /src/synced_video_reader /src/soccer_detector /src/tracker_botsort \
        /src/pose_topdown /src/team_classifier /src/pitch_calib \
        /src/multiview_fuser /src/offside_engine /src/offside_visualizer

# 3. The solution modules.
COPY solutions/offside/offside.py solutions/offside/offside_nodes.py solutions/offside/common.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
