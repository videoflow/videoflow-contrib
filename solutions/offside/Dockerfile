# videoflow-contrib :: solutions/offside — FIFA-style offside detection (CPU).
#
# The graph (offside.py:build_flow) wires nodes from nine videoflow_contrib
# sub-packages plus local glue nodes defined in the script. All node classes must be
# importable in the worker, so this image installs the sub-packages AND the script
# module. Python 3.12.
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/offside/Dockerfile -t videoflow-contrib-offside:latest .
# GPU: docker build -f solutions/offside/gpu.Dockerfile -t videoflow-contrib-offside:gpu .
#
# Deploy:
#   videoflow deploy offside.py:build_flow --nats nats://... --image videoflow-contrib-offside:latest
# The visualizer re-opens the camera videos from disk — mount them into the container.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app

# ffmpeg for audio-based sync (sync_offsets.py).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 1. CPU PyTorch (rfdetr / boxmot / ultralytics need it), then the ML stack.
RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2' 'torchvision>=0.17'
COPY solutions/offside/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# 2. The nine contrib sub-packages the graph imports. --no-deps: videoflow is in base
#    and the ML stack is already resolved above.
COPY synced_video_reader /src/synced_video_reader
COPY soccer_detector /src/soccer_detector
COPY tracker_botsort /src/tracker_botsort
COPY pose_topdown /src/pose_topdown
COPY team_classifier /src/team_classifier
COPY pitch_calib /src/pitch_calib
COPY multiview_fuser /src/multiview_fuser
COPY offside_engine /src/offside_engine
COPY offside_visualizer /src/offside_visualizer
RUN uv pip install --system --no-cache --no-deps \
        /src/synced_video_reader /src/soccer_detector /src/tracker_botsort \
        /src/pose_topdown /src/team_classifier /src/pitch_calib \
        /src/multiview_fuser /src/offside_engine /src/offside_visualizer

# 3. The solution modules, importable so the worker can build the local glue nodes.
COPY solutions/offside/offside.py solutions/offside/offside_nodes.py \
     solutions/offside/common.py solutions/offside/prepare.py \
     solutions/offside/sync_offsets.py solutions/offside/calibrate.py \
     solutions/offside/fit_teams.py solutions/offside/download_weights.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
