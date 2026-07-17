# videoflow-contrib :: pitch_calib (GPU) — pitch-landmark detector, CUDA.
#
# GPU worker image (Python 3.12 + CUDA) building on videoflow-base:py3.12-cuda.
# Schedule the resulting pods onto GPU nodes (nvidia.com/gpu + the NVIDIA runtime).
#
# Build (context = this module directory):
#   docker build -f pitch_calib/gpu.Dockerfile -t videoflow-contrib-pitch_calib:gpu pitch_calib/
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache '.[gpu]'

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
