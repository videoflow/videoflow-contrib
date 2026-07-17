# videoflow-contrib :: pose_topdown — rtmlib top-down pose (GPU, onnxruntime-gpu).
#
# The `.[gpu]` extra pulls onnxruntime-gpu so rtmlib uses the CUDA execution provider.
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache '.[gpu]'

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
