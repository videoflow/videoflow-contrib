# videoflow-contrib :: tracker_botsort — BoxMOT appearance-aware tracker (GPU/CUDA).
#
# CUDA torch first (satisfies boxmot's torch requirement with the GPU build), then
# the component.
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app
RUN uv pip install --system --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache '.[gpu]'

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
