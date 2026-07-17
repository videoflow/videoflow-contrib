# videoflow-contrib :: soccer_detector — RF-DETR soccer detector (GPU/CUDA).
#
# Installs CUDA torch from the PyTorch index first so rfdetr's torch requirement is
# satisfied by the GPU build (not the default CPU wheel), then the component.
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app
RUN uv pip install --system --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache '.[gpu]'

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
