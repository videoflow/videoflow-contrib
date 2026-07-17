# videoflow-contrib :: annotators — OpenCV/NumPy object annotator (no ML framework).
#
# CPU worker image for the new distributed videoflow (Python 3.12). Builds on
# videoflow-base:py3.12 and adds this component's dependencies and package; the worker
# entrypoint (python -m videoflow.worker) is inherited from the base.
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -t videoflow-contrib-annotators:latest annotators/
#
# Deploy: pass it as the flow's --image default, or pin it on a node with image=.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app
# The videoflow_contrib.annotators package (importable by its module path, which is what
# appears in VF_NODE_CLASS). Its dependencies come from pyproject.toml; videoflow is already in the base image.
COPY . ./
RUN uv pip install --system --no-cache .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
