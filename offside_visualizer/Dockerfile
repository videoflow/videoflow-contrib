# videoflow-contrib :: offside_visualizer — offside verdict renderer (CPU).
#
# Note: this consumer re-opens the original camera videos from local disk, so the
# recordings must be mounted into the container (a shared volume on K8s).
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . ./
RUN uv pip install --system --no-cache .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
