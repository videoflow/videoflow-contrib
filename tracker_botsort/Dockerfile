# videoflow-contrib :: tracker_botsort — BoxMOT appearance-aware tracker (CPU).
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . ./
RUN uv pip install --system --no-cache .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
