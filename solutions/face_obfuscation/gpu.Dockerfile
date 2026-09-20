# videoflow-contrib :: solutions/face_obfuscation — detect + track + blur faces — GPU.
#
# GPU variant of the face-obfuscation solution (Python 3.12 + CUDA). The detector node
# (videoflow_contrib.detector_tf) runs on the GPU via tensorflow[and-cuda] when the
# config sets `device: gpu`; the tracker is always CPU. Deploy schedules the detector's
# pods onto GPU nodes (nvidia.com/gpu + NVIDIA runtime).
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/face_obfuscation/gpu.Dockerfile -t videoflow-contrib-face-obfuscation:gpu .
#
# Normally built for you: `videoflow deploy face_obfuscation.py` (and `run-local`)
# pick this file when the config's `device` is gpu (the template's x-gpu).
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app

# Solution dependencies with CUDA TensorFlow.
COPY solutions/face_obfuscation/requirements-gpu.txt ./requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

# tensorflow pins protobuf<5, which silently downgrades the base image's protobuf
# below the >=5.27 the generated videoflow.v1 wire modules require (runtime_version)
# — every worker in the image then dies at import. Restore the core floor, but
# below 6: protobuf 6 removed MessageFactory.GetPrototype, which tensorflow
# < 2.18 still calls (the 5.x line keeps it and satisfies videoflow.v1).
RUN uv pip install --system --break-system-packages --no-cache 'protobuf>=5.27,<6'

# The contrib sub-packages this solution's graph imports. --no-deps: videoflow is
# already in the base image.
COPY detector_tf /src/detector_tf
COPY tracker_sort /src/tracker_sort
RUN uv pip install --system --break-system-packages --no-cache --no-deps \
        /src/detector_tf /src/tracker_sort

# The solution modules: the graph, the importable glue nodes the worker
# reconstructs, the config loader, and the prep hook deploy runs.
COPY solutions/face_obfuscation/face_obfuscation.py \
     solutions/face_obfuscation/face_obfuscation_nodes.py \
     solutions/face_obfuscation/common.py \
     solutions/face_obfuscation/prepare.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
