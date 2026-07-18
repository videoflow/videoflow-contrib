# videoflow-contrib :: solutions/face_obfuscation — detect + track + blur faces — CPU.
#
# This solution's graph (face_obfuscation.py:build_flow) wires together nodes from
# videoflow_contrib.detector_tf (TensorFlow) and videoflow_contrib.tracker_sort, plus
# the glue nodes in face_obfuscation_nodes.py. All node classes must be importable in
# the worker, so this image installs both sub-packages AND the solution modules
# (glue nodes appear in VF_NODE_CLASS as `face_obfuscation_nodes.<Class>`). Python 3.12.
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/face_obfuscation/Dockerfile -t videoflow-contrib-face-obfuscation:latest .
# A GPU build is available: docker build -f solutions/face_obfuscation/gpu.Dockerfile -t videoflow-contrib-face-obfuscation:gpu .
#
# Normally you don't build this by hand — `videoflow deploy face_obfuscation.py`
# builds it (picking gpu.Dockerfile when a GPU is available), loads it into the
# cluster, and runs the flow. See solutions/face_obfuscation/README.md.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app

# Solution dependencies (on top of videoflow-base's built-in node deps).
COPY solutions/face_obfuscation/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# The contrib sub-packages this solution's graph imports. --no-deps: videoflow is
# already in the base image.
COPY detector_tf /src/detector_tf
COPY tracker_sort /src/tracker_sort
RUN uv pip install --system --no-cache --no-deps /src/detector_tf /src/tracker_sort

# The solution modules: the graph, the importable glue nodes the worker
# reconstructs, the config loader, and the prep hook deploy runs.
COPY solutions/face_obfuscation/face_obfuscation.py \
     solutions/face_obfuscation/face_obfuscation_nodes.py \
     solutions/face_obfuscation/common.py \
     solutions/face_obfuscation/prepare.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
