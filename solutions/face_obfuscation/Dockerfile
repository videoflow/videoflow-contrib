# videoflow-contrib :: solutions/face_obfuscation — detect + track + blur faces.
#
# This solution's graph (face_obfuscation.py:build_flow) wires together nodes from
# videoflow_contrib.detector_tf (TensorFlow) and videoflow_contrib.tracker_sort, plus
# a couple of local nodes defined in the script itself. All node classes must be
# importable in the worker, so this image installs both sub-packages AND the script
# module (importable as `face_obfuscation`, which is how its local nodes appear in
# VF_NODE_CLASS). Python is pinned to 3.11 because of the TensorFlow-2 (Keras-2) node.
#
# Build from the videoflow-contrib repo ROOT (context must see the sub-packages):
#   docker build -f solutions/face_obfuscation/Dockerfile -t videoflow-contrib-face-obfuscation:latest .
#
# Deploy (input path comes from VF_INPUT_VIDEO at runtime):
#   videoflow deploy face_obfuscation.py:build_flow --nats nats://... \
#       --image videoflow-contrib-face-obfuscation:latest
ARG BASE_IMAGE=videoflow-base:py3.11
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

# The solution graph module, importable as `face_obfuscation` so the worker can build
# its local node classes (FrameIndexSplitter, BoundingboxObfuscator).
COPY solutions/face_obfuscation/face_obfuscation.py ./

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
