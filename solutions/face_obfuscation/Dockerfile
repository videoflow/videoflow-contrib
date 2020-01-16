# To build: docker build -t detector_tf -f Dockerfile .
# To run: docker run -it detector_tf
FROM tensorflow/tensorflow:1.14.0-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN echo "deb http://old-releases.ubuntu.com/ubuntu/ yakkety universe" | tee -a /etc/apt/sources.list
RUN apt-get update && apt-get install -y \ 
 wget \
 git \
 pkg-config \
 ffmpeg \
 pkg-config \
 python-dev \ 
 python-opencv \ 
 libopencv-dev \ 
 libav-tools  \ 
 libjpeg-dev \ 
 libpng-dev \ 
 libtiff-dev \ 
 libjasper-dev \ 
 python-numpy \ 
 python-pycurl \ 
 python-opencv

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# Installing videoflow
RUN git clone https://github.com/videoflow/videoflow.git
RUN pip3 install --user /home/appuser/videoflow --find-links /home/appuser/videoflow

# Installing videoflow_contrib.detector_tf and tracker
RUN git clone https://github.com/videoflow/videoflow-contrib.git
RUN pip3 install --user /home/appuser/videoflow-contrib/detector_tf --find-links /home/appuser/videoflow-contrib/detector_tf
RUN pip3 install --user /home/appuser/videoflow-contrib/tracker_sort --find-links /home/appuser/videoflow-contrib/tracker_sort

# Command to run example
CMD ["python3", "/home/appuser/videoflow-contrib/solutions/face_obfuscation/face_obfuscation.py"]