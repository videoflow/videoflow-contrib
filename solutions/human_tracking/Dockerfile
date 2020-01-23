# To build: docker build -t human_tracking -f Dockerfile .
# To run: nvidia-docker run -it human_tracking
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

# Installing pip3
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# Installing videoflow
RUN git clone https://github.com/videoflow/videoflow.git
RUN pip3 install --user /home/appuser/videoflow --find-links /home/appuser/videoflow

# Install Pytorch
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip3 install --user torch==1.3 torchvision==0.4 tensorboard==1.14 cython==0.29
RUN pip3 install --user 'git+https://github.com/cocodataset/cocoapi.git@636becdc73d54283b3aac6d4ec363cffbb6f9b20#subdirectory=PythonAPI'
RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore@8694adf300c4e47d575ad1583bfb9d646fe9c12c'
RUN pip3 install --user -U pillow==6.1

# Install detectron2, pointing it to an specific id, 
# since repo does not have tag as of December 18, 2019
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN cd detectron2_repo && git checkout feaa5028c540101c1fbc84e0daf9c36d15550f4a
RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Installing videoflow_contrib packages
RUN ls -lha
RUN git clone https://github.com/videoflow/videoflow-contrib.git
RUN pip3 install --user /home/appuser/videoflow-contrib/detectron2 --find-links /home/appuser/videoflow-contrib/detector_tf
RUN pip3 install --user /home/appuser/videoflow-contrib/tracker_deepsort --find-links /home/appuser/videoflow-contrib/tracker_sort
RUN pip3 install --user /home/appuser/videoflow-contrib/humanencoder --find-links /home/appuser/videoflow-contrib/humanencoder

# Command to run example
#CMD ["python3", "/home/appuser/videoflow-contrib/solutions/human_tracking/human_tracking.py"]