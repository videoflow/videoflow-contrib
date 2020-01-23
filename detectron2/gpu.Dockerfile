# To build: docker build -t detectron2 -f detectron.Dockerfile .
# To run: nvidia-docker run -it detectron2
FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo && \
  rm -rf /var/lib/apt/lists/*

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

# Install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip3 install --user torch==1.3 torchvision==0.4 tensorboard==1.14 cython==0.29
RUN pip3 install --user 'git+https://github.com/cocodataset/cocoapi.git@636becdc73d54283b3aac6d4ec363cffbb6f9b20#subdirectory=PythonAPI'
RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore@8694adf300c4e47d575ad1583bfb9d646fe9c12c'
RUN pip3 install --user -U pillow==6.1

# Install detectron2, pointing it to an specific id, 
# since repo does not have tag as of December 18, 2019
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN cd detectron2_repo && git checkout feaa5028c540101c1fbc84e0daf9c36d15550f4a
ENV FORCE_CUDA="1"
# The line below targets all GPUs, but makes installation slower. If you know the exact
# GPU that you are targeting, feel free to modify line below.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Installing videoflow
RUN git clone https://github.com/videoflow/videoflow.git
RUN pip3 install --user /home/appuser/videoflow --find-links /home/appuser/videoflow

# Installing videoflow_contrib.detectron2
RUN mkdir -p /home/appuser/videoflow_contrib/detectron2
COPY . /home/appuser/videoflow_contrib/detectron2
RUN pip3 install --user /home/appuser/videoflow_contrib/detectron2 --find-links /home/appuser/videoflow_contrib/detectron2

# Command to run example here
CMD ["python3", "/home/appuser/videoflow_contrib/detectron2/examples/example.py"]

