# To build: docker build -t detectron2 -f gpu.Dockerfile .
# To run: nvidia-docker run -it detectron2
FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    bzip2 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} user -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user
WORKDIR /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

 # CUDA 10.1-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0" \
    "torchvision=0.5.0=py36_cu101" \
 && conda clean -ya

 # Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

 # Installing pip3
ENV PATH="/home/user/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# Install detectron2, pointing it to an specific id, 
# since repo does not have tag as of December 18, 2019
RUN pip3 install --user 'git+https://github.com/cocodataset/cocoapi.git@636becdc73d54283b3aac6d4ec363cffbb6f9b20#subdirectory=PythonAPI'
RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore@8694adf300c4e47d575ad1583bfb9d646fe9c12c'
RUN pip3 install --user -U pillow==6.1

ENV FORCE_CUDA=1
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN cd detectron2_repo && git checkout feaa5028c540101c1fbc84e0daf9c36d15550f4a
# The line below targets all GPUs, but makes installation slower. If you know the exact
# GPU that you are targeting, feel free to modify line below.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Installing videoflow
RUN git clone https://github.com/videoflow/videoflow.git
RUN pip3 install --user /home/user/videoflow --find-links /home/user/videoflow

# Installing videoflow_contrib packages
RUN git clone https://github.com/videoflow/videoflow-contrib.git
RUN pip3 install --user /home/user/videoflow-contrib/detectron2 --find-links /home/user/videoflow-contrib/detectron2

# Command to run example here
CMD ["python3", "/home/user/videoflow-contrib/detectron2/examples/humanpose_example.py"]

