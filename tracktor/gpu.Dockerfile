# To build: docker build -t tracking -f gpu.Dockerfile .
# To run: nvidia-docker run -v </path/to/output/folder>:/home/user/videos -d tracking
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

# INstall numpy and scipy
RUN pip3 install numpy==1.17.4 scipy==1.3.2 

# Installing videoflow
RUN git clone https://github.com/jadielam/videoflow.git
RUN pip3 install --user /home/user/videoflow --find-links /home/user/videoflow

# Installing videoflow_contrib packages
RUN git clone https://github.com/jadielam/videoflow-contrib.git
RUN pip3 install --user /home/user/videoflow-contrib/tracktor --find-links /home/user/videoflow-contrib/tracktor

# Copying and running scripts to track people
RUN mkdir /home/user/examples
RUN mkdir /home/user/videos
COPY --chown=user:sudo examples/ /home/user/examples/
RUN chmod u+x /home/user/examples/people_tracking.sh
CMD ["/home/user/examples/people_tracking.sh"]