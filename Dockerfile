FROM nvidia/cuda:10.2-runtime AS jupyter-base
WORKDIR /
# Install Python and its tools
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    unzip 
    
RUN pip3 -q install pip --upgrade
# Install all basic packages
RUN pip3 install \
    # Jupyter itself
    jupyter \
    # jupyter lab
    jupyterlab \
    # Numpy and Pandas are required a-priori
    numpy pandas \
    # PyTorch with CUDA 10.2 support and Torchvision
    torch torchvision torchinfo\
    # Upgraded version of Tensorboard with more features
    tensorboardX \
    # Matplotlib
    matplotlib \
    # kaggle \
    kaggle


# Here we use a base image by its name - "jupyter-base"
FROM jupyter-base
# Install additional packages
RUN pip3 install \
    # Hugging Face Transformers
    transformers \
    # Progress bar to track experiments
    barbar