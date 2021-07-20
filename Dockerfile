ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"


RUN apt-get update && apt-get install -y \
    python-setuptools \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    locales \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./packagelist.txt ./
RUN conda config --append channels conda-forge
RUN conda install --file packagelist.txt

WORKDIR /workspaces/humseg

RUN mkdir humseg
COPY ./humseg ./humseg
COPY ./setup.py ./
RUN pip install setuptools -U && pip install -e .
