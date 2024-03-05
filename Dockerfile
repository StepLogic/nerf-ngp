ARG UBUNTU_VER=22.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.10
ARG PANDAS_VER=1.3

FROM ubuntu:${UBUNTU_VER}
LABEL authors="robotlab2"
# System packages
RUN apt-get update && apt-get install -yq curl wget jq vim git  build-essential

# Use the above args
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ARG PY_VER
#ARG PANDAS_VER
# Install packages from conda
RUN conda install -c anaconda -y python=${PY_VER}
#RUN conda install -c anaconda -y \
#    pandas=${PANDAS_VER}


WORKDIR /home
COPY . /nerf-ngp
WORKDIR /home/nerf-ngp
RUN conda create -n torch-ngp python=${PY_VER}
RUN echo "source activate torch-ngp" > ~/.bashrc
ENV PATH /opt/conda/envs/torch-ngp/bin:$PATH
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#RUN bash /home/nerf-gp/install_tiny_cuda.sh
#RUN #pip3 install -r torch-ngp/requirements.txt && bash torch-ngp/scripts/install_ext.sh


