# 基础镜像, 使用 Conda 管理 Python 环境通常只需要基于 CUDA 镜像, 
# 在镜像配置中心 - 所有镜像 中搜索镜像名称（lab.cuda.devel）, 找到需要的版本, 复制镜像名称即可
FROM hub.byted.org/base/lab.cuda.devel:11.3.1-1

# C is for computer, sometime more standardized (but still in English)
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ARG https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ARG http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ARG no_proxy="byted.org"


# Mambaforge ref to official dockerfile:
# https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
ARG CONDA_NAME=Mambaforge
ARG CONDA_SITE="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease"
# 镜像中设置了新的环境变量 CONDA_DIR 来指定安装位置
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
# Conda 清华源
ARG CONDA_CONFIG="\
auto_activate_base: false\n\n\
channels:\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
"

RUN echo fafafafafafafafafafafafafafa && pwd
COPY ./environment.yaml /root/environment.yml


# apt-get 安装所需要的包
# apt-get some deps, python-dev is now neccessery for tasks in arnold
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        python-dev \
        ca-certificates \
        tini \
        git \
        wget \
        curl \
        zip \
        bzip2 \
        unzip \
        vim \
        zsh \
        tmux \
        exa \
        htop \
        && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install OpenMPI
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        cmake \
        libopenmpi-dev \
        openmpi-bin \
        openssh-client \
        openssh-server \
        && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Configure OpenSSH
RUN mkdir -p /var/run/sshd && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Configure Tmux
RUN mkdir -p /root/.config/tmux && \
    echo 'set -g mouse on\nset -g default-terminal "screen-256color"' >> /root/.config/tmux/tmux.conf

# Conda setup for user root and tiger
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" | tee -a /root/.bashrc /home/tiger/.bashrc && \
    echo ${CONDA_CONFIG} | tee /root/.condarc /home/tiger/.condarc

RUN wget --no-hsts --quiet ${CONDA_SITE}/${CONDA_NAME}-Linux-$(uname -m).sh -O /tmp/${CONDA_NAME}.sh && \
    /bin/bash /tmp/${CONDA_NAME}.sh -b -p ${CONDA_DIR} && \
    rm /tmp/${CONDA_NAME}.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes

RUN printf [easy_install]\\nindex_url=https://bytedpypi.byted.org/simple > ~/.pydistutils.cfg

# set custom python env, here I use the name openmm
RUN /bin/bash -c "\
. ${CONDA_DIR}/etc/profile.d/conda.sh && \
mamba env create -f /root/environment.yml && \
conda run -n decompdiff --no-capture-output python -m pip install -e git+https://github.com/subhadarship/kmeans_pytorch.git#egg=kmeans_pytorch && \
conda run -n decompdiff --no-capture-output python -m pip install scikit-learn-extra meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 && \
conda install -y -n decompdiff rdkit -c rdkit && \
conda install -y -n decompdiff -c openbabel openbabel && \
conda run -n decompdiff --no-capture-output python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3 && \
conda run -n decompdiff --no-capture-output python -m pip install numpy==1.22.3 && \
conda install -y -n decompdiff -c mx reduce && \
conda clean --tarballs --index-cache --packages --yes && \
conda clean --force-pkgs-dirs --all --yes\
"
