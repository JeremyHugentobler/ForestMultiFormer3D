FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST=7.0 \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Base deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip \
      build-essential git cmake ninja-build \
      ffmpeg libsm6 libxext6 libglib2.0-0 libxrender-dev \
      libopenblas-dev software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 1.13.1 + cu117 wheels
RUN python3 -m pip install \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --index-url https://download.pytorch.org/whl/cu117

# Dev tools
RUN python3 -m pip install debugpy

# OpenMMLab (pins)
RUN python3 -m pip install --no-deps \
      mmengine==0.7.3 mmdet==3.0.0 mmsegmentation==1.0.0 \
      git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61 \
 && python3 -m pip install --no-deps \
  mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# MinkowskiEngine (build with OpenBLAS + CUDA)
RUN python3 -m pip install -v --no-deps \
      git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
      --install-option="--blas=openblas" --install-option="--force_cuda"

# torch-scatter (manual build as you had)
RUN git clone https://github.com/rusty1s/pytorch_scatter.git /tmp/pytorch_scatter \
 && cd /tmp/pytorch_scatter \
 && git checkout tags/2.0.9 -b v2.0.9 \
 && FORCE_CUDA=1 python3 -m pip install . \
 && rm -rf /tmp/pytorch_scatter

# segmentator build
RUN git clone https://github.com/Karbo123/segmentator.git /segmentator \
    && cd /segmentator/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') \
        -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/libpython3.10.so')") \
        -DCMAKE_INSTALL_PREFIX=$(python3 -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
    && make \
    && make install

# Remaining Python deps (headless where possible)
RUN python3 -m pip install --no-deps \
      spconv-cu116==2.3.6 \
      addict==2.4.0 yapf==0.33.0 termcolor==2.3.0 packaging==23.1 \
      numpy==1.24.1 rich==13.3.5 opencv-python-headless==4.7.0.72 \
      pycocotools==2.0.6 Shapely==1.8.5 scipy==1.10.1 terminaltables==3.1.10 \
      numba==0.57.0 llvmlite==0.40.0 pccm==0.4.7 ccimport==0.4.2 pybind11==2.10.4 \
      ninja==1.11.1 lark==1.1.5 cumm-cu116==0.4.9 pyquaternion==0.9.9 \
      lyft-dataset-sdk==0.0.8 pandas==2.0.1 python-dateutil==2.8.2 \
      matplotlib==3.5.2 pyparsing==3.0.9 cycler==0.11.0 kiwisolver==1.4.4 \
      scikit-learn==1.2.2 joblib==1.2.0 threadpoolctl==3.1.0 cachetools==5.3.0 \
      nuscenes-devkit==1.1.10 trimesh==3.21.6 open3d==0.17.0 plotly==5.18.0 dash==2.14.2 \
      plyfile==1.0.2 flask==3.0.0 werkzeug==3.0.1 click==8.1.7 blinker==1.7.0 \
      itsdangerous==2.1.2 importlib_metadata==2.1.2 zipp==3.17.0 \
      tensorboard==2.15.1 tensorboard-data-server==0.7.2 protobuf absl-py future \
      MarkupSafe==2.0.1 markdown grpcio google-auth-oauthlib google-auth \
      requests-oauthlib oauthlib \
      jinja2==3.1.2 pyyaml pytz tqdm

# torch-points-kernels (pin)
RUN python3 -m pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0

# Prefer PyG prebuilt wheel for torch-cluster (avoid building on cluster)
RUN python3 -m pip install -U --no-cache-dir   -f https://data.pyg.org/whl/torch-1.13.1+cu117.html torch-cluster

ENV PYTHONPATH=/workspace

# Install tini (Debian/Ubuntu base) - best practice for PID 1 handling in containers 
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/* 

# Default fixing and installation scripts
COPY bootstrap.sh /usr/local/bin/bootstrap 
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/bootstrap"]
# Optional default CMD, can be overridden
CMD ["bash"]

