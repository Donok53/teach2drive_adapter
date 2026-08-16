FROM teach2drive-adapter:dl2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libjpeg-dev \
      libpng-dev \
      xdg-user-dirs \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -y -n tfpp-eval python=3.10 pip \
    && /opt/conda/envs/tfpp-eval/bin/python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu118 \
      "torch==2.1.2" \
      "torchvision==0.16.2" \
    && /opt/conda/envs/tfpp-eval/bin/python -m pip install --no-cache-dir \
      "numpy==1.26.4" \
      "opencv-python==4.8.1.78" \
      "dictor==0.1.12" \
      "diskcache==5.4.0" \
      "einops==0.4.1" \
      "ephem==4.1.5" \
      "filterpy==1.4.5" \
      "h5py==3.10.0" \
      "imgaug==0.4.0" \
      "jsonpickle==3.0.3" \
      "laspy==2.5.4" \
      "lxml==5.1.0" \
      "networkx==3.4.2" \
      "omegaconf==2.3.0" \
      "pexpect==4.9.0" \
      "Pillow==10.2.0" \
      "psutil==5.9.8" \
      "py-trees==0.8.3" \
      "pygame==2.6.1" \
      "pytictoc==1.5.3" \
      "rdp==0.8" \
      "requests==2.31.0" \
      "scikit-image==0.25.2" \
      "scikit-learn==1.7.2" \
      "scipy==1.15.3" \
      "simple-watchdog-timer" \
      "six==1.16.0" \
      "tabulate==0.9.0" \
      "tensorboardX==2.6.2.2" \
      "timm==1.0.11" \
      "transforms3d==0.4.1" \
      "ujson==5.9.0" \
      "webcolors==1.13" \
      "xmlschema==1.0.18"

ENV PATH=/opt/conda/envs/tfpp-eval/bin:${PATH}

# CARLA leaderboard 0.9.15 still imports pkg_resources.
RUN /opt/conda/envs/tfpp-eval/bin/python -m pip install --no-cache-dir "setuptools==80.9.0"
