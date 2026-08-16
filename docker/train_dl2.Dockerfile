FROM perception:torch-2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/data/users/byeongjae/code/teach2drive_adapter:/data/users/byeongjae/code/carla_garage/team_code

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libglib2.0-0 \
      libgl1 \
      tini \
    && rm -rf /var/lib/apt/lists/*

# Keep the numerical and vision stack aligned with the TF++ training environment.
# Torch/CUDA are supplied by the DL2 base image and are intentionally not reinstalled.
# The base image ships a newer OpenCV wheel that requires NumPy 2, so remove it first.
RUN python -m pip uninstall -y opencv-python opencv-python-headless || true && \
    python -m pip install --no-cache-dir \
      "numpy==1.26.4" \
      "opencv-python-headless==4.8.1.78" \
      "timm==1.0.11" \
      "einops==0.8.0" \
      "Pillow==10.2.0" \
      "shapely==2.0.4" \
      "jsonpickle==3.0.3" \
      "laspy==2.5.4" \
      "lazrs==0.6.1"

WORKDIR /data/users/byeongjae/code/teach2drive_adapter

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
