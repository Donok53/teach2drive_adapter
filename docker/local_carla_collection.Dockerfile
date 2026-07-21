FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libasound2 \
    libdbus-1-3 \
    libfontconfig1 \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnss3 \
    libomp5 \
    libpulse0 \
    libsm6 \
    libvulkan1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon-x11-0 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    mesa-utils \
    procps \
    psmisc \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    rsync \
    tini \
    tzdata \
    unzip \
    vulkan-tools \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade "pip<24" "setuptools<68" wheel && \
    python3 -m pip install --no-cache-dir \
      "dictor==0.1.12" \
      "diskcache==5.4.0" \
      "ephem==4.1.5" \
      "filterpy==1.4.5" \
      "gym==0.17.2" \
      "h5py==3.7.0" \
      "jsonpickle==3.0.3" \
      "laspy==2.5.4" \
      "lazrs==0.6.1" \
      "lxml==5.1.0" \
      "networkx==2.8.8" \
      "numpy==1.21.6" \
      "omegaconf==2.3.0" \
      "opencv-python-headless==4.6.0.66" \
      "pexpect==4.9.0" \
      "Pillow==10.2.0" \
      "psutil==5.9.8" \
      "py-trees==0.8.3" \
      "pygame==2.6.0" \
      "rdp==0.8" \
      "requests==2.31.0" \
      "scipy==1.10.1" \
      "Shapely==1.8.5.post1" \
      "simple-watchdog-timer==0.1.1" \
      "six==1.16.0" \
      "tabulate==0.9.0" \
      "tqdm==4.66.1" \
      "transforms3d==0.4.1" \
      "ujson==5.9.0" \
      "xmlschema==1.0.18"

WORKDIR /home/byeongjae/code/teach2drive_adapter
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
