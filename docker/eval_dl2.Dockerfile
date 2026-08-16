FROM teach2drive-adapter:dl2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libjpeg-dev \
      libpng-dev \
      xdg-user-dirs \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
      "numpy==1.26.4" \
      "opencv-python==4.8.1.78" \
      "carla==0.9.16" \
      "dictor==0.1.12" \
      "diskcache==5.4.0" \
      "ephem==4.1.5" \
      "filterpy==1.4.5" \
      "imgaug==0.4.0" \
      "lxml==5.1.0" \
      "omegaconf==2.3.0" \
      "psutil==5.9.8" \
      "py-trees==0.8.3" \
      "pygame==2.6.0" \
      "pytictoc==1.5.3" \
      "rdp==0.8" \
      "simple-watchdog-timer" \
      "tabulate==0.9.0" \
      "tensorboardX==2.6.2.2" \
      "transformers==4.46.3" \
      "transforms3d==0.4.1" \
      "ujson==5.9.0" \
      "webcolors==1.13" \
      "xmlschema==1.0.18"
