#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-teach2drive/carla-collection:ubuntu20.04-cuda11.8}
NAME=${NAME:-teach2drive-carla-collection}
DISPLAY_VALUE=${DISPLAY:-:1}
DOCKER_USER=${DOCKER_USER:-root}
VIDEO_GID=$(getent group video | cut -d: -f3 || true)
RENDER_GID=$(getent group render | cut -d: -f3 || true)
XAUTHORITY_VALUE=${XAUTHORITY:-}
XDG_RUNTIME_DIR_VALUE=${CONTAINER_XDG_RUNTIME_DIR:-/tmp/runtime-byeongjae}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  IMAGE="$IMAGE" scripts/build_local_carla_collection_image.sh
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local: >/dev/null 2>&1 || true
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker_args=(
  run
  --name "$NAME"
  --rm
  --network host
  --ipc host
  --shm-size 16g
  --ulimit stack=67108864
  --group-add "${VIDEO_GID:-44}"
  --group-add "${RENDER_GID:-109}"
  --device /dev/nvidiactl
  --device /dev/nvidia0
  --device /dev/nvidia1
  --device /dev/nvidia-uvm
  --device /dev/nvidia-uvm-tools
  --device /dev/nvidia-modeset
  --device /dev/dri
  -e "HOME=/home/byeongjae"
  -e "USER=byeongjae"
  -e "DISPLAY=${DISPLAY_VALUE}"
  -e "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR_VALUE}"
  -e "NVIDIA_DRIVER_CAPABILITIES=all"
  -e "NVIDIA_RUNTIME_ROOT=/host"
  -e "CARLA_GLX_VENDOR=nvidia"
  -e "LD_LIBRARY_PATH=/host/usr/lib/x86_64-linux-gnu:/host/usr/lib/x86_64-linux-gnu/nvidia/xorg"
  -e "VK_ICD_FILENAMES=/host/usr/share/vulkan/icd.d/nvidia_icd.json"
  -e "__EGL_VENDOR_LIBRARY_FILENAMES=/host/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
  -e "__GLX_VENDOR_LIBRARY_NAME=nvidia"
  -e "QT_X11_NO_MITSHM=1"
  -v /home/byeongjae:/home/byeongjae
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  -v /usr/lib/x86_64-linux-gnu:/host/usr/lib/x86_64-linux-gnu:ro
  -v /usr/share/vulkan:/host/usr/share/vulkan:ro
  -v /usr/share/glvnd:/host/usr/share/glvnd:ro
  -v /etc/localtime:/etc/localtime:ro
  -w /home/byeongjae/code/teach2drive_adapter
)

if [[ "$DOCKER_USER" != "root" ]]; then
  docker_args+=(--user "$DOCKER_USER")
fi

if [[ -x /usr/bin/nvidia-smi ]]; then
  docker_args+=(-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro)
fi

if [[ -n "$XAUTHORITY_VALUE" && -f "$XAUTHORITY_VALUE" ]]; then
  docker_args+=(-e "XAUTHORITY=$XAUTHORITY_VALUE" -v "$XAUTHORITY_VALUE:$XAUTHORITY_VALUE:ro")
fi

docker_args+=("$IMAGE")

if [[ "$#" -gt 0 ]]; then
  docker_args+=("$@")
else
  docker_args+=(bash)
fi

exec docker "${docker_args[@]}"
