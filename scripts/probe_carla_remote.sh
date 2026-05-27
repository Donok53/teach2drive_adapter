#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-2017}
HOST=${HOST:-127.0.0.1}
CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_probe.log"}
PY=${PY:-"$HOME/.venv/carla37/bin/python"}
RUNTIME=${NVIDIA_RUNTIME_ROOT:-"$HOME/.local/nvidia-runtime"}
READY_TIMEOUT_SEC=${READY_TIMEOUT_SEC:-90}
CARLA_RENDER_ARGS=${CARLA_RENDER_ARGS:-"-RenderOffScreen"}
CARLA_SDL_VIDEODRIVER=${CARLA_SDL_VIDEODRIVER:-offscreen}
CARLA_DISPLAY=${CARLA_DISPLAY:-}
CARLA_EXTRA_ARGS=${CARLA_EXTRA_ARGS:-"-stdout -FullStdOutLogOutput"}

LIBDIR="$RUNTIME/usr/lib/x86_64-linux-gnu"
ICD="$RUNTIME/usr/share/vulkan/icd.d/nvidia_icd.json"
LAYERS="$RUNTIME/usr/share/vulkan/implicit_layer.d"
XDG_DIR=${XDG_RUNTIME_DIR:-"/tmp/runtime-$USER"}

mkdir -p "$(dirname "$CARLA_LOG")" "$HOME/.local/bin" "$XDG_DIR" "$HOME/.cache/python-eggs-carla37"
chmod 700 "$XDG_DIR" "$HOME/.cache/python-eggs-carla37" 2>/dev/null || true

if ! command -v xdg-user-dir >/dev/null 2>&1; then
  {
    printf '%s\n' '#!/usr/bin/env sh'
    printf '%s\n' 'case "${1:-}" in'
    printf '%s\n' '  DESKTOP) echo "$HOME/Desktop" ;;'
    printf '%s\n' '  DOWNLOAD) echo "$HOME/Downloads" ;;'
    printf '%s\n' '  DOCUMENTS) echo "$HOME/Documents" ;;'
    printf '%s\n' '  MUSIC) echo "$HOME/Music" ;;'
    printf '%s\n' '  PICTURES) echo "$HOME/Pictures" ;;'
    printf '%s\n' '  VIDEOS) echo "$HOME/Videos" ;;'
    printf '%s\n' '  *) echo "$HOME" ;;'
    printf '%s\n' 'esac'
  } > "$HOME/.local/bin/xdg-user-dir"
  chmod +x "$HOME/.local/bin/xdg-user-dir"
fi

if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
  echo "missing CARLA executable: $CARLA_ROOT/CarlaUE4.sh" >&2
  exit 2
fi
if [[ ! -f "$ICD" || ! -d "$LIBDIR" ]]; then
  echo "missing NVIDIA runtime; expected $ICD and $LIBDIR" >&2
  exit 3
fi

"$PY" -c 'import carla; print("carla import ok")'

if [[ -f "$CARLA_LOG.pid" ]]; then
  old_pid=$(cat "$CARLA_LOG.pid" 2>/dev/null || true)
  if [[ -n "${old_pid:-}" ]]; then
    kill "$old_pid" 2>/dev/null || true
  fi
fi

rm -f "$CARLA_LOG" "$CARLA_LOG.pid"
cd "$CARLA_ROOT"

echo "start CARLA port=$PORT log=$CARLA_LOG"
carla_env=(
  "PATH=$HOME/.local/bin:$PATH"
  "XDG_RUNTIME_DIR=$XDG_DIR"
  "LD_LIBRARY_PATH=$LIBDIR:${LD_LIBRARY_PATH:-}"
  "VK_ICD_FILENAMES=$ICD"
  "VK_LAYER_PATH=$LAYERS"
  "__GLX_VENDOR_LIBRARY_NAME=nvidia"
  "__NV_PRIME_RENDER_OFFLOAD=1"
)
if [[ -n "$CARLA_SDL_VIDEODRIVER" ]]; then
  carla_env+=("SDL_VIDEODRIVER=$CARLA_SDL_VIDEODRIVER")
fi
if [[ -n "$CARLA_DISPLAY" ]]; then
  carla_env+=("DISPLAY=$CARLA_DISPLAY")
fi

setsid env \
  "${carla_env[@]}" \
  ./CarlaUE4.sh \
    $CARLA_RENDER_ARGS \
    -nosound \
    -quality-level=Low \
    -carla-rpc-port="$PORT" \
    $CARLA_EXTRA_ARGS \
  > "$CARLA_LOG" 2>&1 < /dev/null &
carla_pid=$!
echo "$carla_pid" > "$CARLA_LOG.pid"
echo "pid=$carla_pid"

for i in $(seq 1 "$READY_TIMEOUT_SEC"); do
  if ! kill -0 "$carla_pid" 2>/dev/null; then
    echo "CARLA died after ${i}s"
    tail -200 "$CARLA_LOG" || true
    exit 4
  fi

  if "$PY" -c "import carla; c=carla.Client('$HOST', $PORT); c.set_timeout(2.0); w=c.get_world(); print(w.get_map().name)" >/tmp/carla_probe_ok.txt 2>/tmp/carla_probe_err.txt; then
    echo "CARLA ready after ${i}s"
    cat /tmp/carla_probe_ok.txt
    exit 0
  fi
  sleep 1
done

echo "CARLA not ready after ${READY_TIMEOUT_SEC}s"
tail -200 "$CARLA_LOG" || true
cat /tmp/carla_probe_err.txt || true
exit 5
