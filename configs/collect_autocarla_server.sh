#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One-command CARLA collection wrapper:
#   1. reuse CARLA on HOST:PORT if it is already ready
#   2. otherwise start CARLA from CARLA_ROOT
#   3. run COLLECT_CONFIG

export PY=${PY:-"$HOME/.venv/carla37/bin/python"}
export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_collect.log"}
export CARLA_READY_TIMEOUT_SEC=${CARLA_READY_TIMEOUT_SEC:-180}
export CARLA_EXTRA_ARGS=${CARLA_EXTRA_ARGS:-"-stdout -FullStdOutLogOutput"}
export FORCE_RESTART_CARLA=${FORCE_RESTART_CARLA:-0}
export CARLA_PRELOAD_MAP=${CARLA_PRELOAD_MAP:-0}
export CARLA_MAP_LOAD_TIMEOUT_SEC=${CARLA_MAP_LOAD_TIMEOUT_SEC:-180}
export CARLA_CLIENT_TIMEOUT_SEC=${CARLA_CLIENT_TIMEOUT_SEC:-20}
export CARLA_MAP_RETRIES=${CARLA_MAP_RETRIES:-3}
export CARLA_POST_MAP_SETTLE_SEC=${CARLA_POST_MAP_SETTLE_SEC:-8}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHON_EGG_CACHE=${PYTHON_EGG_CACHE:-"$HOME/.cache/python-eggs-carla37"}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-"/tmp/runtime-$USER"}
export NVIDIA_DEB_DIR=${NVIDIA_DEB_DIR:-"$HOME/nvidia-debs"}
export NVIDIA_RUNTIME_ROOT=${NVIDIA_RUNTIME_ROOT:-"/"}
export CARLA_SDL_VIDEODRIVER=${CARLA_SDL_VIDEODRIVER:-x11}
export CARLA_DISPLAY=${CARLA_DISPLAY:-:1}
export CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-0}
export CARLA_GLX_VENDOR=${CARLA_GLX_VENDOR:-}
export CARLA_NV_PRIME_RENDER_OFFLOAD=${CARLA_NV_PRIME_RENDER_OFFLOAD:-}
export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_tesla_front_triplet_target_3h.sh}
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$(dirname "$CARLA_LOG")" "$PYTHON_EGG_CACHE" "$XDG_RUNTIME_DIR" "$HOME/.local/bin"
chmod 700 "$PYTHON_EGG_CACHE" 2>/dev/null || true
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
mkdir -p "$HOME/Desktop" "$HOME/Downloads" "$HOME/Documents" "$HOME/Music" "$HOME/Pictures" "$HOME/Videos"

if ! command -v xdg-user-dir >/dev/null 2>&1; then
  cat > "$HOME/.local/bin/xdg-user-dir" <<'SH'
#!/usr/bin/env sh
case "${1:-}" in
  DESKTOP) echo "$HOME/Desktop" ;;
  DOWNLOAD) echo "$HOME/Downloads" ;;
  DOCUMENTS) echo "$HOME/Documents" ;;
  MUSIC) echo "$HOME/Music" ;;
  PICTURES) echo "$HOME/Pictures" ;;
  VIDEOS) echo "$HOME/Videos" ;;
  *) echo "$HOME" ;;
esac
SH
  chmod +x "$HOME/.local/bin/xdg-user-dir"
fi

setup_nvidia_runtime() {
  local libdir="$NVIDIA_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
  local icd="$NVIDIA_RUNTIME_ROOT/usr/share/vulkan/icd.d/nvidia_icd.json"
  local layers="$NVIDIA_RUNTIME_ROOT/usr/share/vulkan/implicit_layer.d"
  local marker="$NVIDIA_RUNTIME_ROOT/.teach2drive_nvidia_runtime_unpacked"
  local found_deb=0

  if [[ "$NVIDIA_RUNTIME_ROOT" != "/" && ! -f "$marker" && -d "$NVIDIA_DEB_DIR" ]]; then
    mkdir -p "$NVIDIA_RUNTIME_ROOT"
    echo "=== unpack NVIDIA runtime from $NVIDIA_DEB_DIR -> $NVIDIA_RUNTIME_ROOT"
    for deb in \
      "$NVIDIA_DEB_DIR"/libnvidia-compute-*.deb \
      "$NVIDIA_DEB_DIR"/libnvidia-gpucomp-*.deb \
      "$NVIDIA_DEB_DIR"/libnvidia-gl-*.deb
    do
      if [[ -f "$deb" ]]; then
        found_deb=1
        echo "unpack $(basename "$deb")"
        dpkg-deb -x "$deb" "$NVIDIA_RUNTIME_ROOT"
      fi
    done
    if [[ "$found_deb" == 1 ]]; then
      touch "$marker"
    fi
  fi

  if [[ -f "$icd" && -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
    export VK_ICD_FILENAMES="$icd"
    if [[ -d "$layers" ]]; then
      export VK_LAYER_PATH="$layers"
    fi
    if [[ -n "$CARLA_GLX_VENDOR" ]]; then
      export __GLX_VENDOR_LIBRARY_NAME="$CARLA_GLX_VENDOR"
    else
      unset __GLX_VENDOR_LIBRARY_NAME || true
    fi
    if [[ -n "$CARLA_NV_PRIME_RENDER_OFFLOAD" ]]; then
      export __NV_PRIME_RENDER_OFFLOAD="$CARLA_NV_PRIME_RENDER_OFFLOAD"
    else
      unset __NV_PRIME_RENDER_OFFLOAD || true
    fi
    echo "=== NVIDIA runtime configured"
    echo "LD_LIBRARY_PATH=$libdir:..."
    echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
  else
    echo "=== WARNING: NVIDIA Vulkan runtime not configured"
    echo "NVIDIA_DEB_DIR=$NVIDIA_DEB_DIR"
    echo "expected_icd=$icd"
  fi
}

setup_nvidia_runtime

check_carla() {
  "$PY" - "$HOST" "$PORT" <<'PY'
import sys
import carla

host = sys.argv[1]
port = int(sys.argv[2])
client = carla.Client(host, port)
client.set_timeout(3.0)
client.get_world()
PY
}

kill_carla_on_port() {
  local pids=()
  local pid
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(pgrep -f "carla-rpc-port=${PORT}" 2>/dev/null || true)
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(pgrep -f "carla-rpc-port ${PORT}" 2>/dev/null || true)

  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "=== stopping CARLA on port $PORT: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 3
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

start_carla() {
  echo "=== start CARLA host=$HOST port=$PORT"
  (
    cd "$CARLA_ROOT"
    carla_env=(
      "PATH=$PATH"
      "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
    )
    [[ -n "${CARLA_SDL_VIDEODRIVER:-}" ]] && carla_env+=("SDL_VIDEODRIVER=$CARLA_SDL_VIDEODRIVER")
    [[ -n "${CARLA_DISPLAY:-}" ]] && carla_env+=("DISPLAY=$CARLA_DISPLAY")
    [[ -n "${LD_LIBRARY_PATH:-}" ]] && carla_env+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
    [[ -n "${VK_ICD_FILENAMES:-}" ]] && carla_env+=("VK_ICD_FILENAMES=$VK_ICD_FILENAMES")
    [[ -n "${VK_LAYER_PATH:-}" ]] && carla_env+=("VK_LAYER_PATH=$VK_LAYER_PATH")
    [[ -n "${__GLX_VENDOR_LIBRARY_NAME:-}" ]] && carla_env+=("__GLX_VENDOR_LIBRARY_NAME=$__GLX_VENDOR_LIBRARY_NAME")
    [[ -n "${__NV_PRIME_RENDER_OFFLOAD:-}" ]] && carla_env+=("__NV_PRIME_RENDER_OFFLOAD=$__NV_PRIME_RENDER_OFFLOAD")
    nohup env "${carla_env[@]}" \
      ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -graphicsadapter="$CARLA_GRAPHICS_ADAPTER" -carla-rpc-port="$PORT" $CARLA_EXTRA_ARGS > "$CARLA_LOG" 2>&1 &
    echo "$!" > "$CARLA_LOG.pid"
  )
  CARLA_PID=$(cat "$CARLA_LOG.pid")
  echo "CARLA_PID=$CARLA_PID log=$CARLA_LOG"
}

wait_carla_ready() {
  for i in $(seq 1 "$CARLA_READY_TIMEOUT_SEC"); do
    if check_carla >/dev/null 2>&1; then
      echo "=== CARLA ready after ${i}s"
      break
    fi
    if ! kill -0 "$CARLA_PID" 2>/dev/null; then
      echo "=== CARLA died during startup"
      tail -120 "$CARLA_LOG" || true
      exit 1
    fi
    if [[ "$i" == "$CARLA_READY_TIMEOUT_SEC" ]]; then
      echo "=== CARLA not ready after ${CARLA_READY_TIMEOUT_SEC}s"
      tail -120 "$CARLA_LOG" || true
      exit 1
    fi
    sleep 1
  done
}

preload_carla_map() {
  if [[ "$CARLA_PRELOAD_MAP" != "1" || -z "${MAP:-}" ]]; then
    return 0
  fi

  echo "=== ensure CARLA map=$MAP before collector"
  timeout "$CARLA_MAP_LOAD_TIMEOUT_SEC" "$PY" - "$HOST" "$PORT" "$MAP" "$CARLA_POST_MAP_SETTLE_SEC" "$CARLA_CLIENT_TIMEOUT_SEC" <<'PY'
import sys
import time
import carla

host = sys.argv[1]
port = int(sys.argv[2])
expected = str(sys.argv[3]).split("/")[-1]

client = carla.Client(host, port)
client.set_timeout(float(sys.argv[5]))
world = client.get_world()
current = world.get_map().name
if expected not in current:
    print(f"load_world start expected={expected} current={current}", flush=True)
    world = client.load_world(expected)
    time.sleep(2.0)

world = client.get_world()
current = world.get_map().name
print(f"map_ready current={current}", flush=True)
if expected not in current:
    raise SystemExit(f"expected map containing {expected!r}, got {current!r}")

settle_sec = float(sys.argv[4])
if settle_sec > 0:
    time.sleep(settle_sec)
    world = client.get_world()
    current = world.get_map().name
    print(f"map_still_ready current={current}", flush=True)
    if expected not in current:
        raise SystemExit(f"expected map containing {expected!r} after settle, got {current!r}")
PY
}

if [[ "$FORCE_RESTART_CARLA" == "1" || "$FORCE_RESTART_CARLA" == "true" || "$FORCE_RESTART_CARLA" == "TRUE" ]]; then
  kill_carla_on_port
fi

map_attempts=1
if [[ "$CARLA_PRELOAD_MAP" == "1" ]]; then
  map_attempts="$CARLA_MAP_RETRIES"
fi

map_ready=0
for attempt in $(seq 1 "$map_attempts"); do
  if check_carla >/dev/null 2>&1; then
    echo "=== CARLA already ready on $HOST:$PORT"
  else
    start_carla
    wait_carla_ready
  fi

  if preload_carla_map; then
    map_ready=1
    break
  fi

  echo "=== CARLA map preload failed attempt=$attempt/$map_attempts map=${MAP:-}"
  tail -80 "$CARLA_LOG" || true
  kill_carla_on_port
  sleep 3
done

if [[ "$map_ready" != "1" ]]; then
  echo "=== CARLA map preload did not succeed after $map_attempts attempt(s)"
  exit 1
fi

exec bash "$COLLECT_CONFIG"
