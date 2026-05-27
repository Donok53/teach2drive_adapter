#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One-command target-domain paired collection:
#   1. reuse CARLA on HOST:PORT if it is already ready
#   2. otherwise start CARLA from CARLA_ROOT
#   3. collect Town13 vehicle-B paired tfpp_ego/front_triplet_shifted data

export PY=${PY:-"$HOME/.venv/carla37/bin/python"}
export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_town13_collect.log"}
export CARLA_READY_TIMEOUT_SEC=${CARLA_READY_TIMEOUT_SEC:-180}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHON_EGG_CACHE=${PYTHON_EGG_CACHE:-"$HOME/.cache/python-eggs-carla37"}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-"/tmp/runtime-$USER"}
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$(dirname "$CARLA_LOG")" "$PYTHON_EGG_CACHE" "$XDG_RUNTIME_DIR" "$HOME/.local/bin"
chmod 700 "$PYTHON_EGG_CACHE" 2>/dev/null || true
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

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

if check_carla >/dev/null 2>&1; then
  echo "=== CARLA already ready on $HOST:$PORT"
else
  echo "=== start CARLA host=$HOST port=$PORT"
  (
    cd "$CARLA_ROOT"
    nohup env PATH="$PATH" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" SDL_VIDEODRIVER=offscreen \
      ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="$PORT" > "$CARLA_LOG" 2>&1 &
    echo "$!" > "$CARLA_LOG.pid"
  )
  CARLA_PID=$(cat "$CARLA_LOG.pid")
  echo "CARLA_PID=$CARLA_PID log=$CARLA_LOG"

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
fi

exec bash configs/collect_vehicle_b_town13_paired_tfpp_ego_front_triplet_3h.sh
