#!/usr/bin/env bash
set -euo pipefail

# Build a small Python 3.7 environment for CARLA data collection.
#
# CARLA 0.9.15 server bundles PythonAPI artifacts for cp37/py3.7 on some
# servers, while the training environment may be Python 3.10.  Collection needs
# the real CARLA client module, so this script creates a separate py3.7 env and
# leaves the torch training env untouched.

ENV_PREFIX=${ENV_PREFIX:-"$HOME/.venv/carla37"}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-"$HOME/.micromamba"}
MICROMAMBA_BIN=${MICROMAMBA_BIN:-"$HOME/.local/bin/micromamba"}
MICROMAMBA_URL=${MICROMAMBA_URL:-"https://micro.mamba.pm/api/micromamba/linux-64/latest"}

mkdir -p "$(dirname "$MICROMAMBA_BIN")" "$MAMBA_ROOT_PREFIX"

if [[ ! -x "$MICROMAMBA_BIN" ]]; then
  echo "=== install micromamba -> $MICROMAMBA_BIN"
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT
  curl -Ls "$MICROMAMBA_URL" | tar -xvj -C "$tmpdir" bin/micromamba
  install -m 755 "$tmpdir/bin/micromamba" "$MICROMAMBA_BIN"
fi

echo "=== micromamba version"
"$MICROMAMBA_BIN" --version

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "=== create collection env -> $ENV_PREFIX"
  MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MICROMAMBA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge python=3.7 pip
else
  echo "=== reuse collection env -> $ENV_PREFIX"
fi

echo "=== install collection python deps"
"$ENV_PREFIX/bin/python" -m pip install --upgrade "pip<24" "setuptools<68" wheel
"$ENV_PREFIX/bin/python" -m pip install "numpy==1.21.6" "opencv-python-headless==4.5.5.64"

echo "=== ready"
echo "PY=$ENV_PREFIX/bin/python"
