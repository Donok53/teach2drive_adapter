#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-teach2drive/carla-collection:ubuntu20.04-cuda11.8}

docker build \
  -f docker/local_carla_collection.Dockerfile \
  -t "$IMAGE" \
  .

echo "built $IMAGE"
