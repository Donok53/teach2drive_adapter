#!/usr/bin/env python3
"""Keep the CARLA spectator camera following the hero vehicle."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


def _install_carla_path(carla_root: str | None) -> None:
    if not carla_root:
        return
    api_root = Path(carla_root).expanduser() / "PythonAPI" / "carla"
    if api_root.exists() and str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))


def _find_hero(world):
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    return vehicles[0] if vehicles else None


def _follow_transform(carla, vehicle_transform, view: str):
    loc = vehicle_transform.location
    rot = vehicle_transform.rotation
    yaw = math.radians(rot.yaw)

    if view == "topdown":
        return carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 45.0),
            carla.Rotation(pitch=-90.0, yaw=rot.yaw),
        )

    if view == "front":
        distance = 8.0
        x = loc.x + distance * math.cos(yaw)
        y = loc.y + distance * math.sin(yaw)
        return carla.Transform(
            carla.Location(x=x, y=y, z=loc.z + 3.0),
            carla.Rotation(pitch=-15.0, yaw=rot.yaw + 180.0),
        )

    distance = 9.0
    x = loc.x - distance * math.cos(yaw)
    y = loc.y - distance * math.sin(yaw)
    return carla.Transform(
        carla.Location(x=x, y=y, z=loc.z + 4.0),
        carla.Rotation(pitch=-18.0, yaw=rot.yaw),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--carla-root", default="")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--view", choices=["chase", "topdown", "front"], default="chase")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    args = parser.parse_args()

    _install_carla_path(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    spectator = world.get_spectator()

    print(f"Waiting for hero vehicle on {args.host}:{args.port}...")
    hero = None
    start = time.monotonic()
    while hero is None:
        hero = _find_hero(world)
        if hero is not None:
            break
        if args.wait_timeout > 0 and time.monotonic() - start > args.wait_timeout:
            raise TimeoutError("Timed out waiting for a hero vehicle")
        time.sleep(0.2)

    print(f"Following actor {hero.id} ({hero.type_id}) with {args.view} view.")
    sleep_sec = 1.0 / max(args.hz, 1e-3)
    try:
        while hero.is_alive:
            spectator.set_transform(_follow_transform(carla, hero.get_transform(), args.view))
            time.sleep(sleep_sec)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
