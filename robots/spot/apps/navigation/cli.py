from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from apps.navigation.registry import DEFAULT_REGISTRY_PATH, KeypointRegistry
from apps.navigation.spot_client import SpotNavigationClient
from apps.navigation.visualizer import DEFAULT_MAP_PATH, write_graph_html


def _registry(args: argparse.Namespace) -> KeypointRegistry:
    return KeypointRegistry(Path(args.registry))


def _robot(args: argparse.Namespace) -> SpotNavigationClient:
    username = args.username or os.getenv("BOSDYN_CLIENT_USERNAME")
    password = args.password or os.getenv("BOSDYN_CLIENT_PASSWORD")
    if not username:
        username = input("Spot username: ")
    if not password:
        password = getpass.getpass("Spot password: ")
    return SpotNavigationClient(args.hostname, username, password)


def _print_keypoints(registry: KeypointRegistry) -> None:
    keypoints = registry.all()
    if not keypoints:
        print("No keypoints registered.")
        return
    name_width = max(len(k.name) for k in keypoints)
    for keypoint in keypoints:
        suffix = f"  # {keypoint.note}" if keypoint.note else ""
        print(f"{keypoint.name:<{name_width}}  {keypoint.waypoint_id}{suffix}")


def list_keypoints(args: argparse.Namespace) -> int:
    _print_keypoints(_registry(args))
    return 0


def register_keypoint(args: argparse.Namespace) -> int:
    keypoint = _registry(args).register(args.name, args.waypoint_id, args.note, args.overwrite)
    print(f"Registered {keypoint.name}: {keypoint.waypoint_id}")
    return 0


def remove_keypoint(args: argparse.Namespace) -> int:
    keypoint = _registry(args).remove(args.name)
    print(f"Removed {keypoint.name}: {keypoint.waypoint_id}")
    return 0


def rename_keypoint(args: argparse.Namespace) -> int:
    keypoint = _registry(args).rename(args.old_name, args.new_name, args.overwrite)
    print(f"Renamed to {keypoint.name}: {keypoint.waypoint_id}")
    return 0


def sync_keypoints(args: argparse.Namespace) -> int:
    robot = _robot(args)
    registry = _registry(args)
    synced = registry.sync_from_waypoints(robot.named_waypoints(), overwrite=args.overwrite)
    if not synced:
        print("No new named waypoints found.")
        return 0
    for keypoint in synced:
        print(f"Synced {keypoint.name}: {keypoint.waypoint_id}")
    return 0


def go_to_keypoint(args: argparse.Namespace) -> int:
    keypoint = _registry(args).get(args.name)
    result = _robot(args).navigate_to_waypoint(
        keypoint.waypoint_id,
        command_duration=args.command_duration,
        timeout=args.timeout,
        feedback_interval=args.feedback_interval,
        power_on=args.power_on,
        stand=args.stand,
        acquire_lease=not args.no_lease,
        take_lease=args.take_lease,
    )
    print(f"{keypoint.name}: {result.status_name} (command_id={result.command_id})")
    return 0 if result.reached_goal else 2


def visualize_keypoints(args: argparse.Namespace) -> int:
    robot = _robot(args)
    graph = robot.download_graph()
    snapshots = {}
    if not args.no_point_clouds:
        snapshot_ids = sorted({waypoint.snapshot_id for waypoint in graph.waypoints if waypoint.snapshot_id})
        for index, snapshot_id in enumerate(snapshot_ids, start=1):
            print(f"Downloading point cloud snapshot {index}/{len(snapshot_ids)}", end="\r")
            snapshots[snapshot_id] = robot.download_waypoint_snapshot(snapshot_id)
        if snapshot_ids:
            print()
    output = write_graph_html(graph, _registry(args), Path(args.output), snapshots=snapshots)
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage named Spot GraphNav keypoints and navigate by name.")
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to the JSON keypoint registry.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List saved keypoints.")
    list_parser.set_defaults(func=list_keypoints)

    register_parser = subparsers.add_parser("register", help="Register a name for a GraphNav waypoint ID.")
    register_parser.add_argument("name")
    register_parser.add_argument("waypoint_id")
    register_parser.add_argument("--note", default="")
    register_parser.add_argument("--overwrite", action="store_true")
    register_parser.set_defaults(func=register_keypoint)

    remove_parser = subparsers.add_parser("remove", help="Remove a saved keypoint.")
    remove_parser.add_argument("name")
    remove_parser.set_defaults(func=remove_keypoint)

    rename_parser = subparsers.add_parser("rename", help="Rename a saved keypoint.")
    rename_parser.add_argument("old_name")
    rename_parser.add_argument("new_name")
    rename_parser.add_argument("--overwrite", action="store_true")
    rename_parser.set_defaults(func=rename_keypoint)

    sync_parser = subparsers.add_parser("sync", help="Import named waypoints from Spot's current GraphNav map.")
    _add_robot_args(sync_parser)
    sync_parser.add_argument("--overwrite", action="store_true")
    sync_parser.set_defaults(func=sync_keypoints)

    go_parser = subparsers.add_parser("go", help="Navigate Spot to a saved keypoint by name.")
    _add_robot_args(go_parser)
    go_parser.add_argument("name")
    go_parser.add_argument("--command-duration", type=float, default=30.0)
    go_parser.add_argument("--timeout", type=float, default=120.0)
    go_parser.add_argument("--feedback-interval", type=float, default=1.0)
    go_parser.add_argument("--power-on", action="store_true")
    go_parser.add_argument("--stand", action="store_true")
    go_parser.add_argument("--no-lease", action="store_true", help="Do not acquire/keep a body lease.")
    go_parser.add_argument(
        "--take-lease",
        action="store_true",
        help="Forcefully take the body lease from another client before navigating.",
    )
    go_parser.set_defaults(func=go_to_keypoint)

    visualize_parser = subparsers.add_parser("visualize", help="Export an HTML map of the current GraphNav waypoints.")
    _add_robot_args(visualize_parser)
    visualize_parser.add_argument("--output", default=str(DEFAULT_MAP_PATH), help="Path for the generated HTML map.")
    visualize_parser.add_argument("--no-point-clouds", action="store_true", help="Render only waypoints and edges.")
    visualize_parser.set_defaults(func=visualize_keypoints)

    return parser


def _add_robot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hostname", required=True, help="Spot hostname or IP address.")
    parser.add_argument("--username", help="Spot username. Defaults to BOSDYN_CLIENT_USERNAME.")
    parser.add_argument("--password", help="Spot password. Defaults to BOSDYN_CLIENT_PASSWORD.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
