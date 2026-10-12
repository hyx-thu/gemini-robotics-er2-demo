from __future__ import annotations

import argparse
import getpass
import os

from apps.manipulation.gemini_detector import GeminiObjectDetector
from apps.manipulation.spot_client import SpotManipulationClient


def _robot(args: argparse.Namespace) -> SpotManipulationClient:
    username = args.username or os.getenv("BOSDYN_CLIENT_USERNAME")
    password = args.password or os.getenv("BOSDYN_CLIENT_PASSWORD")
    if not username:
        username = input("Spot username: ")
    if not password:
        password = getpass.getpass("Spot password: ")
    return SpotManipulationClient(args.hostname, username, password)


def deploy(args: argparse.Namespace) -> int:
    command_id = _robot(args).deploy_arm(take_lease=args.take_lease, power_on=args.power_on)
    print(f"arm deployed command_id={command_id}")
    return 0


def stow(args: argparse.Namespace) -> int:
    command_id = _robot(args).stow_arm(take_lease=args.take_lease)
    print(f"arm stowed command_id={command_id}")
    return 0


def open_gripper(args: argparse.Namespace) -> int:
    command_id = _robot(args).open_gripper(take_lease=args.take_lease)
    print(f"gripper opened command_id={command_id}")
    return 0


def detect(args: argparse.Namespace) -> int:
    robot = _robot(args)
    detector = GeminiObjectDetector(model=args.model)
    detection, observation = robot.detect_object_2d(args.instruction, detector=detector)
    pose = robot.detection_to_3d_pose(detection, observation, frame_name=args.frame)
    print(detection)
    print(pose)
    return 0


def force(args: argparse.Namespace) -> int:
    result = _robot(args).detect_external_force_change(
        threshold_newtons=args.threshold,
        sample_window_sec=args.window,
    )
    print(result)
    return 0 if result.changed else 1


def pick(args: argparse.Namespace) -> int:
    robot = _robot(args)
    detector = GeminiObjectDetector(model=args.model)
    detection, observation = robot.detect_object_2d(args.instruction, detector=detector)
    pose = robot.detection_to_3d_pose(detection, observation)
    print(detection)
    print(pose)
    command_id = robot.pick_detection_in_image(
        detection,
        observation,
        take_lease=args.take_lease,
        timeout=args.timeout,
    )
    print(f"pick command_id={command_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spot manipulation APIs and CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, func in [
        ("deploy-arm", deploy),
        ("stow-arm", stow),
        ("open-gripper", open_gripper),
        ("detect", detect),
        ("force-change", force),
        ("pick", pick),
    ]:
        cmd = subparsers.add_parser(name)
        _add_robot_args(cmd)
        cmd.set_defaults(func=func)

    subparsers.choices["deploy-arm"].set_defaults(power_on=True)
    subparsers.choices["deploy-arm"].add_argument("--no-power-on", action="store_false", dest="power_on")

    for name in ["deploy-arm", "stow-arm", "open-gripper", "pick"]:
        subparsers.choices[name].add_argument("--take-lease", action="store_true")

    for name in ["detect", "pick"]:
        subparsers.choices[name].add_argument("instruction")
        subparsers.choices[name].add_argument("--model", default="gemini-robotics-er-1.6-preview")

    subparsers.choices["detect"].add_argument("--frame", default="vision")
    subparsers.choices["force-change"].add_argument("--threshold", type=float, default=5.0)
    subparsers.choices["force-change"].add_argument("--window", type=float, default=3.0)
    subparsers.choices["pick"].add_argument("--timeout", type=float, default=30.0)
    return parser


def _add_robot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--username", help="Defaults to BOSDYN_CLIENT_USERNAME.")
    parser.add_argument("--password", help="Defaults to BOSDYN_CLIENT_PASSWORD.")


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
