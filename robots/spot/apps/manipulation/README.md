# Manipulation API

This package provides Python APIs for Spot arm manipulation:

- deploy the arm to ready position
- stow the arm
- open the gripper
- detect a language-described object with Gemini Robotics ER
- project a 2D grasp point into a 3D pose using Spot depth
- command Spot's manipulation service to pick from the selected image pixel
- detect external force changes at the end effector

Default aligned image sources:

- color: `hand_color_image`
- depth: `hand_depth_in_hand_color_frame`

Alternative lower-resolution aligned pair:

- color: `hand_color_in_hand_depth_frame`
- depth: `hand_depth`

Set credentials:

```bash
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=password
export GEMINI_API_KEY=...
```

Examples:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m apps.manipulation.cli deploy-arm --hostname 192.168.80.3 --take-lease
UV_CACHE_DIR=.uv-cache uv run python -m apps.manipulation.cli open-gripper --hostname 192.168.80.3 --take-lease
UV_CACHE_DIR=.uv-cache uv run python -m apps.manipulation.cli detect --hostname 192.168.80.3 "the red cup"
UV_CACHE_DIR=.uv-cache uv run python -m apps.manipulation.cli pick --hostname 192.168.80.3 "the red cup" --take-lease
UV_CACHE_DIR=.uv-cache uv run python -m apps.manipulation.cli stow-arm --hostname 192.168.80.3 --take-lease
```
