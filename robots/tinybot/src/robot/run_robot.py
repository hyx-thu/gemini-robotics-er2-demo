import os
import sys
import uvicorn

# Add current directory to path to ensure robot_api can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the TinyBot FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    parser.add_argument(
        "--device-name",
        "-d",
        "--device",
        type=str,
        default=os.environ.get("TINYBOT_DEVICE", os.environ.get("ROBOT_DEVICE_NAME", "/dev/ttyUSB0")),
        help="Serial port device path for TinyBot (e.g., /dev/ttyUSB0)",
    )
    args, _ = parser.parse_known_args()

    os.environ["TINYBOT_DEVICE"] = args.device_name

    # Run the FastAPI server
    # We use the module name relative to the sys.path
    uvicorn.run("robot_api:app", host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
