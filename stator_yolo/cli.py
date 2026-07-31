from __future__ import annotations

import argparse

from . import __version__
from .env_check import main as env_main
from .web import main as web_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Stator YOLO workflow launcher.")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("web", help="Open the browser-based labeling and training workflow.")
    subparsers.add_parser("env", help="Check Python, package, and camera environment.")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        return

    if args.command in (None, "web"):
        web_main()
    elif args.command == "env":
        env_main()
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
