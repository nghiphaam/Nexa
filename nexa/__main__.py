from __future__ import annotations

import argparse

from nexa import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nexa",
        description="Nexa exposes a small Python API for loading checkpoints and generating text.",
    )
    parser.add_argument("--version", action="store_true", help="Print the installed Nexa version")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.version:
        print(__version__)
        return 0

    print("Nexa is a model package focused on checkpoint loading and text generation.")
    print("Use the Python API for loading checkpoints and generating text.")
    return 0


if __name__ == "__main__":
    main()
