"""Command-line capability probe: ``python -m skellybench capabilities``."""

from __future__ import annotations

import argparse
import json

from .capabilities import capabilities, describe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m skellybench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capabilities_parser = subparsers.add_parser("capabilities", help="print the capability contract")
    capabilities_parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    if args.format == "text":
        print(describe())
    else:
        print(json.dumps(capabilities(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
