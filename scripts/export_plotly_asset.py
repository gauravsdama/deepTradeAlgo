#!/usr/bin/env python3
"""Export the installed Plotly.js runtime for an offline, cacheable dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from plotly.offline import get_plotlyjs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "static" / "vendor" / "plotly.min.js"


def expected_content() -> str:
    return get_plotlyjs().rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = expected_content()

    if args.check:
        if not args.output.exists() or args.output.read_text() != expected:
            print(f"Plotly.js asset is stale: {args.output}")
            return 1
        print(f"Plotly.js asset is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
