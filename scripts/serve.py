"""Serve the local API and the chat page on 127.0.0.1 only.

    python -m scripts.serve
    python -m scripts.serve --port 8000

There is no host flag. This process does not listen on other interfaces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from api.main import app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve ChatTwin on 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port on 127.0.0.1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.port < 1 or args.port > 65535:
        print("port 必须在 1 到 65535 之间", file=sys.stderr)
        return 1
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
