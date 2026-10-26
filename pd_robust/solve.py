"""Solve one model in a fresh process: python -m pd_robust.solve FILE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Settings, read_instance
from .models import METHODS, SolveOptions, solve


def add_instance_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gamma-q", type=int, default=1)
    parser.add_argument("--gamma-t", type=int, default=1)
    parser.add_argument("--sigma-q", type=float, default=0.1)
    parser.add_argument("--sigma-t", type=float, default=0.1)
    parser.add_argument("--rounding", choices=("hundredth", "none"), default="hundredth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--method", choices=METHODS, default="branch-cut")
    add_instance_options(parser)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuts-per-family", type=int, default=50)
    parser.add_argument("--no-root-cuts", action="store_true")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--raw-demands", action="store_true")
    parser.add_argument("--literal-big-m", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings(args.gamma_q, args.gamma_t, args.sigma_q, args.sigma_t,
                        args.rounding, adjust_demands=not args.raw_demands)
    options = SolveOptions(args.seconds, args.threads, args.seed, args.cuts_per_family,
                           not args.no_root_cuts, not args.no_warm_start, args.literal_big_m, args.log)
    result = solve(read_instance(args.instance, settings), args.method, options)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
