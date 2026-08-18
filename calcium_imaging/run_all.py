# run_all.py
# -*- coding: utf-8 -*-
"""
One-command runner for the full calcium imaging pipeline.

Usage (from this directory, in any terminal — Anaconda Prompt, cmd, bash, ...):

    python run_all.py             # run everything: build tables, then all 3 plot scripts
    python run_all.py plots       # run only the 3 plot scripts (assumes tables exist)
    python run_all.py build       # rebuild tables only
    python run_all.py q2          # run a single step
    python run_all.py build q3    # run several specific steps in order

Available steps:
    build  -- calcium_build_tables.py       (raw CSVs -> analysis tables)
    q1     -- plot_Q1_vDA_dDA_ratio.py      (E3 control vDA/dDA ratio)
    q2     -- plot_Q2_cells_events.py       (single-cell traces & event boxplots)
    q3     -- plot_Q3_vDA_trace.py          (vDA band trace, all conditions)

Aliases:
    all    -- build + q1 + q2 + q3   (default when no argument given)
    plots  -- q1 + q2 + q3

Steps are executed in the order given (left to right). If any step fails,
the runner stops and reports the failing step.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Map short step names -> actual script filenames.
# When you rename a script, update only this map.
SCRIPTS: dict[str, str] = {
    "build":    "calcium_build_tables.py",
    "q1":       "plot_Q1_vDA_dDA_ratio.py",
    "q2":       "plot_Q2_cells_events.py",
    "q3":       "plot_Q3_vDA_trace.py",
}

# Convenience aliases for groups of steps.
ALIASES: dict[str, list[str]] = {
    "all":   ["build", "q1", "q2", "q3"],
    "plots": ["q1", "q2", "q3"],
}


def resolve_steps(requested: list[str]) -> list[str]:
    """Expand aliases like 'all' / 'plots' into the canonical step list."""
    if not requested:
        return ALIASES["all"]
    resolved: list[str] = []
    for token in requested:
        if token in ALIASES:
            resolved.extend(ALIASES[token])
        elif token in SCRIPTS:
            resolved.append(token)
        else:
            valid = sorted(set(SCRIPTS) | set(ALIASES))
            raise SystemExit(
                f"Unknown step: {token!r}\n"
                f"Valid steps: {valid}"
            )
    # de-duplicate while preserving order
    seen = set()
    out = []
    for s in resolved:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def run_step(step: str, script_dir: Path) -> int:
    """Run one script in a subprocess. Returns its exit code."""
    script = script_dir / SCRIPTS[step]
    if not script.exists():
        print(f"⚠️  Script not found: {script}")
        return 127

    banner = "=" * 64
    print(f"\n{banner}\n>>> Step: {step}    ({SCRIPTS[step]})\n{banner}", flush=True)
    t0 = time.time()
    # sys.executable points to the current Python (respects active conda env).
    result = subprocess.run([sys.executable, str(script)], cwd=str(script_dir))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
    print(f"\n--- {step}: {status}    ({elapsed:.1f}s)", flush=True)
    return result.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the calcium imaging pipeline in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_all.py             # everything\n"
            "  python run_all.py plots       # only the plot scripts\n"
            "  python run_all.py q2          # only one plot script\n"
            "  python run_all.py build q3    # specific steps in order\n"
        ),
    )
    ap.add_argument(
        "steps",
        nargs="*",
        help=(
            "Steps to run. Options: "
            + ", ".join(sorted(set(SCRIPTS) | set(ALIASES)))
            + ".  Default = all."
        ),
    )
    args = ap.parse_args()

    # Make stdout robust to non-UTF-8 consoles (e.g. Windows cp1252) so the
    # status emoji in the banners below don't raise UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    script_dir = Path(__file__).resolve().parent
    steps = resolve_steps(args.steps)

    print(f"Will run {len(steps)} step(s): {steps}")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {script_dir}")

    t_all = time.time()
    for step in steps:
        rc = run_step(step, script_dir)
        if rc != 0:
            print(f"\n❌  Pipeline stopped at step '{step}' (exit code {rc}).")
            sys.exit(rc)

    print(f"\n✅  All steps completed in {time.time() - t_all:.1f}s.")


if __name__ == "__main__":
    main()
