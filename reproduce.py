"""Single entry point for all manuscript computations and figures."""

from __future__ import annotations

import json
from pathlib import Path

from burau_switch import run_verification
from symbolic_checks import run_symbolic_checks


FIGURE_PATH = Path("figures/witness_gap_summary.png")
RESULTS_PATH = Path("results/verification.json")


def main() -> None:
    symbolic = run_symbolic_checks()
    diagnostics = run_verification(
        save_figure=True,
        show_figure=False,
        figure_path=FIGURE_PATH,
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"symbolic_checks": symbolic, "numerical_checks": diagnostics},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print("Exact symbolic checks: passed")
    for name, value in diagnostics.items():
        print(f"{name}: {value:.12g}")


if __name__ == "__main__":
    main()
