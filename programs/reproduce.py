"""Single entry point for the independent applied-research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from programs.digital_twin.reproduce import run_digital_twin
from programs.inference.reproduce import run_inference_probe
from programs.security.reproduce import run_security_probe


RESULTS_PATH = Path(__file__).resolve().parent / "results" / "programs_summary.json"


def main() -> None:
    security = run_security_probe()
    inference = run_inference_probe()
    digital_twin = run_digital_twin()
    summary = {
        "scope": (
            "independent security, inference, and classical digital-twin "
            "studies; not manuscript results"
        ),
        "security": security,
        "inference": inference,
        "digital_twin": digital_twin,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {RESULTS_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
