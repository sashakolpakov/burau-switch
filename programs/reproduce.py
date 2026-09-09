"""Single entry point for the independent applied-research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from programs.compiler.reproduce import run_compiler_study
from programs.digital_twin.reproduce import run_digital_twin
from programs.inference.reproduce import run_inference_probe
from programs.inference_scaling.reproduce import run_inference_scaling
from programs.passive_security.reproduce import run_passive_security_study
from programs.security.reproduce import run_security_probe


RESULTS_PATH = Path(__file__).resolve().parent / "results" / "programs_summary.json"


def main() -> None:
    security = run_security_probe()
    passive_security = run_passive_security_study()
    inference = run_inference_probe()
    inference_scaling = run_inference_scaling()
    digital_twin = run_digital_twin()
    compiler = run_compiler_study()
    summary = {
        "scope": (
            "independent security-control, passive-security, inference, "
            "inference-scaling, classical digital-twin, and compiler studies; "
            "not experimental hardware results"
        ),
        "security": security,
        "passive_security": passive_security,
        "inference": inference,
        "inference_scaling": inference_scaling,
        "digital_twin": digital_twin,
        "compiler": compiler,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {RESULTS_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
