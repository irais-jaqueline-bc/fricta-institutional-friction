from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STAGE5C_REPORT = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "official_exact"
    / "stage5c_report.json"
)
STAGE5D_REPORT = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "reachability_audit"
    / "reachability_audit_report.json"
)
STAGE5D_SUMMARY = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "reachability_audit"
    / "reachability_summary.csv"
)

THESIS_FREEZE = ROOT / "cipher" / "design" / "cipher_thesis_revision_v2.json"
AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage5e_thesis_revision_audit.json"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_summary_row(summary: pd.DataFrame, dimension: str, group: str) -> pd.Series:
    rows = summary[
        (summary["dimension"] == dimension)
        & (summary["group"].astype(str) == str(group))
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {dimension}={group}; found {len(rows)}."
        )
    return rows.iloc[0]


def main() -> None:
    if THESIS_FREEZE.exists():
        raise FileExistsError(f"Thesis revision already exists: {THESIS_FREEZE}")

    stage5c = load_json(STAGE5C_REPORT)
    stage5d = load_json(STAGE5D_REPORT)
    summary = pd.read_csv(STAGE5D_SUMMARY, dtype={"group": str})

    p1 = get_summary_row(summary, "REFERENCE_PROFILE", "1")
    p2 = get_summary_row(summary, "REFERENCE_PROFILE", "2")
    boundary = get_summary_row(summary, "CERTAINTY_CLASS", "BOUNDARY")
    core = get_summary_row(summary, "CERTAINTY_CLASS", "CORE")
    halo = get_summary_row(summary, "CERTAINTY_CLASS", "HALO")

    checks = {
        "stage5c_failed_original_coverage_gate": (
            stage5c.get("gate_status") == "FAIL_STAGE_5C"
        ),
        "stage5d_audit_complete": (
            stage5d.get("gate_status") == "STAGE_5D_REACHABILITY_AUDIT_COMPLETE"
        ),
        "observed_coverage_below_frozen_070": (float(stage5c["coverage"]) < 0.70),
        "stage5c_19_of_81_solvable": (int(stage5c["solvable_institutions"]) == 19),
        "boundary_6_of_6_reachable": (
            int(boundary["n"]) == 6 and int(boundary["solvable"]) == 6
        ),
        "profile1_10_of_68_reachable": (
            int(p1["n"]) == 68 and int(p1["solvable"]) == 10
        ),
        "profile2_9_of_13_reachable": (int(p2["n"]) == 13 and int(p2["solvable"]) == 9),
    }

    if not all(checks.values()):
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("\nGATE STATUS: FAIL_STAGE_5E_THESIS_REVISION")
        raise SystemExit(1)

    revision = {
        "version": "CIPHER_THESIS_V2",
        "status": ("FROZEN_AFTER_PREDECLARED_COUNTERFACTUAL_COVERAGE_STOP_RULE"),
        "reason_for_revision": {
            "original_stage5c_coverage_gate": 0.70,
            "observed_single_model_exact_coverage": float(stage5c["coverage"]),
            "solvable_institutions": int(stage5c["solvable_institutions"]),
            "total_institutions": 81,
            "decision": (
                "The original broad-coverage counterfactual thesis is abandoned. "
                "No sparsity, plausibility, or coverage threshold is relaxed."
            ),
        },
        "original_counterfactual_claim_status": {
            "claim": (
                "Sparse plausible counterfactual profile transitions are broadly "
                "available across the institutional sample."
            ),
            "status": "REJECTED_BY_FROZEN_STAGE5C_GATE",
        },
        "revised_empirical_thesis": (
            "Sparse plausible counterfactual reachability is heterogeneous rather "
            "than broadly available: in this sample, reachability is concentrated "
            "among institutions with uncertain profile membership and differs "
            "substantially by reference profile."
        ),
        "revised_methodological_question": (
            "When stable institutional profiles are discovered under perturbation, "
            "does membership uncertainty identify where sparse plausible profile "
            "transitions remain reachable, and which of those reachable transitions "
            "remain valid across a heterogeneous perturbation ensemble?"
        ),
        "evidence_triggering_revision": {
            "coverage": {
                "overall": float(stage5c["coverage"]),
                "profile_1": float(p1["coverage"]),
                "profile_2": float(p2["coverage"]),
                "boundary": float(boundary["coverage"]),
                "halo": float(halo["coverage"]),
                "core": float(core["coverage"]),
            },
            "exploratory_stage5d": {
                "profile2_vs_profile1_reachability_fisher": (
                    stage5d["profile2_vs_profile1_reachability_fisher"]
                ),
                "boundary_vs_nonboundary_reachability_fisher": (
                    stage5d["boundary_vs_nonboundary_reachability_fisher"]
                ),
                "certainty_class_vs_reachability": (
                    stage5d["certainty_class_vs_reachability"]
                ),
                "profile_vs_failure_mode": (stage5d["profile_vs_failure_mode"]),
                "profile1_uncertainty_comparisons": (
                    stage5d["profile1_uncertainty_comparisons"]
                ),
            },
        },
        "interpretive_constraints": [
            (
                "Stage 5D analyses are explicitly exploratory post-failure analyses "
                "and are not presented as preregistered confirmatory tests."
            ),
            (
                "Membership certainty and counterfactual reachability are derived "
                "from the same institutional feature geometry; their association is "
                "structural, not external validation."
            ),
            (
                "Counterfactuals remain diagnostic profile-transition explanations, "
                "not causal interventions, treatment effects, or recommendations."
            ),
            (
                "Profile-specific reachability asymmetry must not be interpreted as "
                "one profile being easier to improve; target directions can include "
                "both increases and decreases in friction indicators."
            ),
            (
                "The 81-institution Mexican sample supports an empirical case study, "
                "not national representativeness."
            ),
        ],
        "revised_stage6_objective": {
            "name": "ENSEMBLE ROBUSTNESS AUDIT OF REACHABLE TRANSITIONS",
            "scope": (
                "Begin from the 19 institutions with at least one exact, sparse, "
                "plausible single-model transition."
            ),
            "primary_outputs": [
                (
                    "For each single-model reachable institution, quantify ensemble "
                    "target-profile support for exact counterfactual candidates."
                ),
                (
                    "Determine which reachable institutions retain at least one "
                    "ensemble-valid transition at tau=0.90, with tau=0.80 and 0.95 "
                    "reported as sensitivity analyses."
                ),
                (
                    "Compare robustness across CORE/HALO/BOUNDARY and by reference "
                    "profile descriptively; any inferential analyses remain exploratory."
                ),
            ],
            "important_change_from_old_stage6": (
                "Stage 6 no longer attempts to rescue >=70% global counterfactual "
                "coverage. It audits robustness conditional on Stage 5C reachability."
            ),
        },
        "motif_stage_policy": {
            "status": "CONDITIONAL",
            "rule": (
                "Signed motif mining proceeds only if Stage 6 leaves enough "
                "ensemble-valid institutions for the already-frozen support and "
                "recurrence requirements to be meaningful. Otherwise the motif claim "
                "is removed rather than thresholds being relaxed."
            ),
        },
        "synthetic_validation_policy": {
            "status": "REQUIRED",
            "reason": (
                "Because the uncertainty-reachability relationship was identified "
                "after the empirical coverage stop rule, synthetic validation must "
                "test whether the revised CIPHER interpretation behaves as expected "
                "under stable-core/ambiguous-boundary, severity-only, confounded, and "
                "no-cluster scenarios."
            ),
        },
        "source_hashes": {
            "stage5c_report": sha256_file(STAGE5C_REPORT),
            "stage5d_report": sha256_file(STAGE5D_REPORT),
            "stage5d_summary": sha256_file(STAGE5D_SUMMARY),
        },
        "gate_status": "PASS_STAGE_5E_THESIS_REVISION",
    }

    THESIS_FREEZE.parent.mkdir(parents=True, exist_ok=True)
    THESIS_FREEZE.write_text(
        json.dumps(revision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "checks": checks,
                "thesis_revision_path": str(THESIS_FREEZE.relative_to(ROOT)),
                "thesis_revision_sha256": sha256_file(THESIS_FREEZE),
                "gate_status": "PASS_STAGE_5E_THESIS_REVISION",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 5E — THESIS REVISION FREEZE ===\n")
    print("Original broad-coverage counterfactual claim:")
    print("  REJECTED BY FROZEN STAGE 5C GATE")
    print(f"  observed coverage = {float(stage5c['coverage']):.4f}")

    print("\nRevised empirical thesis:")
    print("  Sparse plausible counterfactual reachability is heterogeneous rather")
    print("  than broadly available: reachability is concentrated among institutions")
    print("  with uncertain profile membership and differs by reference profile.")

    print("\nRevised Stage 6:")
    print(
        "  Ensemble robustness audit conditional on the 19 "
        "Stage 5C-reachable institutions."
    )
    print("  The >=70% global counterfactual coverage target is NOT reintroduced.")

    print("\nInterpretation:")
    print("  Stage 5D associations remain exploratory and structural, not causal.")

    print("\n=== FREEZE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_5E_THESIS_REVISION")
    print("Frozen thesis revision:", THESIS_FREEZE.relative_to(ROOT))


if __name__ == "__main__":
    main()
