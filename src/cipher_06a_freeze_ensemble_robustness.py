from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

THESIS_V2 = ROOT / "cipher" / "design" / "cipher_thesis_revision_v2.json"
STAGE5C_RESULTS = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "official_exact"
    / "official_counterfactuals.csv"
)
STAGE5C_DIAGNOSTICS = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "official_exact"
    / "institution_counterfactual_diagnostics.csv"
)
CIPHER_CONFIG = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"

FREEZE_PATH = ROOT / "cipher" / "design" / "stage6_ensemble_robustness_freeze.json"
AUDIT_PATH = ROOT / "cipher" / "outputs" / "audit" / "stage6a_design_freeze_audit.json"


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


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Stage 6 design freeze already exists: {FREEZE_PATH}")

    thesis = load_json(THESIS_V2)
    config = load_json(CIPHER_CONFIG)

    if thesis.get("gate_status") != "PASS_STAGE_5E_THESIS_REVISION":
        raise ValueError("Stage 5E thesis revision has not passed.")

    results = pd.read_csv(STAGE5C_RESULTS)
    diagnostics = pd.read_csv(STAGE5C_DIAGNOSTICS)

    results["institution_id"] = results["institution_id"].astype(str)
    diagnostics["institution_id"] = diagnostics["institution_id"].astype(str)

    reachable = diagnostics[diagnostics["failure_mode"] == "SOLVABLE"].copy()

    rank1 = results[results["rank"] == 1].copy()

    reachable_ids = sorted(reachable["institution_id"].astype(str).tolist())
    rank1_ids = sorted(rank1["institution_id"].astype(str).tolist())

    checks = {
        "stage5e_revision_passed": (
            thesis.get("gate_status") == "PASS_STAGE_5E_THESIS_REVISION"
        ),
        "19_stage5c_reachable_institutions": (len(reachable_ids) == 19),
        "one_rank1_candidate_per_reachable_institution": (
            len(rank1_ids) == 19 and rank1_ids == reachable_ids
        ),
        "all_reachable_have_at_least_one_saved_candidate": (
            set(reachable_ids) <= set(results["institution_id"].astype(str))
        ),
    }

    if not all(checks.values()):
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("\nGATE STATUS: FAIL_STAGE_6A_DESIGN_FREEZE")
        raise SystemExit(1)

    # Pull already-frozen values when available; otherwise use the values
    # explicitly preserved by the CIPHER master design.
    cf_cfg = config.get("counterfactuals", {})

    primary_tau = float(cf_cfg.get("robust_validity_threshold", 0.90))
    sensitivity_taus = cf_cfg.get(
        "robust_validity_sensitivity_thresholds",
        [0.80, 0.95],
    )

    if isinstance(sensitivity_taus, (int, float)):
        sensitivity_taus = [float(sensitivity_taus)]

    sensitivity_taus = sorted({float(value) for value in sensitivity_taus})

    if primary_tau in sensitivity_taus:
        sensitivity_taus.remove(primary_tau)

    max_candidates_per_institution = int(
        cf_cfg.get(
            "max_diverse_counterfactuals_per_institution",
            5,
        )
    )

    freeze = {
        "version": "STAGE6_ENSEMBLE_ROBUSTNESS_V1",
        "status": "FROZEN_BEFORE_ENSEMBLE_COUNTERFACTUAL_EVALUATION",
        "parent_thesis_revision": {
            "path": str(THESIS_V2.relative_to(ROOT)),
            "sha256": sha256_file(THESIS_V2),
        },
        "scope": {
            "institutions": 19,
            "institution_ids": reachable_ids,
            "candidate_source": (
                "Stage 5C exact, sparse, single-model plausible counterfactuals"
            ),
            "maximum_saved_candidates_per_institution": (
                max_candidates_per_institution
            ),
            "global_81_institution_coverage_gate": ("RETIRED_AFTER_STAGE5C_FAILURE"),
        },
        "primary_question": (
            "Among institutions with at least one exact sparse plausible "
            "single-model transition, which candidate transitions retain "
            "target-profile support across the frozen heterogeneous ensemble?"
        ),
        "candidate_evaluation": {
            "unit": ("institution-counterfactual candidate"),
            "candidate_set": (
                "all Stage 5C saved exact Pareto/diverse candidates, " "not only rank 1"
            ),
            "source_profile": ("the institution's frozen reference profile"),
            "target_profile": ("the opposite frozen profile"),
            "ensemble_support_definition": (
                "fraction of eligible frozen Stage 4 ensemble members that "
                "predict the counterfactual candidate as the target profile"
            ),
            "primary_tau": primary_tau,
            "sensitivity_taus": sensitivity_taus,
            "primary_candidate_validity_rule": (
                f"ensemble target-profile support >= {primary_tau:.2f}"
            ),
        },
        "member_evaluation_rules": {
            "ensemble_source": (
                "the 984 Stage-4 counterfactual-eligible frozen members"
            ),
            "member_filtering": (
                "No additional member is dropped based on counterfactual results."
            ),
            "feature_subset_handling": (
                "Each member receives the candidate values on its own frozen "
                "11-feature subset; omitted features are irrelevant to that member."
            ),
            "representation_handling": (
                "R0 members use their frozen standardized original-feature "
                "representation; R1 members apply their own frozen scaler/PCA."
            ),
            "algorithm_handling": {
                "KMEANS": ("Predict by the member's fitted cluster-centroid rule."),
                "WARD": (
                    "Predict by the already-validated Stage-4 nearest-centroid "
                    "inductive extension in that member's fitted representation."
                ),
            },
            "label_alignment": (
                "Use each member's frozen Stage-1 alignment to the two reference "
                "profiles; never realign using counterfactual outcomes."
            ),
        },
        "institution_level_outputs": {
            "robustly_reachable_primary": (
                "institution has >=1 saved Stage-5C candidate with "
                f"ensemble support >= {primary_tau:.2f}"
            ),
            "best_robust_candidate": (
                "among primary-valid candidates, choose lowest Stage-5C exact "
                "cost; ties by higher ensemble support, then lower plausibility "
                "distance, then deterministic candidate rank"
            ),
            "if_none_primary_valid": (
                "retain maximum observed ensemble support and mark institution "
                "NOT_ROBUSTLY_REACHABLE; do not generate new looser candidates"
            ),
        },
        "primary_outputs": [
            "candidate-level ensemble support across all eligible members",
            "candidate support by ensemble family",
            "institution-level robust reachability at tau=0.90",
            "sensitivity at tau=0.80 and tau=0.95",
            "best robust candidate per institution",
            "robust reachability by reference profile and certainty class",
        ],
        "required_family_breakdown": [
            "R0_KMEANS",
            "R0_WARD",
            "R1_PCA85_KMEANS",
            "R1_PCA85_WARD",
        ],
        "interpretive_constraints": [
            (
                "Stage 6 is conditional on Stage-5C single-model reachability "
                "and does not restore the failed 70% global coverage claim."
            ),
            (
                "Ensemble support measures predictive robustness under the "
                "frozen perturbation ensemble, not causal validity."
            ),
            (
                "Comparisons by certainty class/profile are descriptive unless "
                "explicitly labeled exploratory."
            ),
            (
                "No new candidate may be generated after observing ensemble "
                "support in Stage 6A/6B."
            ),
        ],
        "stage6b_gate": {
            "purpose": "artifact and prediction-engine audit before official evaluation",
            "requirements": [
                "exactly 984 eligible frozen members recovered",
                "family counts match Stage 4: 250,244,250,240 in the frozen order",
                "member predictions reproduce stored Stage-4 inductive behavior where auditable",
                "all 19 institutions and all saved Stage-5C candidates are loadable",
                "no candidate exceeds four changed features",
                "no candidate violates the frozen Stage-5C plausibility constraint",
            ],
        },
        "stage6c_policy": {
            "official_evaluation": (
                "Only after Stage 6B passes, evaluate every saved candidate "
                "for all 19 reachable institutions across all 984 members."
            ),
            "no_global_pass_fail_threshold": (
                "There is no new minimum proportion of robustly reachable "
                "institutions. The observed robust reachability rate is a result."
            ),
        },
        "source_hashes": {
            "stage5c_results": sha256_file(STAGE5C_RESULTS),
            "stage5c_diagnostics": sha256_file(STAGE5C_DIAGNOSTICS),
            "cipher_config": sha256_file(CIPHER_CONFIG),
        },
        "gate_status": "PASS_STAGE_6A_DESIGN_FREEZE",
    }

    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(
        json.dumps(
            freeze,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "checks": checks,
                "reachable_institutions": len(reachable_ids),
                "saved_candidate_rows": len(results),
                "primary_tau": primary_tau,
                "sensitivity_taus": sensitivity_taus,
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "gate_status": "PASS_STAGE_6A_DESIGN_FREEZE",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 6A — ENSEMBLE ROBUSTNESS DESIGN FREEZE ===\n")
    print("Stage-5C reachable institutions:", len(reachable_ids))
    print("Saved Stage-5C candidate rows:", len(results))
    print("Primary ensemble support threshold:", f"{primary_tau:.2f}")
    print("Sensitivity thresholds:", sensitivity_taus)

    print("\nOfficial Stage 6 question:")
    print(
        "  Which exact sparse plausible Stage-5C transitions survive the "
        "984-member heterogeneous ensemble?"
    )

    print("\nImportant:")
    print("  The failed >=70% global coverage claim remains retired.")
    print("  Stage 6 does not generate looser replacement candidates.")
    print("  No minimum robust-reachability percentage is imposed post hoc.")

    print("\n=== FREEZE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_6A_DESIGN_FREEZE")
    print("Frozen Stage 6 design:", FREEZE_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
