from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CF_FREEZE_V1 = (
    ROOT / "cipher" / "design" / "stage8_counterfactual_evaluator_freeze_v1.json"
)
EVALUATOR_V2 = ROOT / "cipher" / "design" / "stage8_evaluator_freeze_v2.json"
STAGE8D2_V2_DIR = ROOT / "cipher" / "outputs" / "synthetic" / "counterfactual_smoke_v2"
OFFICIAL_DIR = ROOT / "cipher" / "outputs" / "synthetic" / "official"

FREEZE_PATH = (
    ROOT / "cipher" / "design" / "stage8_counterfactual_evaluator_freeze_v2.json"
)
AUDIT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "audit"
    / "stage8c3_insufficient_ensemble_policy_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def official_results_absent() -> bool:
    if not OFFICIAL_DIR.exists():
        return True

    return not any(path.is_file() for path in OFFICIAL_DIR.rglob("*"))


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Stage 8C3 freeze already exists: {FREEZE_PATH}")

    cf_v1 = load_json(CF_FREEZE_V1)

    evaluator_v2 = load_json(EVALUATOR_V2)

    checks = {
        "stage8c2_cf_freeze_passed": (
            cf_v1.get("gate_status") == "PASS_STAGE_8C2_COUNTERFACTUAL_EVALUATOR_FREEZE"
        ),
        "stage8c1_evaluator_v2_passed": (
            evaluator_v2.get("gate_status") == "PASS_STAGE_8C1_MULTICLASS_AMENDMENT"
        ),
        "partial_cf_smoke_v2_exists": (STAGE8D2_V2_DIR.exists()),
        "official_results_absent": (official_results_absent()),
        "minimum_eligible_members_still_120": (
            int(cf_v1["robust_ensemble"]["minimum_eligible_members"]) == 120
        ),
        "ward_fidelity_threshold_still_095": (
            float(cf_v1["robust_ensemble"]["ward_member_fidelity_min"]) == 0.95
        ),
    }

    if not all(checks.values()):
        print("\n=== CIPHER STAGE 8C3 — INSUFFICIENT-ENSEMBLE POLICY FREEZE ===\n")

        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")

        print("\nGATE STATUS: FAIL_STAGE_8C3_INSUFFICIENT_ENSEMBLE_POLICY")

        raise SystemExit(1)

    cf_v2 = dict(cf_v1)

    cf_v2["version"] = "STAGE8_COUNTERFACTUAL_EVALUATOR_FREEZE_V2"

    cf_v2["status"] = "FROZEN_BEFORE_OFFICIAL_SYNTHETIC_COUNTERFACTUAL_RUNS"

    cf_v2["parent_cf_evaluator"] = {
        "path": str(CF_FREEZE_V1.relative_to(ROOT)),
        "sha256": sha256_file(CF_FREEZE_V1),
    }

    cf_v2["prospective_insufficient_ensemble_amendment"] = {
        "reason": (
            "The Stage 8D2 v2 smoke reached a stable k=2 null replicate whose "
            "heterogeneous ensemble retained fewer than the already-frozen "
            "minimum of 120 eligible members after Ward fidelity filtering. "
            "The threshold is not relaxed. This amendment defines the required "
            "non-crashing evaluation status for that pre-existing failure condition."
        ),
        "timing": (
            "AFTER_NON_OFFICIAL_SMOKE_FAILURE_AND_BEFORE_ANY_OFFICIAL_SYNTHETIC_RUN"
        ),
        "scientific_thresholds_changed": False,
        "minimum_eligible_members": 120,
        "ward_fidelity_threshold": 0.95,
    }

    cf_v2["robust_ensemble"]["insufficient_eligible_member_policy"] = {
        "condition": ("eligible_members < 120"),
        "status": ("ROBUST_LAYER_NOT_EVALUABLE_INSUFFICIENT_ELIGIBLE_ENSEMBLE"),
        "single_model_counterfactuals_remain_valid": True,
        "robust_support_computation": ("DO_NOT_COMPUTE"),
        "robust_reachable_tau_080": None,
        "robust_reachable_tau_090": None,
        "robust_reachable_tau_095": None,
        "do_not_impute_as_negative_prediction": True,
        "do_not_lower_minimum_member_requirement": True,
        "do_not_lower_ward_fidelity_threshold": True,
    }

    cf_v2["insufficient_ensemble_truth_scoring"] = {
        "S3_DIRECTIONAL_REACHABILITY": {
            "single_model_oracle_metrics": (
                "MAY_BE_COMPUTED because they do not depend on the robust ensemble"
            ),
            "robust_precision_recall_f1": ("NOT_EVALUABLE"),
            "robust_direction_recovery": ("NOT_EVALUABLE"),
            "robust_label_swap_metric": ("NOT_EVALUABLE"),
            "reason": (
                "A robust performance metric cannot be scored when the frozen "
                "minimum heterogeneous ensemble was not formed."
            ),
        },
        "S6_NO_CLUSTER_NULL": {
            "cf_search_applicable": True,
            "robust_layer_applicable": False,
            "false_robust_cf_claim": None,
            "status": ("NOT_EVALUABLE_INSUFFICIENT_ELIGIBLE_ENSEMBLE"),
            "do_not_record_false_as_no_false_claim": True,
        },
    }

    cf_v2["official_reporting_for_non_evaluable_robust_layer"] = {
        "required_counts": [
            "total_replicates",
            "robust_layer_evaluable_replicates",
            "robust_layer_not_evaluable_replicates",
        ],
        "required_rates_when_defined": [
            "unconditional false-claim count / total replicates",
            "conditional false-claim rate among robust-layer-evaluable replicates",
        ],
        "missing_robust_metrics": (
            "Keep as NA/NOT_EVALUABLE. Never coerce to zero, False, or successful abstention."
        ),
        "interpretation": (
            "A low robust-layer evaluability rate must be reported as a limitation; "
            "it cannot be hidden by excluding those replicates from the narrative."
        ),
        "new_success_threshold_added": False,
    }

    cf_v2["no_post_result_changes"] = list(
        dict.fromkeys(
            cf_v2.get(
                "no_post_result_changes",
                [],
            )
            + [
                "no lowering the 120-member minimum after insufficient-ensemble smoke",
                "no lowering Ward fidelity .95 after insufficient-ensemble smoke",
                "no converting robust-layer NOT_EVALUABLE into a negative prediction",
                "no dropping non-evaluable replicates from applicability reporting",
            ]
        )
    )

    cf_v2["next_stage"] = {
        "id": ("STAGE_8D2_V3"),
        "name": ("Counterfactual smoke with graceful insufficient-ensemble handling"),
        "rule": (
            "Rerun the non-official CF smoke in a fresh output namespace. "
            "If an applicable replicate has fewer than 120 eligible ensemble "
            "members, preserve its single-model CF results and mark the robust "
            "layer NOT_EVALUABLE instead of crashing."
        ),
    }

    cf_v2["gate_status"] = "PASS_STAGE_8C3_INSUFFICIENT_ENSEMBLE_POLICY"

    FREEZE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FREEZE_PATH.write_text(
        json.dumps(
            cf_v2,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_PATH.write_text(
        json.dumps(
            {
                "checks": checks,
                "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "gate_status": ("PASS_STAGE_8C3_INSUFFICIENT_ENSEMBLE_POLICY"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8C3 — INSUFFICIENT-ENSEMBLE POLICY FREEZE ===\n")

    print("Smoke failure condition:")
    print("  eligible robust-ensemble members < frozen minimum 120")

    print("\nThresholds remain unchanged:")
    print("  minimum eligible members = 120")
    print("  Ward inductive fidelity >= .95")

    print("\nFrozen handling:")
    print("  single-model CF outputs remain valid")
    print("  robust support is NOT computed")
    print("  status = ROBUST_LAYER_NOT_EVALUABLE_INSUFFICIENT_ELIGIBLE_ENSEMBLE")
    print("  tau=.80/.90/.95 outputs = NA, not False")

    print("\nTruth scoring:")
    print("  S3 single-model oracle metrics may remain evaluable")
    print("  S3 robust metrics/direction = NOT_EVALUABLE")
    print("  S6 false robust-CF claim = NA, not False")

    print("\nOfficial reporting later:")
    print("  report evaluable and non-evaluable replicate counts separately")
    print("  never hide or coerce non-evaluable robust replicates")

    print("\n=== FREEZE CHECKS ===\n")

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_8C3_INSUFFICIENT_ENSEMBLE_POLICY")

    print(
        "Frozen CF evaluator v2:",
        FREEZE_PATH.relative_to(ROOT),
    )

    print("Next: Stage 8D2 v3 smoke. Do NOT run official replicates.")


if __name__ == "__main__":
    main()
