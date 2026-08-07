from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EVALUATOR_V1_PATH = ROOT / "cipher" / "design" / "stage8_evaluator_freeze_v1.json"
STAGE8A_PATH = ROOT / "cipher" / "design" / "stage8_synthetic_validation_freeze_v1.json"
OFFICIAL_SYNTHETIC_DIR = ROOT / "cipher" / "outputs" / "synthetic" / "official"

AMENDMENT_PATH = ROOT / "cipher" / "design" / "stage8_evaluator_freeze_v2.json"
AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8c1_multiclass_amendment_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def official_results_absent() -> bool:
    if not OFFICIAL_SYNTHETIC_DIR.exists():
        return True
    return not any(path.is_file() for path in OFFICIAL_SYNTHETIC_DIR.rglob("*"))


def main() -> None:
    if AMENDMENT_PATH.exists():
        raise FileExistsError(f"Evaluator amendment already exists: {AMENDMENT_PATH}")

    evaluator_v1 = load_json(EVALUATOR_V1_PATH)
    stage8a = load_json(STAGE8A_PATH)

    candidate_k = stage8a["pipeline_under_test"]["discovery"]["candidate_k"]
    ensemble_k = stage8a["synthetic_compute_plan"]["primary_synthetic_ensemble"]["k"]

    checks = {
        "stage8c_v1_passed": (
            evaluator_v1.get("gate_status") == "PASS_STAGE_8C_EVALUATOR_FREEZE"
        ),
        "candidate_search_includes_multiclass_k": any(int(k) > 2 for k in candidate_k),
        "synthetic_uncertainty_ensemble_is_frozen_at_k2": int(ensemble_k) == 2,
        "no_official_synthetic_performance_results_exist": official_results_absent(),
    }

    if not all(checks.values()):
        print("\n=== CIPHER STAGE 8C1 — MULTICLASS EVALUATOR AMENDMENT ===\n")
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("\nGATE STATUS: FAIL_STAGE_8C1_MULTICLASS_AMENDMENT")
        raise SystemExit(1)

    amended = evaluator_v1.copy()

    amended["version"] = "STAGE8_EVALUATOR_FREEZE_V2"
    amended["status"] = "FROZEN_BEFORE_SYNTHETIC_MODEL_PERFORMANCE"
    amended["parent_evaluator"] = {
        "path": str(EVALUATOR_V1_PATH.relative_to(ROOT)),
        "sha256": sha256_file(EVALUATOR_V1_PATH),
    }

    amended["prospective_multiclass_amendment"] = {
        "reason": (
            "Stage 8A froze model selection over k=2..6, while the inherited real-case "
            "severity thresholding and CIPHER uncertainty/reachability layer were binary. "
            "This mismatch was identified before any synthetic model-performance result. "
            "The amendment defines how k>2 selected solutions are evaluated without "
            "changing any previously frozen thresholds."
        ),
        "timing": "BEFORE_ANY_SYNTHETIC_MODEL_PERFORMANCE",
        "candidate_k_values": [int(k) for k in candidate_k],
        "frozen_uncertainty_ensemble_k": int(ensemble_k),
    }

    amended["severity_null"]["multiclass_extension"] = {
        "k_equals_2": (
            "Use the inherited exact training-fold threshold-and-direction search "
            "without modification."
        ),
        "k_greater_than_2": {
            "model": "DecisionTreeClassifier on the single scalar severity score",
            "features": ["unweighted_mean_of_13_features"],
            "max_leaf_nodes": "selected_k",
            "class_weight": "balanced",
            "min_samples_leaf": 2,
            "random_state": "replicate-derived deterministic seed",
            "cv": "same repeated stratified train/test folds as the severity null",
            "role": (
                "A one-dimensional non-linear severity baseline that generalizes "
                "the binary threshold idea to k>2 discovered partitions."
            ),
        },
        "flag_thresholds_unchanged": {
            "balanced_accuracy_median_min": 0.90,
            "ari_median_min": 0.80,
        },
        "matched_severity_pairs_multiclass": (
            "A matched pair is any pair assigned to different discovered clusters "
            "with absolute mean-severity gap <=0.05. This remains descriptive only."
        ),
    }

    amended["governance_null"]["multiclass_extension"] = {
        "association": (
            "Bias-corrected Cramer's V and permutation testing already support "
            "R x C contingency tables and are used unchanged."
        ),
        "prediction_baseline": (
            "The training-fold governance-category -> majority discovered-cluster "
            "mapping already supports arbitrary selected k and is used unchanged."
        ),
        "flag_thresholds_unchanged": True,
    }

    amended["membership_uncertainty"]["selected_k_policy"] = {
        "k_equals_2": (
            "Run the frozen heterogeneous k=2 ensemble, binary normalized entropy, "
            "CORE/HALO/BOUNDARY classification, and reachability layer."
        ),
        "k_greater_than_2": (
            "Do not run or claim the binary CIPHER membership-uncertainty or "
            "counterfactual-reachability layer for that replicate. Record the layer "
            "as NOT_APPLICABLE_SELECTED_K_NOT_2."
        ),
        "reason": (
            "Stage 8A explicitly froze the heterogeneous synthetic ensemble at k=2. "
            "Retrofitting a k>2 uncertainty ensemble after seeing synthetic outcomes "
            "would violate the freeze."
        ),
    }

    amended["reachability_evaluation"]["selected_k_required"] = 2
    amended["reachability_evaluation"][
        "k_not_2_policy"
    ] = "NOT_APPLICABLE_SELECTED_K_NOT_2; no robust-reachability claim is made."

    amended["configurational_profile_claim"]["multiclass_policy"] = {
        "stable_partition_claim": (
            "May be evaluated for any selected k in 2..6 using the frozen stability gate."
        ),
        "severity_falsification": (
            "For k=2 use the inherited exact threshold baseline; for k>2 use the "
            "prospectively frozen one-dimensional decision-tree severity baseline."
        ),
        "governance_falsification": "Applies unchanged for any selected k.",
        "uncertainty_or_reachability_required": False,
        "note": (
            "The configurational-profile claim is a clustering/falsification claim. "
            "The binary CIPHER uncertainty/reachability contribution is evaluated "
            "only on replicates selecting k=2."
        ),
    }

    amended["null_scenario_accounting"] = {
        "S4_SEVERITY_CONTINUUM": (
            "Any configurational_profile_claim=True is a false configurational claim, "
            "regardless of selected k."
        ),
        "S5_GOVERNANCE_CONFOUNDED": (
            "Any configurational_profile_claim=True is a false configurational claim, "
            "regardless of selected k."
        ),
        "S6_NO_CLUSTER_NULL": (
            "Any stable_partition_claim=True is a false stable-profile claim. "
            "A false robust-reachability claim can occur only when selected_k=2 and "
            "the binary reachability layer makes such a claim."
        ),
    }

    amended["no_post_result_changes"] = list(
        dict.fromkeys(
            amended.get("no_post_result_changes", [])
            + [
                "no replacing the k>2 one-dimensional severity baseline after smoke results",
                "no running a k>2 uncertainty ensemble unless declared later as a separate post-hoc analysis",
                "no treating NOT_APPLICABLE reachability as a successful negative prediction",
            ]
        )
    )

    amended["next_stage"] = {
        "id": "STAGE_8D1",
        "name": "One-replicate discovery, stability, falsification, and uncertainty smoke",
        "rule": (
            "Run one non-official performance smoke replicate per scenario. "
            "First validate selection/stability/null/uncertainty mechanics. "
            "Do not run official replicates and do not interpret smoke values scientifically."
        ),
    }

    amended["gate_status"] = "PASS_STAGE_8C1_MULTICLASS_AMENDMENT"

    AMENDMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AMENDMENT_PATH.write_text(
        json.dumps(
            amended,
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
                "amendment_path": str(AMENDMENT_PATH.relative_to(ROOT)),
                "amendment_sha256": sha256_file(AMENDMENT_PATH),
                "gate_status": "PASS_STAGE_8C1_MULTICLASS_AMENDMENT",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8C1 — MULTICLASS EVALUATOR AMENDMENT ===\n")

    print("Why this amendment exists:")
    print("  candidate search is k=2..6")
    print("  inherited real severity threshold is binary")
    print("  frozen CIPHER uncertainty/reachability ensemble is k=2")
    print("  mismatch detected BEFORE synthetic performance")

    print("\nSeverity policy:")
    print("  k=2: inherited exact threshold/direction baseline")
    print("  k>2: 1D severity-only DecisionTreeClassifier")
    print(
        "       max_leaf_nodes=selected_k, class_weight='balanced', min_samples_leaf=2"
    )
    print("  reconstruction thresholds remain BA>=.90 AND ARI>=.80")

    print("\nGovernance policy:")
    print("  already multiclass-compatible; no methodological change")

    print("\nUncertainty/reachability policy:")
    print("  selected k=2: run frozen binary CIPHER layer")
    print("  selected k>2: NOT_APPLICABLE_SELECTED_K_NOT_2")
    print("  NOT_APPLICABLE is not counted as a successful negative prediction")

    print("\nNull accounting:")
    print("  S4/S5: any configurational claim is false, regardless of selected k")
    print("  S6: any stable-partition claim is false")
    print("      robust-CF false claim can only be evaluated when selected k=2")

    print("\n=== FREEZE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_8C1_MULTICLASS_AMENDMENT")
    print("Frozen evaluator v2:", AMENDMENT_PATH.relative_to(ROOT))
    print("Next: Stage 8D1 smoke. Do NOT start official synthetic replicates.")


if __name__ == "__main__":
    main()
