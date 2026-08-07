from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MASTER_SEED = 20260807

EVALUATOR_V2_PATH = ROOT / "cipher" / "design" / "stage8_evaluator_freeze_v2.json"
CF_EVALUATOR_V2_PATH = (
    ROOT / "cipher" / "design" / "stage8_counterfactual_evaluator_freeze_v2.json"
)
STAGE8D1_V2_AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8d1_core_pipeline_smoke_audit_v2.json"
)
STAGE8D2_V3_AUDIT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "audit"
    / "stage8d2_counterfactual_smoke_audit_v3.json"
)

OFFICIAL_ROOT = ROOT / "cipher" / "outputs" / "synthetic" / "official"

FREEZE_PATH = ROOT / "cipher" / "design" / "stage8_official_run_plan_freeze_v1.json"
AUDIT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "audit"
    / "stage8e_official_run_plan_freeze_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_outputs_absent() -> bool:
    if not OFFICIAL_ROOT.exists():
        return True
    return not any(path.is_file() for path in OFFICIAL_ROOT.rglob("*"))


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(
            f"Official Stage-8 run plan already exists: {FREEZE_PATH}"
        )

    evaluator = load_json(EVALUATOR_V2_PATH)
    cf_evaluator = load_json(CF_EVALUATOR_V2_PATH)
    core_smoke = load_json(STAGE8D1_V2_AUDIT_PATH)
    cf_smoke = load_json(STAGE8D2_V3_AUDIT_PATH)

    official_replicates = list(range(2, 102))
    convergence_replicates = list(range(2, 12))
    remaining_replicates = list(range(12, 102))

    checks = {
        "stage8c1_evaluator_passed": (
            evaluator.get("gate_status") == "PASS_STAGE_8C1_MULTICLASS_AMENDMENT"
        ),
        "stage8c3_cf_evaluator_passed": (
            cf_evaluator.get("gate_status")
            == "PASS_STAGE_8C3_INSUFFICIENT_ENSEMBLE_POLICY"
        ),
        "stage8d1_v2_smoke_passed": (
            core_smoke.get("status") == "PASS_STAGE_8D1_V2_CORE_PIPELINE_SMOKE"
        ),
        "stage8d2_v3_smoke_passed": (
            cf_smoke.get("status") == "PASS_STAGE_8D2_V3_COUNTERFACTUAL_SMOKE"
        ),
        "official_outputs_absent": official_outputs_absent(),
        "official_replicate_count_is_100": len(official_replicates) == 100,
        "convergence_replicate_count_is_10": len(convergence_replicates) == 10,
        "smoke_replicate_1_excluded": 1 not in official_replicates,
        "generator_smoke_replicate_0_excluded": 0 not in official_replicates,
        "convergence_subset_of_official": set(convergence_replicates).issubset(
            official_replicates
        ),
        "remaining_plus_convergence_cover_official": (
            set(convergence_replicates).union(remaining_replicates)
            == set(official_replicates)
        ),
        "no_overlap_between_audit_and_remaining": (
            set(convergence_replicates).isdisjoint(remaining_replicates)
        ),
    }

    if not all(checks.values()):
        print("\n=== CIPHER STAGE 8E — OFFICIAL RUN PLAN FREEZE ===\n")
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("\nGATE STATUS: FAIL_STAGE_8E_OFFICIAL_RUN_PLAN_FREEZE")
        raise SystemExit(1)

    freeze = {
        "version": "STAGE8_OFFICIAL_RUN_PLAN_FREEZE_V1",
        "status": "FROZEN_BEFORE_ANY_OFFICIAL_SYNTHETIC_RESULT",
        "master_seed": MASTER_SEED,
        "purpose": (
            "Freeze official replicate identities, convergence-audit identities, "
            "ensemble escalation rules, execution order, and final aggregation "
            "semantics before any Stage-8 official result is generated."
        ),
        "parent_artifacts": {
            "evaluator_v2": {
                "path": str(EVALUATOR_V2_PATH.relative_to(ROOT)),
                "sha256": sha256_file(EVALUATOR_V2_PATH),
            },
            "cf_evaluator_v2": {
                "path": str(CF_EVALUATOR_V2_PATH.relative_to(ROOT)),
                "sha256": sha256_file(CF_EVALUATOR_V2_PATH),
            },
            "stage8d1_v2_smoke_audit": {
                "path": str(STAGE8D1_V2_AUDIT_PATH.relative_to(ROOT)),
                "sha256": sha256_file(STAGE8D1_V2_AUDIT_PATH),
            },
            "stage8d2_v3_smoke_audit": {
                "path": str(STAGE8D2_V3_AUDIT_PATH.relative_to(ROOT)),
                "sha256": sha256_file(STAGE8D2_V3_AUDIT_PATH),
            },
        },
        "replicate_manifest": {
            "excluded_nonofficial_replicates": {
                "0": "generator implementation smoke; never official",
                "1": (
                    "core/CF performance smoke; outcomes already observed and "
                    "therefore prospectively excluded from official inference"
                ),
            },
            "official_replicates_per_scenario": official_replicates,
            "official_replicate_count_per_scenario": 100,
            "convergence_audit_replicates_per_scenario": convergence_replicates,
            "convergence_audit_count_per_scenario": 10,
            "remaining_official_replicates_after_audit": remaining_replicates,
            "remaining_count_per_scenario": 90,
            "scenarios": [
                "S1_CONFIG_TWO_PROFILE",
                "S2_CORE_BOUNDARY",
                "S3_DIRECTIONAL_REACHABILITY",
                "S4_SEVERITY_CONTINUUM",
                "S5_GOVERNANCE_CONFOUNDED",
                "S6_NO_CLUSTER_NULL",
            ],
            "total_official_datasets": 600,
            "note": (
                "Indices 2..101 preserve 100 official replicates while excluding "
                "the performance-smoke replicate that has already been inspected."
            ),
        },
        "official_discovery": {
            "candidate_set": {
                "representations": ["R0_STANDARDIZED", "R1_PCA85"],
                "algorithms": ["HAC_WARD", "KMEANS"],
                "k_values": [2, 3, 4, 5, 6],
            },
            "truth_blind_selection": True,
            "full_data_min_cluster_size": 5,
            "resampling_sample_fraction": 0.80,
            "official_stability_resamples_per_shortlisted_candidate": 1000,
            "selection_semantics": (
                "Use the already-frozen Stage-8 evaluator v2 model-selection rules "
                "without modification."
            ),
            "stable_partition_claim_threshold_median_ari": 0.70,
        },
        "primary_heterogeneous_ensemble": {
            "members": 200,
            "members_per_family": 50,
            "families": [
                "R0_WARD",
                "R1_PCA85_WARD",
                "R0_KMEANS",
                "R1_PCA85_KMEANS",
            ],
            "row_sample_fraction": 0.80,
            "feature_count": 11,
            "pca_variance_threshold": 0.85,
            "k": 2,
            "kmeans_n_init": 25,
        },
        "convergence_audit": {
            "official_indices": convergence_replicates,
            "important": (
                "These ten replicates are official replicates as well as convergence "
                "audit replicates. Their 200-member results are not interpreted or "
                "used to tune the method."
            ),
            "comparison_ensemble": {
                "members": 1000,
                "members_per_family": 250,
                "row_sample_fraction": 0.80,
                "feature_count": 11,
                "pca_variance_threshold": 0.85,
                "k": 2,
                "kmeans_n_init": 25,
            },
            "uncertainty_metric": {
                "applicable_when": "stable selected k=2",
                "comparison": (
                    "Spearman rank correlation across the 80 institutions between "
                    "200-member and 1000-member normalized-entropy scores"
                ),
                "pass_threshold": 0.95,
                "scenario_decision_rule": (
                    "Use 200 members for that scenario's official uncertainty metric "
                    "only if every applicable one of the ten fixed audit replicates "
                    "has Spearman >=.95. If any applicable audit replicate is <.95, "
                    "use 1000 members for uncertainty on every applicable official "
                    "replicate in that scenario. If none of the ten audit replicates "
                    "is applicable, default to 1000 for future applicable replicates "
                    "in that scenario."
                ),
            },
            "robust_reachability_metric": {
                "applicable_when": (
                    "CF scenario policy permits evaluation AND stable selected k=2 "
                    "AND the single-model CF search is evaluable"
                ),
                "comparison": (
                    "Agreement of the 20 frozen query-level robust-reachable decisions "
                    "at tau=.90 between the 200-member and 1000-member ensembles, "
                    "using the identical frozen query set and identical saved "
                    "single-model CF candidates."
                ),
                "pass_threshold": 0.95,
                "minimum_eligible_members_200": 120,
                "minimum_eligible_members_1000": 600,
                "minimum_rule_rationale": (
                    "The already-frozen 200-member minimum is 120 (60%). "
                    "The 1000-member convergence ensemble therefore uses the same "
                    "60% minimum, i.e. 600, matching the earlier CIPHER large-ensemble "
                    "design scale."
                ),
                "scenario_decision_rule": (
                    "Use 200 members for that scenario's official robust-CF metric "
                    "only if every applicable audit replicate has an evaluable "
                    "200-member robust layer, an evaluable 1000-member robust layer, "
                    "and decision agreement >=.95. Any 200-member under-sized ensemble, "
                    "any 1000-member under-sized ensemble, or any agreement <.95 "
                    "escalates that scenario's robust-CF metric to 1000 members for "
                    "all applicable official replicates. If none of the ten audit "
                    "replicates is applicable, default to 1000 for future applicable "
                    "replicates in that scenario."
                ),
            },
            "no_mixed_reporting_after_decision": (
                "Once a scenario/metric is assigned 200 or 1000 members, use that "
                "ensemble size consistently for all 100 official replicates where "
                "that metric is applicable, including rerunning the ten audit "
                "replicates at the selected size when necessary."
            ),
        },
        "counterfactual_official_rules": {
            "query_count": 20,
            "query_selection": "truth-blind, stratified by discovered reference profile",
            "max_changed_features": 4,
            "candidate_grid": (
                "current value plus feature values from the three nearest discovered "
                "target-profile anchors"
            ),
            "plausibility": (
                "target-profile 5NN Euclidean distance <= target 95th percentile"
            ),
            "cost": "weighted L1/IQR + 0.25*L0",
            "max_saved_pareto_candidates": 5,
            "primary_tau": 0.90,
            "sensitivity_taus": [0.80, 0.95],
            "insufficient_ensemble_policy": (
                "Use Stage-8 counterfactual evaluator v2 unchanged: robust metrics "
                "remain NA/NOT_EVALUABLE rather than False."
            ),
        },
        "execution_order": [
            {
                "phase": "A",
                "name": "convergence audit official subset",
                "replicates": convergence_replicates,
                "action": (
                    "Run full official pipeline for these 10 fixed replicates per scenario, "
                    "including both 200- and 1000-member ensemble comparisons wherever "
                    "the corresponding metric is applicable."
                ),
            },
            {
                "phase": "B",
                "name": "freeze ensemble-size decisions",
                "action": (
                    "Create a read-only decision artifact assigning 200 or 1000 members "
                    "separately for uncertainty and robust-CF for each scenario. "
                    "No threshold or generator change is allowed."
                ),
            },
            {
                "phase": "C",
                "name": "remaining official replicates",
                "replicates": remaining_replicates,
                "action": (
                    "Run replicates 12..101 using the Phase-B ensemble-size decisions."
                ),
            },
            {
                "phase": "D",
                "name": "final aggregation",
                "action": (
                    "Aggregate exactly 100 official replicates per scenario. "
                    "Do not include replicate 0 or 1."
                ),
            },
        ],
        "final_aggregation_semantics": {
            "general": {
                "official_denominator_per_scenario": 100,
                "smoke_results_excluded": True,
                "truth_used_only_post_pipeline": True,
                "report_metric_applicability_counts": True,
                "report_robust_layer_evaluable_and_non_evaluable_counts": True,
            },
            "S1_CONFIG_TWO_PROFILE": {
                "profile_recovery": "median ARI over all 100 official replicates",
                "k2_rate": "selected k=2 count / 100",
                "success": "median ARI >=.70 AND k2 rate >=.80",
            },
            "S2_CORE_BOUNDARY": {
                "profile_recovery": "median ARI over all 100 official replicates",
                "uncertainty_applicability": (
                    "count stable selected-k2 replicates with evaluable uncertainty / 100"
                ),
                "boundary_auc": (
                    "median boundary-vs-core normalized-entropy ROC-AUC among evaluable "
                    "binary uncertainty replicates; if zero evaluable replicates, criterion fails"
                ),
                "boundary_entropy_contrast": (
                    "for each evaluable replicate compute median entropy(boundary) - "
                    "median entropy(core); official contrast is the median of those "
                    "replicate differences"
                ),
                "success": (
                    "median ARI >=.60 AND median evaluable boundary AUROC >=.80 "
                    "AND median entropy contrast >0"
                ),
                "note": (
                    "No new applicability-rate threshold is introduced after smoke; "
                    "the applicability fraction must nevertheless be reported prominently."
                ),
            },
            "S3_DIRECTIONAL_REACHABILITY": {
                "profile_recovery": "median ARI over all 100 official replicates",
                "single_model_oracle_f1": (
                    "micro-pooled precision/recall/F1 over the truth-blind query predictions "
                    "from CF-applicable official replicates; report CF applicability count"
                ),
                "robust_precision": (
                    "micro-pooled robust precision over robust-layer-evaluable query "
                    "predictions only; report robust-layer evaluability count separately"
                ),
                "direction_recovery_rate": (
                    "number of official replicates with recovered planted latent direction / 100; "
                    "inapplicable or robust-layer-NOT_EVALUABLE replicates are not counted "
                    "as recovered, while their status remains explicitly NA"
                ),
                "numeric_label_swap_invariance_rate": (
                    "number of official replicates whose outputs/status and all evaluable "
                    "post-hoc truth metrics are unchanged under numeric true_profile 1<->2 swap / 100"
                ),
                "success": (
                    "median profile ARI >=.60 AND micro-pooled single-model oracle F1 >=.70 "
                    "AND micro-pooled robust precision >=.80 AND direction recovery rate >=.80 "
                    "AND numeric-label-swap invariance rate >=.95"
                ),
                "note": (
                    "Robust precision is conditional on robust-layer evaluability and "
                    "cannot be silently coerced to zero for insufficient ensembles."
                ),
            },
            "S4_SEVERITY_CONTINUUM": {
                "false_configurational_claim_rate": (
                    "configurational_profile_claim=True count / 100"
                ),
                "severity_falsification_rate": (
                    "severity_nearly_reconstructs_profiles=True count / 100"
                ),
                "success": (
                    "false configurational claim rate <=.10; report severity "
                    "falsification behavior alongside it"
                ),
            },
            "S5_GOVERNANCE_CONFOUNDED": {
                "false_configurational_claim_rate": (
                    "configurational_profile_claim=True count / 100"
                ),
                "governance_falsification_rate": (
                    "(strong_governance_association OR "
                    "governance_nearly_reconstructs_profiles) count / 100"
                ),
                "success": (
                    "false configurational claim rate <=.10; report governance "
                    "falsification behavior alongside it"
                ),
            },
            "S6_NO_CLUSTER_NULL": {
                "false_stable_profile_claim_rate": (
                    "stable_partition_claim=True count / 100"
                ),
                "robust_layer_evaluability_rate": (
                    "robust-layer-evaluable replicate count / 100"
                ),
                "false_robust_cf_claim_unconditional_rate": (
                    "false robust-CF claim count / 100; NOT_EVALUABLE replicates "
                    "are not relabeled as negative predictions"
                ),
                "false_robust_cf_claim_conditional_rate": (
                    "false robust-CF claim count / robust-layer-evaluable replicate count, "
                    "reported when denominator >0"
                ),
                "success": (
                    "false stable-profile claim rate <=.10 AND unconditional false "
                    "robust-CF claim rate <=.05"
                ),
                "note": (
                    "Robust-layer evaluability is reported separately because a low "
                    "evaluability rate limits interpretation even if the unconditional "
                    "false-claim criterion passes."
                ),
            },
        },
        "failure_and_rerun_policy": {
            "scientific_failure": (
                "A predeclared success criterion may fail. Report it; do not retune "
                "thresholds, generators, query budgets, or scenario geometry."
            ),
            "technical_failure": (
                "Fix only the implementation bug while preserving frozen semantics, "
                "retain the failed log, and rerun the same replicate index with the "
                "same deterministic seed."
            ),
            "never_replace_failed_replicate_with_new_index": True,
            "never_cherry_pick_replicates": True,
            "never_include_smoke_replicates": True,
        },
        "next_stage": {
            "id": "STAGE_8F",
            "name": "Convergence-audit runner",
            "scope": (
                "Implement and run official replicate indices 2..11 for all six scenarios, "
                "with 200-vs-1000 ensemble comparisons wherever applicable. "
                "Do not run replicates 12..101 until the ensemble-size decision artifact "
                "has been frozen."
            ),
        },
        "gate_status": "PASS_STAGE_8E_OFFICIAL_RUN_PLAN_FREEZE",
    }

    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "checks": checks,
                "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "gate_status": "PASS_STAGE_8E_OFFICIAL_RUN_PLAN_FREEZE",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8E — OFFICIAL RUN PLAN FREEZE ===\n")

    print("Official replicate manifest:")
    print("  generator smoke: replicate 0 EXCLUDED")
    print("  performance smoke: replicate 1 EXCLUDED")
    print("  official replicates: 2..101 (100/scenario)")
    print("  convergence-audit official subset: 2..11 (10/scenario)")
    print("  remaining official replicates: 12..101 (90/scenario)")
    print("  total official datasets: 600")

    print("\nOfficial discovery:")
    print("  R0/R1 x Ward/KMeans x k=2..6")
    print("  truth-blind frozen selection")
    print("  1000 stability resamples per shortlisted candidate")

    print("\nConvergence audit:")
    print("  compare 200 vs 1000 heterogeneous members")
    print("  uncertainty: entropy-rank Spearman >= .95")
    print("  robust CF: tau=.90 query-decision agreement >= .95")
    print("  200 robust minimum eligible = 120")
    print("  1000 robust minimum eligible = 600")
    print("  any applicable audit failure => scenario/metric escalates to 1000")
    print("  no applicable audit replicate => conservative default 1000")

    print("\nAggregation:")
    print("  exactly 100 official replicates/scenario")
    print("  smoke 0/1 never included")
    print("  applicability/evaluability counts always reported")
    print("  S6 NOT_EVALUABLE robust layers are never coerced to False")

    print("\n=== FREEZE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_8E_OFFICIAL_RUN_PLAN_FREEZE")
    print("Frozen plan:", FREEZE_PATH.relative_to(ROOT))
    print("Next: Stage 8F convergence-audit runner for official replicates 2..11 only.")


if __name__ == "__main__":
    main()
