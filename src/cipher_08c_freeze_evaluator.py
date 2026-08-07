from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

STAGE8A_PATH = ROOT / "cipher" / "design" / "stage8_synthetic_validation_freeze_v1.json"
STAGE8B_PATH = (
    ROOT / "cipher" / "design" / "stage8_generator_implementation_freeze_v1.json"
)
STAGE8B_AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8b_generator_smoke_audit.json"
)

MODEL_SELECTION_SOURCE = ROOT / "src" / "icdm_select_model.py"
SEVERITY_SOURCE = ROOT / "src" / "cipher_03_null_models.py"
GOVERNANCE_SOURCE = ROOT / "src" / "cipher_03b_governance_null.py"
UNCERTAINTY_SOURCE = ROOT / "src" / "cipher_02_membership_certainty.py"

FREEZE_PATH = ROOT / "cipher" / "design" / "stage8_evaluator_freeze_v1.json"
AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8c_evaluator_freeze_audit.json"
)

OFFICIAL_SYNTHETIC_DIR = ROOT / "cipher" / "outputs" / "synthetic" / "official"


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


def source_contains(path: Path, snippets: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(snippet in text for snippet in snippets)


def official_results_absent() -> bool:
    if not OFFICIAL_SYNTHETIC_DIR.exists():
        return True
    return not any(path.is_file() for path in OFFICIAL_SYNTHETIC_DIR.rglob("*"))


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Stage 8 evaluator freeze already exists: {FREEZE_PATH}")

    stage8a = load_json(STAGE8A_PATH)
    stage8b = load_json(STAGE8B_PATH)
    stage8b_audit = load_json(STAGE8B_AUDIT_PATH)

    checks = {
        "stage8a_design_passed": (
            stage8a.get("gate_status") == "PASS_STAGE_8A_SYNTHETIC_DESIGN_FREEZE"
        ),
        "stage8b_implementation_frozen": (
            stage8b.get("gate_status") == "FROZEN_STAGE8_GENERATOR_IMPLEMENTATION_V1"
        ),
        "stage8b_smoke_passed": (
            stage8b_audit.get("gate_status") == "PASS_STAGE_8B_GENERATOR_SMOKE_AUDIT"
        ),
        "no_official_synthetic_performance_results_exist": official_results_absent(),
        "real_model_selection_rule_found": source_contains(
            MODEL_SELECTION_SOURCE,
            [
                'merged["minimum_resample_cluster_size_min"] >= 5',
                'merged["minimum_cluster_size"] >= 5',
                'best_median = eligible["ari_median"].max()',
                '"partition_equivalence_threshold": 0.95',
                '"silhouette", "davies_bouldin", "calinski_harabasz", "k_requested"',
                '"ari_p025"',
                '"clusterwise_jaccard_min_mean"',
            ],
        ),
        "real_severity_flag_found": source_contains(
            SEVERITY_SOURCE,
            [
                '"severity_nearly_reconstructs_profiles"',
                'severity_summary["balanced_accuracy_median"] >= 0.90',
                'severity_summary["ari_median"] >= 0.80',
                '"matched_severity_opposite_profile_pairs_exist"',
            ],
        ),
        "real_governance_flags_found": source_contains(
            GOVERNANCE_SOURCE,
            [
                '"strong_governance_association"',
                "observed_v >= 0.50 and permutation_p < 0.05",
                '"governance_nearly_reconstructs_profiles"',
                'cv["balanced_accuracy"].median() >= 0.90 and cv["ari"].median() >= 0.80',
            ],
        ),
        "real_uncertainty_formulas_found": source_contains(
            UNCERTAINTY_SOURCE,
            [
                "def normalized_entropy_binary",
                "entropy = -np.sum(probs * np.log(probs))",
                "return float(entropy / np.log(2.0))",
                "margin = float(2.0 * abs(p1 - 0.5))",
                "family_consistency = float(min(family_reference_probs))",
                "if reference_probability >= core_p and family_consistency >= core_family",
                "elif reference_probability < boundary_p or family_consistency < boundary_family",
            ],
        ),
    }

    if not all(checks.values()):
        print("\n=== CIPHER STAGE 8C — EVALUATOR FREEZE ===\n")
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("\nGATE STATUS: FAIL_STAGE_8C_EVALUATOR_FREEZE")
        raise SystemExit(1)

    evaluator = {
        "version": "STAGE8_EVALUATOR_FREEZE_V1",
        "status": "FROZEN_BEFORE_SYNTHETIC_MODEL_PERFORMANCE",
        "purpose": (
            "Operationalize every evaluator decision before the first synthetic "
            "model-performance result is observed."
        ),
        "parent_artifacts": {
            "stage8a_design": {
                "path": str(STAGE8A_PATH.relative_to(ROOT)),
                "sha256": sha256_file(STAGE8A_PATH),
            },
            "stage8b_generator_implementation": {
                "path": str(STAGE8B_PATH.relative_to(ROOT)),
                "sha256": sha256_file(STAGE8B_PATH),
            },
            "stage8b_smoke_audit": {
                "path": str(STAGE8B_AUDIT_PATH.relative_to(ROOT)),
                "sha256": sha256_file(STAGE8B_AUDIT_PATH),
            },
        },
        "source_semantics": {
            "model_selection": {
                "source": str(MODEL_SELECTION_SOURCE.relative_to(ROOT)),
                "sha256": sha256_file(MODEL_SELECTION_SOURCE),
            },
            "severity_null": {
                "source": str(SEVERITY_SOURCE.relative_to(ROOT)),
                "sha256": sha256_file(SEVERITY_SOURCE),
            },
            "governance_null": {
                "source": str(GOVERNANCE_SOURCE.relative_to(ROOT)),
                "sha256": sha256_file(GOVERNANCE_SOURCE),
            },
            "uncertainty": {
                "source": str(UNCERTAINTY_SOURCE.relative_to(ROOT)),
                "sha256": sha256_file(UNCERTAINTY_SOURCE),
            },
        },
        "candidate_set": {
            "representations": ["R0_STANDARDIZED", "R1_PCA85"],
            "algorithms": ["HAC_WARD", "KMEANS"],
            "k_values": [2, 3, 4, 5, 6],
            "note": (
                "Stage 8A froze the synthetic candidate set to Ward and KMeans. "
                "The real ICDM selection semantics are reused on that frozen set; "
                "GMM is not reintroduced."
            ),
        },
        "model_selection": {
            "eligibility": {
                "full_data_minimum_cluster_size": 5,
                "resampling_minimum_cluster_size": 5,
                "rule": (
                    "A candidate is eligible only if both its full-data minimum "
                    "cluster size and the minimum cluster size observed across "
                    "resampling are at least five."
                ),
            },
            "primary_rule": (
                "Among eligible candidates, maximize median resampling ARI."
            ),
            "median_ari_tie_tolerance": 1e-12,
            "partition_equivalence_ari_threshold": 0.95,
            "equivalent_partition_tiebreak": [
                "silhouette descending",
                "davies_bouldin ascending",
                "calinski_harabasz descending",
                "k ascending",
            ],
            "non_equivalent_partition_tiebreak": [
                "ari_p025 descending",
                "clusterwise_jaccard_min_mean descending",
                "silhouette descending",
                "k ascending",
            ],
            "ground_truth_blinding": (
                "Synthetic truth labels, true boundary flags, oracle reachability "
                "labels, and governance truth are forbidden from model selection."
            ),
        },
        "stability_claim_gate": {
            "status": "PROSPECTIVELY_OPERATIONALIZED_FOR_SYNTHETIC_NULL_CONTROL",
            "reason": (
                "The real pipeline selected by stability and described the observed "
                "solution as a stable empirical partition, but it did not encode a "
                "numeric pass/fail stability threshold. Stage 8 therefore needs a "
                "per-replicate claim gate before null-scenario evaluation."
            ),
            "selected_candidate_median_ari_min": 0.70,
            "minimum_cluster_eligibility_required": True,
            "justification": (
                "0.70 is not tuned from synthetic outcomes: it reuses the already "
                "frozen Stage-8A S1 profile-recovery ARI threshold as the minimum "
                "per-replicate stability level required to make a stable-partition "
                "claim."
            ),
            "formula": (
                "stable_partition_claim = selection_eligible AND "
                "selected_candidate_ari_median >= 0.70"
            ),
        },
        "severity_null": {
            "severity_score": "unweighted mean of the 13 synthetic features",
            "cv": {
                "splits": 5,
                "repeats": 20,
                "threshold_fitted_on_training_fold_only": True,
                "direction_selected_on_training_fold_only": True,
            },
            "severity_nearly_reconstructs_profiles": {
                "balanced_accuracy_median_min": 0.90,
                "ari_median_min": 0.80,
                "formula": ("balanced_accuracy_median >= 0.90 AND ari_median >= 0.80"),
            },
            "matched_severity_pairs": {
                "tolerance": 0.05,
                "flag": "at least one opposite-profile pair within tolerance",
                "role": (
                    "Descriptive falsification evidence only. This flag does not "
                    "independently suppress or authorize a configurational-profile claim."
                ),
            },
        },
        "governance_null": {
            "synthetic_permutations": 2000,
            "cv": {
                "splits": 5,
                "repeats": 20,
            },
            "strong_governance_association": {
                "bias_corrected_cramers_v_min": 0.50,
                "permutation_p_max_exclusive": 0.05,
                "formula": "V >= 0.50 AND p < 0.05",
            },
            "governance_nearly_reconstructs_profiles": {
                "balanced_accuracy_median_min": 0.90,
                "ari_median_min": 0.80,
                "formula": ("balanced_accuracy_median >= 0.90 AND ari_median >= 0.80"),
            },
        },
        "configurational_profile_claim": {
            "formula": (
                "stable_partition_claim AND NOT severity_nearly_reconstructs_profiles "
                "AND NOT strong_governance_association "
                "AND NOT governance_nearly_reconstructs_profiles"
            ),
            "scenario_application": {
                "S1_CONFIG_TWO_PROFILE": (
                    "Should usually be allowed if recovery is stable and severity "
                    "does not reconstruct the partition."
                ),
                "S2_CORE_BOUNDARY": (
                    "Should usually be allowed if recovery is stable; uncertainty "
                    "is evaluated separately against the planted boundary."
                ),
                "S3_DIRECTIONAL_REACHABILITY": (
                    "Must be established before reachability/directionality metrics "
                    "are interpreted."
                ),
                "S4_SEVERITY_CONTINUUM": (
                    "A configurational claim is false positive unless the severity "
                    "simpler explanation suppresses it."
                ),
                "S5_GOVERNANCE_CONFOUNDED": (
                    "A configurational claim is false positive unless governance "
                    "association/reconstruction suppresses it."
                ),
                "S6_NO_CLUSTER_NULL": (
                    "Any stable_partition_claim is a false stable-profile claim; "
                    "there is no planted simpler categorical explanation."
                ),
            },
        },
        "membership_uncertainty": {
            "existing_real_pipeline_primary_score": None,
            "prospective_primary_for_stage8": "normalized_entropy",
            "selection_status": ("PROSPECTIVELY_FROZEN_BEFORE_SYNTHETIC_PERFORMANCE"),
            "rationale": (
                "The real pipeline computed several continuous certainty quantities "
                "but did not designate one as primary. Normalized binary entropy is "
                "chosen prospectively because it is continuous, bounded, symmetric "
                "under profile-label swapping, and directly quantifies dispersion of "
                "ensemble membership votes."
            ),
            "formula": (
                "H = -sum(p_k * ln p_k) / ln(2), k in {1,2}, ignoring zero terms"
            ),
            "direction": "higher = more uncertain",
            "primary_boundary_metric": {
                "scenario": "S2_CORE_BOUNDARY",
                "target": "true_boundary = 1",
                "score": "normalized_entropy",
                "metric": "ROC-AUC",
            },
            "secondary_scores": {
                "reference_profile_probability": "lower = more uncertain",
                "family_consistency": "lower = more uncertain",
                "membership_margin": "lower = more uncertain",
                "consensus_gap": "lower = more uncertain",
            },
            "certainty_classes": {
                "CORE": (
                    "reference_profile_probability >= 0.90 AND "
                    "family_consistency >= 0.80"
                ),
                "BOUNDARY": (
                    "reference_profile_probability < 0.75 OR "
                    "family_consistency < 0.60"
                ),
                "HALO": "otherwise",
            },
        },
        "reachability_evaluation": {
            "only_after": [
                "reference model selected without truth labels",
                "stable/configurational claim state computed",
                "membership uncertainty computed",
            ],
            "single_model_reachable": (
                "At least one exact <=4-feature candidate transitions the frozen "
                "reference model and passes the frozen target-manifold 5-NN plausibility rule."
            ),
            "robust_reachable_tau_090": (
                "At least one saved single-model reachable candidate receives "
                "target-profile support >=0.90 across the eligible heterogeneous ensemble."
            ),
            "sensitivity_taus": [0.80, 0.95],
            "diagnostic_not_causal": True,
        },
        "synthetic_truth_usage_policy": {
            "truth_is_hidden_during": [
                "representation fitting",
                "clustering",
                "candidate selection",
                "stability evaluation",
                "ensemble alignment except alignment to the selected synthetic reference partition",
                "severity/governance prediction of discovered cluster IDs",
                "counterfactual search",
                "ensemble robustness evaluation",
            ],
            "truth_is_used_only_after_pipeline_outputs_are_frozen_for_replicate": [
                "ARI/NMI versus planted profiles",
                "boundary AUROC",
                "oracle reachability precision/recall/F1",
                "direction recovery",
                "label-swap invariance",
                "false-claim accounting",
            ],
        },
        "stage8a_success_criteria_reaffirmed_without_change": (
            stage8a["predeclared_success_criteria"]
        ),
        "no_post_result_changes": [
            "no threshold relaxation",
            "no alternative primary uncertainty score after seeing S2",
            "no alternate stable-claim threshold after seeing S4-S6",
            "no relabeling of S3 accessible direction to favor numeric Profile 1 or Profile 2",
            "no restoration of empirical motif claim",
        ],
        "next_stage": {
            "id": "STAGE_8D",
            "name": "One-replicate end-to-end synthetic pipeline smoke",
            "rule": (
                "Run exactly one non-official pipeline-performance smoke replicate "
                "per scenario using these frozen evaluator semantics. Do not count "
                "those six smoke datasets toward the 100 official replicates."
            ),
        },
        "gate_status": "PASS_STAGE_8C_EVALUATOR_FREEZE",
    }

    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(
        json.dumps(
            evaluator,
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
                "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "gate_status": "PASS_STAGE_8C_EVALUATOR_FREEZE",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8C — SYNTHETIC EVALUATOR FREEZE ===\n")

    print("Model selection:")
    print("  eligibility: full-data min cluster >=5 AND resampling min cluster >=5")
    print("  primary: maximum median resampling ARI")
    print("  tied equivalent partitions (ARI>=.95): Silhouette, DB, CH, lower k")
    print(
        "  tied non-equivalent: ARI q025, weakest-cluster Jaccard, Silhouette, lower k"
    )

    print("\nSeverity null:")
    print("  severity nearly reconstructs = median BA >= .90 AND median ARI >= .80")
    print("  matched-severity pairs remain descriptive only")

    print("\nGovernance null:")
    print("  strong association = bias-corrected Cramer's V >= .50 AND p < .05")
    print("  nearly reconstructs = median BA >= .90 AND median ARI >= .80")

    print("\nPrimary uncertainty metric:")
    print("  normalized_entropy (prospectively selected; higher = more uncertain)")
    print("  S2 primary metric = boundary-vs-core ROC-AUC")

    print("\nStable/configurational claim gate:")
    print(
        "  stable partition = eligible selected model AND median stability ARI >= .70"
    )
    print(
        "  configurational claim additionally requires severity/governance flags to be false"
    )

    print("\nProspective-status check:")
    print(
        "  official synthetic performance artifacts absent:", official_results_absent()
    )

    print("\n=== FREEZE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_8C_EVALUATOR_FREEZE")
    print("Frozen evaluator:", FREEZE_PATH.relative_to(ROOT))
    print("Next: Stage 8D one-replicate end-to-end pipeline smoke per scenario.")
    print("Do NOT start the 600 official replicates yet.")


if __name__ == "__main__":
    main()
