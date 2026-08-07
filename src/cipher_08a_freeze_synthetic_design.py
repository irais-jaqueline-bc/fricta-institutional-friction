from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

THESIS_V3_PATH = ROOT / "cipher" / "design" / "cipher_thesis_revision_v3.json"
STAGE7B_AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage7b_motif_retirement_audit.json"
)

DESIGN_PATH = ROOT / "cipher" / "design" / "stage8_synthetic_validation_freeze_v1.json"
AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8a_synthetic_design_freeze_audit.json"
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


def main() -> None:
    if DESIGN_PATH.exists():
        raise FileExistsError(
            f"Synthetic validation freeze already exists: {DESIGN_PATH}"
        )

    thesis = load_json(THESIS_V3_PATH)
    stage7b = load_json(STAGE7B_AUDIT_PATH)

    checks = {
        "thesis_v3_is_frozen_after_motif_stop": (
            thesis.get("gate_status") == "PASS_STAGE_7B_MOTIF_RETIREMENT"
        ),
        "stage7b_audit_passed": (
            stage7b.get("gate_status") == "PASS_STAGE_7B_MOTIF_RETIREMENT"
        ),
        "empirical_motif_claim_is_retired": (
            thesis.get("empirical_motif_claim", {}).get("status") == "RETIRED"
        ),
        "stage8_is_required_next": (
            thesis.get("stage8_policy", {}).get("status") == "REQUIRED_NEXT"
        ),
    }

    if not all(checks.values()):
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("\nGATE STATUS: FAIL_STAGE_8A_SYNTHETIC_DESIGN_FREEZE")
        raise SystemExit(1)

    feature_names = [
        "willingness_constraint_score",
        "digital_usage_constraint_score",
        "training_deficit_score",
        "device_constraint",
        "digital_tool_variety_constraint",
        "internet_stability_constraint",
        "staffing_constraint_score",
        "time_constraint_score",
        "administrative_disorganization_constraint",
        "recording_system_constraint",
        "system_change_resistance_constraint",
        "admin_time_load_constraint",
        "resource_constraint_score",
    ]

    design = {
        "version": "STAGE8_SYNTHETIC_VALIDATION_FREEZE_V1",
        "status": "FROZEN_BEFORE_SYNTHETIC_RESULTS",
        "parent_thesis_revision": {
            "path": str(THESIS_V3_PATH.relative_to(ROOT)),
            "sha256": sha256_file(THESIS_V3_PATH),
        },
        "primary_goal": (
            "Test whether the retained CIPHER pipeline recovers stable "
            "configurational profiles, localizes uncertainty to ambiguous "
            "regions, distinguishes structural reachability from forced "
            "counterfactuals, and avoids unsupported profile claims under "
            "severity-only, governance-confounded, and no-cluster data."
        ),
        "claims_under_test": [
            "heterogeneous-ensemble profile stability",
            "membership uncertainty localizes ambiguous boundary regions",
            "sparse plausible counterfactual reachability can be heterogeneous",
            "ensemble robustness filters single-model reachable transitions",
            "directional reachability asymmetry can be recovered without label bias",
            "severity and governance falsification suppress simpler explanations",
            "the method does not force profile/reachability claims under null structure",
        ],
        "claims_not_under_test_as_headline": [
            "empirical multi-feature motif discovery",
            "causal intervention efficacy",
            "national representativeness",
        ],
        "common_simulation": {
            "n_institutions": 80,
            "n_features": 13,
            "feature_names": feature_names,
            "feature_domain": [0.0, 1.0],
            "master_seed": 20260807,
            "official_replicates_per_scenario": 100,
            "scenario_count": 6,
            "total_structural_replicates": 600,
            "profile_labels_are_semantically_arbitrary": True,
            "label_swap_policy": (
                "For every structured two-profile replicate, randomly swap "
                "numeric profile labels with probability 0.5 before evaluation."
            ),
        },
        "synthetic_compute_plan": {
            "reason": (
                "Exact Stage-5C enumeration and a 1000-member perturbation ensemble "
                "over 600 datasets would be unnecessarily expensive. Synthetic "
                "validation therefore uses a frozen computational proxy plus a "
                "convergence audit against the real-case ensemble size."
            ),
            "primary_synthetic_ensemble": {
                "members_total": 200,
                "members_per_family": {
                    "R0_WARD": 50,
                    "R1_PCA85_WARD": 50,
                    "R0_KMEANS": 50,
                    "R1_PCA85_KMEANS": 50,
                },
                "row_sample_fraction": 0.80,
                "features_sampled": 11,
                "pca_variance_threshold": 0.85,
                "k": 2,
                "kmeans_n_init": 25,
                "ward_inductive_rule": "nearest centroid in fitted representation",
                "ward_min_extension_fidelity": 0.95,
            },
            "convergence_audit": {
                "replicates_per_scenario": 10,
                "comparison_ensemble_members": 1000,
                "required_certainty_rank_correlation_spearman": 0.95,
                "required_tau090_reachability_agreement": 0.95,
                "policy_if_failed": (
                    "Do not interpret the 200-member synthetic ensemble as a valid "
                    "proxy; rerun official synthetic evaluation with 1000 members "
                    "for the affected metric family."
                ),
            },
            "counterfactual_query_budget": {
                "structured_scenarios_only": [
                    "S1_CONFIG_TWO_PROFILE",
                    "S2_CORE_BOUNDARY",
                    "S3_DIRECTIONAL_REACHABILITY",
                ],
                "query_institutions_per_replicate": 20,
                "sampling_policy": (
                    "Stratified by true profile and, where defined, true core/boundary "
                    "or oracle-reachability status. Sampling seed is derived only "
                    "from master seed + scenario + replicate."
                ),
                "exact_search_max_changed_features": 4,
                "candidate_values": "levels observed within the synthetic replicate",
                "plausibility_rule": (
                    "5-NN Euclidean distance to target-profile observations; "
                    "candidate must not exceed the target-profile 95th percentile "
                    "of within-profile 5-NN distance."
                ),
                "cost": "weighted L1 by replicate IQR + 0.25 * L0",
                "ensemble_support_primary_tau": 0.90,
                "ensemble_support_sensitivity": [0.80, 0.95],
            },
        },
        "base_generator": {
            "center": 0.50,
            "profile_deviation": 0.24,
            "core_noise_sd": 0.06,
            "boundary_noise_sd": 0.04,
            "continuum_noise_sd": 0.08,
            "clip_to_unit_interval": True,
            "balanced_profile_sizes_when_profiles_exist": True,
            "configuration_vector": [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0],
            "note": (
                "The configuration vector sums to zero, so the two primary "
                "prototype profiles have equal mean aggregate severity by construction."
            ),
        },
        "scenarios": [
            {
                "id": "S1_CONFIG_TWO_PROFILE",
                "truth": "two stable configurational profiles",
                "generator": (
                    "Profile A = center + deviation * configuration_vector; "
                    "Profile B = center - deviation * configuration_vector; "
                    "independent truncated-normal noise with core_noise_sd."
                ),
                "true_profiles": True,
                "true_boundary": False,
                "oracle_reachability": "symmetric baseline; no planted directional advantage",
                "purpose": (
                    "Test stable profile recovery without relying on aggregate severity."
                ),
            },
            {
                "id": "S2_CORE_BOUNDARY",
                "truth": "stable cores plus ambiguous boundary",
                "generator": (
                    "70% of observations are generated from the two core prototypes. "
                    "30% are convex mixtures of the prototypes with lambda Uniform(0.35,0.65), "
                    "plus boundary_noise_sd. Boundary observations receive a true boundary flag."
                ),
                "true_profiles": True,
                "true_boundary": True,
                "boundary_fraction": 0.30,
                "purpose": (
                    "Test whether ensemble uncertainty localizes genuinely ambiguous geometry."
                ),
            },
            {
                "id": "S3_DIRECTIONAL_REACHABILITY",
                "truth": "stable profiles with planted asymmetric sparse accessibility",
                "generator": (
                    "Start from S1 prototypes. In each replicate, randomly choose one "
                    "source profile as the accessible direction. For 40% of that source "
                    "profile, generate bridge observations whose target-manifold mismatch "
                    "is concentrated in three randomly selected discriminative features. "
                    "For the reverse direction, require mismatch across at least six "
                    "discriminative features. The accessible direction is randomized "
                    "before numeric label assignment."
                ),
                "true_profiles": True,
                "true_boundary": False,
                "oracle_reachability": True,
                "accessible_fraction_within_source_profile": 0.40,
                "accessible_changed_features": 3,
                "reverse_minimum_required_features": 6,
                "purpose": (
                    "Test directional reachability recovery without privileging Profile 1->2."
                ),
            },
            {
                "id": "S4_SEVERITY_CONTINUUM",
                "truth": "one-dimensional severity continuum; no categorical profiles",
                "generator": (
                    "Draw latent severity z ~ Uniform(0,1). Each feature is a monotone "
                    "noisy transform of z with randomly signed-free positive loading "
                    "between 0.65 and 1.00 and continuum_noise_sd, clipped to [0,1]."
                ),
                "true_profiles": False,
                "true_boundary": False,
                "purpose": (
                    "Test whether severity falsification prevents a configurational-profile claim."
                ),
            },
            {
                "id": "S5_GOVERNANCE_CONFOUNDED",
                "truth": "governance-driven feature offsets; no latent friction profile beyond governance",
                "generator": (
                    "Assign four governance categories with probabilities matching "
                    "[0.33, 0.27, 0.21, 0.19]. Governance-specific offsets affect "
                    "feature blocks, but there is no independent latent profile. "
                    "Overall noise uses continuum_noise_sd."
                ),
                "true_profiles": False,
                "true_boundary": False,
                "governance_categories": 4,
                "purpose": (
                    "Test whether governance falsification suppresses a confounded profile claim."
                ),
            },
            {
                "id": "S6_NO_CLUSTER_NULL",
                "truth": "single unimodal population with mild correlation; no profiles",
                "generator": (
                    "Draw one 13-dimensional Gaussian-copula population with pairwise "
                    "correlation 0.20, transform marginals to Beta(2.5,2.5), and clip "
                    "numerically to [0,1]. No mixture components are present."
                ),
                "true_profiles": False,
                "true_boundary": False,
                "purpose": (
                    "Estimate false stable-profile and forced-counterfactual behavior under null structure."
                ),
            },
        ],
        "pipeline_under_test": {
            "discovery": {
                "representations": ["R0_STANDARDIZED", "R1_PCA85"],
                "algorithms": ["HAC_WARD", "KMEANS"],
                "candidate_k": [2, 3, 4, 5, 6],
                "selection_metrics": [
                    "silhouette",
                    "davies_bouldin",
                    "calinski_harabasz",
                    "perturbation_stability",
                    "minimum_cluster_size",
                ],
                "minimum_cluster_size": 5,
                "no_use_of_true_labels_during_selection": True,
            },
            "uncertainty": {
                "same_core_halo_boundary_logic_as_real_case": True,
                "core_thresholds": {
                    "reference_probability_min": 0.90,
                    "family_consistency_min": 0.80,
                },
                "boundary_thresholds": {
                    "reference_probability_below": 0.75,
                    "family_consistency_below": 0.60,
                },
            },
            "falsification": {
                "severity_null": "same train/test thresholding logic as Stage 3",
                "governance_null": "same permutation/CV logic as Stage 3",
                "governance_permutations": 2000,
                "note": (
                    "Synthetic permutations are reduced from 10,000 for compute; "
                    "the threshold and interpretation are unchanged."
                ),
            },
            "counterfactuals": {
                "exact_enumeration": True,
                "max_features": 4,
                "diagnostic_not_causal": True,
                "plausibility_required": True,
                "ensemble_robustness_required_for_robust_label": True,
            },
        },
        "primary_metrics": {
            "structured_profile_recovery": [
                "ARI",
                "NMI",
                "selected_k_accuracy",
            ],
            "uncertainty_localization": [
                "AUROC_boundary_vs_core_using_membership_uncertainty",
                "median_uncertainty_boundary_minus_core",
            ],
            "reachability": [
                "oracle_reachability_precision",
                "oracle_reachability_recall",
                "oracle_reachability_F1",
                "robust_reachability_precision",
                "robust_reachability_recall",
                "robust_reachability_F1",
            ],
            "directionality": [
                "accessible_direction_recovery_accuracy",
                "label_swap_invariance",
            ],
            "null_control": [
                "false_configurational_profile_claim_rate",
                "false_robust_reachability_claim_rate",
            ],
        },
        "predeclared_success_criteria": {
            "S1_CONFIG_TWO_PROFILE": {
                "median_ARI_min": 0.70,
                "selected_k_equals_2_rate_min": 0.80,
            },
            "S2_CORE_BOUNDARY": {
                "median_ARI_min": 0.60,
                "boundary_uncertainty_AUROC_min": 0.80,
                "median_boundary_uncertainty_gt_core": True,
            },
            "S3_DIRECTIONAL_REACHABILITY": {
                "median_profile_ARI_min": 0.60,
                "oracle_reachability_F1_min": 0.70,
                "robust_reachability_precision_min": 0.80,
                "accessible_direction_recovery_rate_min": 0.80,
                "label_swap_invariance_rate_min": 0.95,
            },
            "S4_SEVERITY_CONTINUUM": {
                "false_configurational_profile_claim_rate_max": 0.10,
                "require_severity_falsification_to_flag_simpler_explanation": True,
            },
            "S5_GOVERNANCE_CONFOUNDED": {
                "false_configurational_profile_claim_rate_max": 0.10,
                "require_governance_falsification_to_flag_simpler_explanation": True,
            },
            "S6_NO_CLUSTER_NULL": {
                "false_stable_profile_claim_rate_max": 0.10,
                "false_robust_reachability_claim_rate_max": 0.05,
            },
        },
        "claim_policy_after_synthetic_validation": {
            "full_support": (
                "Retain the uncertainty-aware ensemble-robust reachability thesis "
                "only if all scenario-specific primary criteria pass."
            ),
            "partial_support": (
                "If profile recovery passes but reachability/directionality criteria fail, "
                "retain only the stable-profile + uncertainty contribution."
            ),
            "null_failure": (
                "If severity, governance, or no-cluster false-claim criteria fail, "
                "the corresponding CIPHER claim must be weakened or removed."
            ),
            "no_threshold_relaxation_after_results": True,
            "all_failures_reported": True,
        },
        "motif_policy": {
            "empirical_claim_remains_retired": True,
            "synthetic_role": (
                "Optional negative-control diagnostic only. Do not restore motif "
                "discovery as a headline contribution from synthetic success."
            ),
        },
        "next_stage": {
            "id": "STAGE_8B",
            "name": "Synthetic generator implementation + smoke/audit",
            "rule": (
                "Implement generators exactly as frozen, run one smoke replicate per "
                "scenario, and audit truth labels/geometry before any official 100-replicate run."
            ),
        },
        "source_hashes": {
            "thesis_v3": sha256_file(THESIS_V3_PATH),
            "stage7b_audit": sha256_file(STAGE7B_AUDIT_PATH),
        },
        "gate_status": "PASS_STAGE_8A_SYNTHETIC_DESIGN_FREEZE",
    }

    DESIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    DESIGN_PATH.write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "checks": checks,
                "design_path": str(DESIGN_PATH.relative_to(ROOT)),
                "design_sha256": sha256_file(DESIGN_PATH),
                "gate_status": "PASS_STAGE_8A_SYNTHETIC_DESIGN_FREEZE",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8A — SYNTHETIC VALIDATION DESIGN FREEZE ===\n")

    print("Official scenarios: 6")
    print("Official structural replicates: 100/scenario = 600 total")
    print("Synthetic primary ensemble: 200 members (50/family)")
    print("1000-member convergence audit: 10 replicates/scenario")
    print("Exact-CF query budget: 20 institutions/replicate in S1-S3")

    print("\nScenarios:")
    for scenario in design["scenarios"]:
        print(f"  {scenario['id']}: {scenario['truth']}")

    print("\nPrimary scientific safeguards:")
    print("  - numeric profile labels randomly swapped")
    print("  - asymmetric reachable direction randomly assigned")
    print("  - severity-only null")
    print("  - governance-confounded null")
    print("  - no-cluster null")
    print("  - no post-result threshold relaxation")

    print("\nPredeclared success criteria:")
    print("  S1 median ARI >= 0.70; k=2 recovery >= 0.80")
    print("  S2 boundary-uncertainty AUROC >= 0.80")
    print("  S3 reachability F1 >= 0.70; robust precision >= 0.80")
    print("     direction recovery >= 0.80; label-swap invariance >= 0.95")
    print("  S4/S5 false configurational claim rate <= 0.10")
    print("  S6 false stable-profile rate <= 0.10; false robust-CF rate <= 0.05")

    print("\nNext step:")
    print("  Stage 8B = implement generators + one smoke/audit replicate per scenario")
    print("  Do NOT start the 600 official synthetic replicates yet.")

    print("\n=== FREEZE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_8A_SYNTHETIC_DESIGN_FREEZE")
    print("Frozen design:", DESIGN_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
