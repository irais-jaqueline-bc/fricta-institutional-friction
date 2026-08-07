from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "icdm" / "outputs" / "audit" / "icdm_feature_audit.csv"
DESIGN_DIR = PROJECT_ROOT / "icdm" / "design"

MANIFEST_PATH = DESIGN_DIR / "feature_manifest.csv"
CONFIG_PATH = DESIGN_DIR / "experiment_config.json"
PROTOCOL_PATH = DESIGN_DIR / "model_selection_protocol.md"


FINAL_DECISIONS = {
    # Primary clustering representation
    "device_constraint": {
        "final_role": "PRIMARY",
        "concept": "Device availability",
        "decision_reason": (
            "Direct access constraint. Retained despite correlation because it measures "
            "hardware availability rather than tool use or administrative digitization."
        ),
    },
    "internet_stability_constraint": {
        "final_role": "PRIMARY",
        "concept": "Internet stability",
        "decision_reason": "Independent connectivity constraint.",
    },
    "digital_tool_variety_constraint": {
        "final_role": "PRIMARY",
        "concept": "Digital tool diversity",
        "decision_reason": (
            "Measures breadth of tools currently used; conceptually distinct from device count."
        ),
    },
    "recording_system_constraint": {
        "final_role": "PRIMARY",
        "concept": "Administrative recording digitization",
        "decision_reason": (
            "Measures paper/Excel/mixed/software recording maturity; distinct from device access."
        ),
    },
    "admin_time_load_constraint": {
        "final_role": "PRIMARY",
        "concept": "Daily administrative workload",
        "decision_reason": (
            "Measures reported hours spent on administration; distinct from perceived lack of time."
        ),
    },
    "administrative_disorganization_constraint": {
        "final_role": "PRIMARY",
        "concept": "Administrative organization",
        "decision_reason": "Direct organizational-process constraint.",
    },
    "system_change_resistance_constraint": {
        "final_role": "PRIMARY",
        "concept": "System transition resistance",
        "decision_reason": (
            "Measures difficulty changing operational systems; distinct from willingness to try a tool."
        ),
    },
    "digital_usage_constraint_score": {
        "final_role": "PRIMARY",
        "concept": "Digital usage frequency",
        "decision_reason": (
            "Measures routine digital exposure/use; retained as a distinct behavioral indicator."
        ),
    },
    "time_constraint_score": {
        "final_role": "PRIMARY",
        "concept": "Perceived time scarcity",
        "decision_reason": (
            "Measures perceived lack of time; distinct from actual administrative-hour burden."
        ),
    },
    "staffing_constraint_score": {
        "final_role": "PRIMARY",
        "concept": "Staffing scarcity",
        "decision_reason": "Direct operational-capacity constraint.",
    },
    "training_deficit_score": {
        "final_role": "PRIMARY",
        "concept": "Training deficit",
        "decision_reason": "Direct human-capacity constraint.",
    },
    "resource_constraint_score": {
        "final_role": "PRIMARY",
        "concept": "General resource scarcity",
        "decision_reason": (
            "Broad resource-capacity signal retained because it is not a deterministic duplicate "
            "of device or staffing measures."
        ),
    },
    "willingness_constraint_score": {
        "final_role": "PRIMARY",
        "concept": "Adoption willingness",
        "decision_reason": (
            "Measures willingness to try a tool; distinct from system-change resistance."
        ),
    },
    # Not used to form the primary clusters
    "implementation_difficulty_constraint": {
        "final_role": "VALIDATION_ONLY",
        "concept": "Experienced implementation difficulty",
        "decision_reason": (
            "Reserved as an external descriptive criterion. It must not enter clustering because "
            "the study may later compare clusters against experienced implementation difficulty. "
            "Missing values are not imputed for validation."
        ),
    },
    "previous_implementation_constraint": {
        "final_role": "SENSITIVITY_ONLY",
        "concept": "Prior digital implementation exposure",
        "decision_reason": (
            "Binary historical exposure proxy, not a current continuous friction condition. "
            "May be added only in a pre-specified sensitivity analysis."
        ),
    },
    "perceived_utility_constraint": {
        "final_role": "SENSITIVITY_ONLY",
        "concept": "Perceived digital utility",
        "decision_reason": (
            "Auxiliary attitudinal indicator with low variance. Excluded from the primary "
            "representation and used only in sensitivity analysis."
        ),
    },
    "pilot_openness_constraint": {
        "final_role": "METADATA_ONLY",
        "concept": "Pilot participation openness",
        "decision_reason": (
            "Research-participation metadata rather than digital-adoption friction. Never used "
            "to form clusters."
        ),
    },
}


def load_audit() -> pd.DataFrame:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Feature audit not found: {AUDIT_PATH}\n"
            "Run src/icdm_feature_audit.py first."
        )

    audit = pd.read_csv(AUDIT_PATH)

    expected = set(FINAL_DECISIONS)
    observed = set(audit["feature"])

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    if missing or unexpected:
        raise ValueError(
            "Feature mismatch between audit and frozen decisions.\n"
            f"Missing from audit: {missing}\n"
            f"Unexpected in audit: {unexpected}"
        )

    return audit


def build_manifest(audit: pd.DataFrame) -> pd.DataFrame:
    manifest = audit.copy()

    manifest["concept"] = manifest["feature"].map(
        lambda feature: FINAL_DECISIONS[feature]["concept"]
    )
    manifest["final_role"] = manifest["feature"].map(
        lambda feature: FINAL_DECISIONS[feature]["final_role"]
    )
    manifest["final_decision_reason"] = manifest["feature"].map(
        lambda feature: FINAL_DECISIONS[feature]["decision_reason"]
    )

    role_order = {
        "PRIMARY": 0,
        "VALIDATION_ONLY": 1,
        "SENSITIVITY_ONLY": 2,
        "METADATA_ONLY": 3,
        "EXCLUDED": 4,
    }

    manifest["_role_order"] = manifest["final_role"].map(role_order)
    manifest = manifest.sort_values(
        ["_role_order", "conceptual_group", "feature"]
    ).drop(columns="_role_order")

    return manifest


def build_config(manifest: pd.DataFrame) -> dict:
    role_to_features = {
        role: manifest.loc[manifest["final_role"] == role, "feature"].tolist()
        for role in [
            "PRIMARY",
            "VALIDATION_ONLY",
            "SENSITIVITY_ONLY",
            "METADATA_ONLY",
            "EXCLUDED",
        ]
    }

    return {
        "project": "FRICTA ICDM Teen Research Track",
        "design_status": "FROZEN_BEFORE_CLUSTERING",
        "random_seed": 42,
        "input_dataset": "data/processed/fricta_scored.csv",
        "id_column": "institution_id",
        "primary_features": role_to_features["PRIMARY"],
        "validation_only_features": role_to_features["VALIDATION_ONLY"],
        "sensitivity_only_features": role_to_features["SENSITIVITY_ONLY"],
        "metadata_only_features": role_to_features["METADATA_ONLY"],
        "excluded_features": role_to_features["EXCLUDED"],
        "never_model_columns": [
            "institution_id",
            "state",
            "ICI",
            "OCI",
            "OLI",
            "HCARI",
            "AFS_baseline",
            "AFS_theoretical",
            "friction_archetype",
        ],
        "missing_data": {
            "primary_analysis": (
                "Median imputation fitted inside each resampling/training partition; "
                "current primary matrix has no missing values."
            ),
            "validation_implementation_difficulty": (
                "Complete-case analysis only; do not impute missing implementation difficulty."
            ),
        },
        "scaling": {
            "primary": "StandardScaler fitted within each resampling/training partition",
            "sensitivity": "Original [0,1] scale without standardization",
        },
        "representations": {
            "R0": "Standardized primary features",
            "R1": "PCA fitted on standardized primary features; retain cumulative variance >= 0.85",
        },
        "clustering": {
            "k_values": [2, 3, 4, 5, 6],
            "kmeans": {"n_init": 50, "random_state": 42},
            "hac_ward": {"linkage": "ward", "metric": "euclidean"},
            "gmm": {
                "covariance_type": "diag",
                "n_init": 50,
                "reg_covar": 1e-6,
                "random_state": 42,
            },
        },
        "internal_metrics": [
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "bic_for_gmm",
            "minimum_cluster_size",
        ],
        "minimum_cluster_size": 5,
        "stability": {
            "method": "subsampling_without_replacement",
            "iterations": 1000,
            "sample_fraction": 0.80,
            "metrics": [
                "adjusted_rand_index",
                "clusterwise_jaccard",
                "consensus_matrix",
            ],
        },
        "correlation_handling": {
            "rule": (
                "Do not automatically remove features solely because |r| >= 0.80 "
                "when their survey meanings are conceptually distinct."
            ),
            "pre_registered_sensitivity": (
                "Run leave-one-feature-out checks for every feature participating "
                "in a high-correlation pair."
            ),
        },
        "theory_alignment": {
            "reference": "data/processed/friction_archetypes.csv",
            "interpretation": "structural concordance, not ground-truth validation",
            "metrics": ["ARI", "NMI", "contingency_matrix"],
        },
    }


def write_protocol() -> None:
    text = """# Frozen Model-Selection Protocol

## Candidate solutions

Evaluate K-Means, HAC-Ward, and diagonal-covariance GMM for k = 2, 3, 4, 5, 6
on both R0 (standardized primary features) and R1 (PCA retaining at least 85%
cumulative explained variance).

## Exclusion rule

Discard any solution containing a cluster with fewer than 5 institutions.

## Selection rule

1. Compare internal validity using Silhouette, Davies-Bouldin, and
   Calinski-Harabasz; use BIC additionally for GMM.
2. Retain competitive non-degenerate solutions.
3. Evaluate 1,000 80% subsamples without replacement.
4. Prefer the solution with the strongest median stability.
5. If solutions are practically tied, prefer:
   a. higher Silhouette;
   b. fewer clusters;
   c. clearer institutional interpretation.
6. Do not change this rule after inspecting attractive visualizations.

## Correlated-feature policy

High correlation alone does not force feature deletion when survey items measure
different institutional concepts. Every high-correlation feature will be subjected
to a pre-registered leave-one-feature-out sensitivity test.

## FRICTA v1 comparison

Legacy rule-based profiles are used only for theory-data structural concordance.
They are not labels, ground truth, or an external validation target.
"""
    PROTOCOL_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)

    audit = load_audit()
    manifest = build_manifest(audit)
    config = build_config(manifest)

    manifest.to_csv(MANIFEST_PATH, index=False)
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_protocol()

    print("\n=== FROZEN FEATURE ROLES ===\n")
    print(
        manifest[
            [
                "feature",
                "concept",
                "final_role",
                "n_missing",
                "variance",
            ]
        ].to_string(index=False)
    )

    print("\n=== ROLE COUNTS ===\n")
    print(manifest["final_role"].value_counts().to_string())

    print("\n=== GENERATED DESIGN FILES ===\n")
    print(MANIFEST_PATH)
    print(CONFIG_PATH)
    print(PROTOCOL_PATH)

    print("\nGATE STATUS: DESIGN FROZEN. " "Next step is src/icdm_prepare_features.py.")


if __name__ == "__main__":
    main()
