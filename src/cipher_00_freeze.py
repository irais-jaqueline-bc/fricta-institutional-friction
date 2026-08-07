from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRICTA_CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
CIPHER_CONFIG_PATH = (
    PROJECT_ROOT / "cipher" / "design" / "cipher_experiment_config.json"
)
PRIMARY_MATRIX_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
)
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
SELECTED_MODEL_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "selected_model.json"
)

DESIGN_DIR = PROJECT_ROOT / "cipher" / "design"
AUDIT_DIR = PROJECT_ROOT / "cipher" / "outputs" / "audit"

ACTIONABILITY_PATH = DESIGN_DIR / "actionability_manifest.csv"
FREEZE_PATH = DESIGN_DIR / "analysis_freeze.json"
HASH_PATH = AUDIT_DIR / "input_hashes.json"
NOVELTY_PATH = DESIGN_DIR / "novelty_claim_log.md"
CONFIG_SNAPSHOT_PATH = DESIGN_DIR / "cipher_experiment_config_frozen.json"
INITIALIZATION_REPORT_PATH = AUDIT_DIR / "stage0_initialization_report.json"

EXPECTED_FEATURES = [
    "device_constraint",
    "internet_stability_constraint",
    "digital_tool_variety_constraint",
    "recording_system_constraint",
    "admin_time_load_constraint",
    "administrative_disorganization_constraint",
    "system_change_resistance_constraint",
    "digital_usage_constraint_score",
    "time_constraint_score",
    "staffing_constraint_score",
    "training_deficit_score",
    "resource_constraint_score",
    "willingness_constraint_score",
]

FORBIDDEN_EXACT_COLUMNS = {
    "ici",
    "oci",
    "oli",
    "hcari",
    "afs",
    "afs_baseline",
    "afs_theoretical",
    "archetype",
    "friction_profile",
    "fricta_archetype",
    "fricta_profile",
    "profile_label",
    "cluster_label",
}

FORBIDDEN_PREFIXES = (
    "ici_",
    "oci_",
    "oli_",
    "hcari_",
    "afs_",
    "fricta_",
)

FORBIDDEN_SUFFIXES = (
    "_archetype",
    "_friction_profile",
    "_profile_label",
)


FEATURE_METADATA = {
    "device_constraint": {
        "domain": "Infrastructure",
        "actionability_class": "DIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Can change through device acquisition, replacement, or shared-access redesign.",
    },
    "internet_stability_constraint": {
        "domain": "Infrastructure",
        "actionability_class": "DIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Can change through connectivity upgrades, redundancy, or provider changes.",
    },
    "digital_tool_variety_constraint": {
        "domain": "Infrastructure and integration",
        "actionability_class": "DIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Can change by adopting or consolidating appropriate digital tools.",
    },
    "recording_system_constraint": {
        "domain": "Infrastructure and integration",
        "actionability_class": "DIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Can change through digitization or redesign of institutional record systems.",
    },
    "admin_time_load_constraint": {
        "domain": "Operational capacity",
        "actionability_class": "INDIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "May change through workflow redesign, automation, or redistribution of tasks.",
    },
    "administrative_disorganization_constraint": {
        "domain": "Organizational capacity",
        "actionability_class": "INDIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "May change through process standardization, role clarity, and governance routines.",
    },
    "system_change_resistance_constraint": {
        "domain": "Organizational capacity",
        "actionability_class": "INDIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "May change through implementation support, participation, and change management.",
    },
    "digital_usage_constraint_score": {
        "domain": "Digital integration",
        "actionability_class": "INDIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Usage can change after access, workflow integration, training, and support improve.",
    },
    "time_constraint_score": {
        "domain": "Operational capacity",
        "actionability_class": "INDIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "May change through staffing, scheduling, simplification, or task redistribution.",
    },
    "staffing_constraint_score": {
        "domain": "Human capacity",
        "actionability_class": "CONTEXTUALLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Can change, but often depends on budgets, hiring conditions, and external constraints.",
    },
    "training_deficit_score": {
        "domain": "Human capacity",
        "actionability_class": "DIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Can change through training, onboarding, and continuing technical support.",
    },
    "resource_constraint_score": {
        "domain": "Resource capacity",
        "actionability_class": "CONTEXTUALLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "Can change but may depend on funding, donations, policy, or external support.",
    },
    "willingness_constraint_score": {
        "domain": "Organizational readiness",
        "actionability_class": "INDIRECTLY_MODIFIABLE",
        "realistic_improvement_direction": "DECREASE_FRICTION",
        "diagnostic_counterfactual_direction": "BOTH",
        "rationale": "May change through trust, usefulness evidence, participation, and implementation experience.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize or finalize CIPHER Stage 0."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--initialize",
        action="store_true",
        help="Validate inputs and create the actionability manifest for review.",
    )
    mode.add_argument(
        "--finalize",
        action="store_true",
        help="Freeze Stage 0 after every manifest row has been confirmed.",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def ensure_directories() -> None:
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> dict:
    required_paths = [
        FRICTA_CONFIG_PATH,
        CIPHER_CONFIG_PATH,
        PRIMARY_MATRIX_PATH,
        FINAL_LABELS_PATH,
        SELECTED_MODEL_PATH,
    ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Missing required inputs:\n- " + "\n- ".join(missing_paths)
        )

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    selected_model = load_json(SELECTED_MODEL_PATH)

    id_column = fricta_config["id_column"]
    configured_features = fricta_config["primary_features"]

    if len(configured_features) != len(set(configured_features)):
        raise ValueError(
            "The frozen primary feature list contains duplicate names.\n"
            f"Configured: {configured_features}"
        )

    if set(configured_features) != set(EXPECTED_FEATURES):
        missing = sorted(set(EXPECTED_FEATURES) - set(configured_features))
        unexpected = sorted(set(configured_features) - set(EXPECTED_FEATURES))

        raise ValueError(
            "The frozen primary feature set does not match the "
            "expected 13-feature CIPHER design.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}\n"
            f"Configured: {configured_features}\n"
            f"Expected set: {EXPECTED_FEATURES}"
        )

    if configured_features != EXPECTED_FEATURES:
        print(
            "\nNOTE: The 13 primary features are correct, but their order "
            "differs from the human-readable CIPHER list. "
            "The pipeline will preserve the frozen order from "
            "icdm/design/experiment_config.json.\n"
        )

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)

    required_primary_columns = [id_column] + configured_features

    missing_primary_columns = [
        column for column in required_primary_columns if column not in primary.columns
    ]

    if missing_primary_columns:
        raise KeyError(
            "Primary matrix is missing columns:\n- "
            + "\n- ".join(missing_primary_columns)
        )

    if primary[id_column].duplicated().any():
        raise ValueError("Duplicate institution IDs in primary matrix.")

    if labels[id_column].duplicated().any():
        raise ValueError("Duplicate institution IDs in final labels.")

    aligned = primary[required_primary_columns].merge(
        labels[[id_column, "cluster_id"]],
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) != len(primary) or len(aligned) != len(labels):
        raise ValueError(
            "Institution IDs do not align between primary matrix and final labels."
        )

    X = aligned[configured_features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "The primary CIPHER matrix contains missing or non-numeric values."
        )

    minimum = float(X.min().min())
    maximum = float(X.max().max())

    if minimum < 0 or maximum > 1:
        raise ValueError(
            f"Primary feature range must remain within [0,1]; found [{minimum}, {maximum}]."
        )

    forbidden_columns = []

    for column in primary.columns:
        normalized = column.strip().lower()

        is_forbidden = (
            normalized in FORBIDDEN_EXACT_COLUMNS
            or normalized.startswith(FORBIDDEN_PREFIXES)
            or normalized.endswith(FORBIDDEN_SUFFIXES)
        )

        if is_forbidden:
            forbidden_columns.append(column)

    if forbidden_columns:
        raise ValueError(
            "FRICTA-derived scores or labels were found in the primary matrix:\n- "
            + "\n- ".join(forbidden_columns)
        )

    cluster_sizes = aligned["cluster_id"].value_counts().sort_index().to_dict()

    if len(cluster_sizes) != 2:
        raise ValueError(
            f"CIPHER expects the frozen two-profile partition; found {cluster_sizes}."
        )

    selected_candidate = selected_model.get("candidate_id")

    if selected_candidate != cipher_config["reference_model"]:
        raise ValueError(
            "Selected model and CIPHER reference model do not match.\n"
            f"Selected: {selected_candidate}\n"
            f"CIPHER config: {cipher_config['reference_model']}"
        )

    return {
        "fricta_config": fricta_config,
        "cipher_config": cipher_config,
        "selected_model": selected_model,
        "id_column": id_column,
        "features": configured_features,
        "n_institutions": int(len(aligned)),
        "feature_count": int(len(configured_features)),
        "feature_minimum": minimum,
        "feature_maximum": maximum,
        "cluster_sizes": {
            str(int(key)): int(value) for key, value in cluster_sizes.items()
        },
    }


def build_actionability_manifest(features: list[str]) -> pd.DataFrame:
    rows = []

    for feature in features:
        if feature not in FEATURE_METADATA:
            raise KeyError(f"No actionability metadata exists for feature: {feature}")

        metadata = FEATURE_METADATA[feature]

        rows.append(
            {
                "feature": feature,
                "domain": metadata["domain"],
                "included_in_clustering": True,
                "included_in_diagnostic_counterfactual_search": True,
                "actionability_class": metadata["actionability_class"],
                "realistic_improvement_direction": metadata[
                    "realistic_improvement_direction"
                ],
                "diagnostic_counterfactual_direction": metadata[
                    "diagnostic_counterfactual_direction"
                ],
                "candidate_values": "OBSERVED_NORMALIZED_LEVELS_ONLY",
                "causal_intervention_claim_allowed": False,
                "rationale": metadata["rationale"],
                "review_status": "REVIEW_REQUIRED",
                "review_note": "",
            }
        )

    return pd.DataFrame(rows)


def write_novelty_claim_log() -> None:
    text = """# CIPHER Novelty Claim Log

## Established components

The following components exist in prior methodological literature and must not be
presented as inventions of CIPHER:

- consensus and ensemble clustering;
- row and feature perturbation;
- cluster stability analysis;
- institution-level or observation-level membership uncertainty;
- counterfactual explanations;
- robust counterfactual explanations;
- counterfactual explanations for clustering;
- frequent and closed itemset mining;
- bootstrap and permutation validation.

## Integration investigated by CIPHER

CIPHER investigates the combination of:

1. heterogeneous clustering uncertainty across observations, features,
   representation, and algorithm;
2. institution-level membership certainty;
3. counterfactual profile-transition explanations required to remain valid
   across an admissible clustering ensemble;
4. signed recurrent motif mining over robust counterfactual sets;
5. falsification against one-dimensional severity and governance explanations;
6. synthetic testing of false-profile and false-motif discovery.

## Permitted provisional wording

> Prior work has studied consensus clustering, robust counterfactual
> explanations, and clustering counterfactuals separately. CIPHER investigates
> their integration for uncertainty-aware institutional profiling.

## Prohibited wording before a systematic novelty review

- first method ever;
- first counterfactual clustering framework;
- revolutionary;
- universally valid institutional profiles;
- causal mechanism discovery;
- guaranteed intervention recommendations.

## Required empirical condition

The integrated contribution is retained only if it improves measurable behavior
over simpler baselines, particularly post-perturbation counterfactual validity,
boundary identification, and false-motif control.
"""

    NOVELTY_PATH.write_text(text, encoding="utf-8")


def initialize_stage0(validation: dict) -> None:
    if ACTIONABILITY_PATH.exists():
        manifest = pd.read_csv(ACTIONABILITY_PATH)

        print("\nActionability manifest already exists. " "It was not overwritten.")
    else:
        manifest = build_actionability_manifest(validation["features"])

        manifest.to_csv(
            ACTIONABILITY_PATH,
            index=False,
        )

    write_novelty_claim_log()

    report = {
        "status": "ACTIONABILITY_REVIEW_REQUIRED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_institutions": validation["n_institutions"],
        "feature_count": validation["feature_count"],
        "feature_range": [
            validation["feature_minimum"],
            validation["feature_maximum"],
        ],
        "cluster_sizes": validation["cluster_sizes"],
        "reference_model": validation["cipher_config"]["reference_model"],
        "manifest_path": str(ACTIONABILITY_PATH.relative_to(PROJECT_ROOT)),
        "required_next_action": (
            "Review every manifest row. Change review_status "
            "from REVIEW_REQUIRED to CONFIRMED only when the "
            "classification and rationale are accurate."
        ),
    }

    INITIALIZATION_REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pending = int((manifest["review_status"] != "CONFIRMED").sum())

    print("\n=== CIPHER STAGE 0 INITIALIZATION ===\n")
    print(f"Institutions validated: {validation['n_institutions']}")
    print(f"Primary features validated: {validation['feature_count']}")
    print(
        "Primary feature range: "
        f"[{validation['feature_minimum']}, "
        f"{validation['feature_maximum']}]"
    )
    print(f"Frozen cluster sizes: {validation['cluster_sizes']}")
    print("Reference model: " f"{validation['cipher_config']['reference_model']}")
    print(f"Manifest rows pending review: {pending}")

    print("\n=== ACTIONABILITY MANIFEST PREVIEW ===\n")
    print(
        manifest[
            [
                "feature",
                "domain",
                "actionability_class",
                "realistic_improvement_direction",
                "diagnostic_counterfactual_direction",
                "review_status",
            ]
        ].to_string(index=False)
    )

    print(
        "\nGATE STATUS: ACTIONABILITY REVIEW REQUIRED. "
        "Do not build the ensemble yet."
    )


def finalize_stage0(validation: dict) -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(
            "Stage 0 is already frozen. " f"Existing freeze file: {FREEZE_PATH}"
        )

    if not ACTIONABILITY_PATH.exists():
        raise FileNotFoundError(
            "Actionability manifest does not exist. " "Run --initialize first."
        )

    manifest = pd.read_csv(ACTIONABILITY_PATH)

    required_manifest_columns = [
        "feature",
        "domain",
        "included_in_clustering",
        "included_in_diagnostic_counterfactual_search",
        "actionability_class",
        "realistic_improvement_direction",
        "diagnostic_counterfactual_direction",
        "candidate_values",
        "causal_intervention_claim_allowed",
        "rationale",
        "review_status",
        "review_note",
    ]

    missing_manifest_columns = [
        column for column in required_manifest_columns if column not in manifest.columns
    ]

    if missing_manifest_columns:
        raise KeyError(
            "Actionability manifest is missing columns:\n- "
            + "\n- ".join(missing_manifest_columns)
        )

    if manifest["feature"].duplicated().any():
        raise ValueError("Duplicate features in actionability manifest.")

    if set(manifest["feature"]) != set(validation["features"]):
        raise ValueError(
            "Actionability manifest features do not match "
            "the frozen primary feature set."
        )

    unconfirmed = manifest.loc[
        manifest["review_status"] != "CONFIRMED",
        ["feature", "review_status"],
    ]

    if not unconfirmed.empty:
        raise ValueError(
            "Every feature must be confirmed before freezing:\n"
            + unconfirmed.to_string(index=False)
        )

    allowed_classes = {
        "DIRECTLY_MODIFIABLE",
        "INDIRECTLY_MODIFIABLE",
        "CONTEXTUALLY_MODIFIABLE",
        "EXCLUDED",
    }

    invalid_classes = sorted(set(manifest["actionability_class"]) - allowed_classes)

    if invalid_classes:
        raise ValueError(f"Invalid actionability classes: {invalid_classes}")

    source_paths = [
        FRICTA_CONFIG_PATH,
        CIPHER_CONFIG_PATH,
        PRIMARY_MATRIX_PATH,
        FINAL_LABELS_PATH,
        SELECTED_MODEL_PATH,
        ACTIONABILITY_PATH,
        NOVELTY_PATH,
    ]

    hashes = {
        str(path.relative_to(PROJECT_ROOT)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in source_paths
    }

    HASH_PATH.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "algorithm": "SHA-256",
                "files": hashes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    frozen_config = dict(validation["cipher_config"])
    frozen_config["design_status"] = "FROZEN_BEFORE_CIPHER_RESULTS"
    frozen_config["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()

    CONFIG_SNAPSHOT_PATH.write_text(
        json.dumps(
            frozen_config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    freeze = {
        "status": "CIPHER_STAGE_0_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_institutions": validation["n_institutions"],
        "feature_count": validation["feature_count"],
        "primary_features_in_order": validation["features"],
        "feature_range": [
            validation["feature_minimum"],
            validation["feature_maximum"],
        ],
        "reference_model": validation["cipher_config"]["reference_model"],
        "reference_k": validation["cipher_config"]["reference_k"],
        "reference_cluster_sizes": validation["cluster_sizes"],
        "governance_used_for_model_fitting": False,
        "fricta_scores_used_for_model_fitting": False,
        "fricta_archetypes_used_for_model_fitting": False,
        "actionability_manifest_confirmed": True,
        "hash_manifest": str(HASH_PATH.relative_to(PROJECT_ROOT)),
        "frozen_config_snapshot": str(CONFIG_SNAPSHOT_PATH.relative_to(PROJECT_ROOT)),
        "next_stage": ("Stage 1 — heterogeneous stability ensemble"),
    }

    FREEZE_PATH.write_text(
        json.dumps(
            freeze,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 0 FREEZE SUMMARY ===\n")
    print(f"Institutions frozen: {validation['n_institutions']}")
    print(f"Primary features frozen: {validation['feature_count']}")
    print(
        "Reference model frozen: " f"{validation['cipher_config']['reference_model']}"
    )
    print(f"Cluster sizes frozen: {validation['cluster_sizes']}")
    print(f"Hashed source files: {len(source_paths)}")
    print("Governance used for fitting: False")
    print("FRICTA scores/archetypes used for fitting: False")

    print("\nGATE STATUS: CIPHER STAGE 0 FROZEN. " "Stage 1 may now begin.")


def main() -> None:
    args = parse_args()
    ensure_directories()
    validation = validate_inputs()

    if args.finalize:
        finalize_stage0(validation)
    else:
        initialize_stage0(validation)


if __name__ == "__main__":
    main()
