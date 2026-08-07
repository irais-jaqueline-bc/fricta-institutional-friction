from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "cipher" / "design" / "actionability_manifest.csv"
BACKUP_PATH = (
    PROJECT_ROOT / "cipher" / "design" / "actionability_manifest_before_review.csv"
)

EXPECTED_FEATURES = {
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
}


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run cipher_00_freeze.py --initialize first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    if set(manifest["feature"]) != EXPECTED_FEATURES:
        raise ValueError(
            "Manifest features do not match the 13 frozen CIPHER indicators."
        )

    if len(manifest) != 13:
        raise ValueError(f"Expected 13 rows; found {len(manifest)}.")

    if not BACKUP_PATH.exists():
        shutil.copy2(MANIFEST_PATH, BACKUP_PATH)

    # Conservative adjustment:
    # Internet stability may depend on external infrastructure, providers,
    # geography, and budget, so it is contextually rather than directly modifiable.
    internet_mask = manifest["feature"] == "internet_stability_constraint"

    manifest.loc[
        internet_mask,
        "actionability_class",
    ] = "CONTEXTUALLY_MODIFIABLE"

    manifest.loc[
        internet_mask,
        "rationale",
    ] = (
        "Can change through provider, redundancy, or infrastructure upgrades, "
        "but feasibility may depend on geography, market access, funding, and "
        "external connectivity conditions."
    )

    # The remaining classifications are accepted after methodological review.
    manifest["review_status"] = "CONFIRMED"
    manifest["review_note"] = (
        "Confirmed during CIPHER Stage 0 methodological review; "
        "counterfactuals remain diagnostic and non-causal."
    )

    manifest.to_csv(MANIFEST_PATH, index=False)

    print("\n=== CIPHER ACTIONABILITY REVIEW ===\n")
    print(
        manifest[
            [
                "feature",
                "actionability_class",
                "realistic_improvement_direction",
                "diagnostic_counterfactual_direction",
                "review_status",
            ]
        ].to_string(index=False)
    )

    print("\nRows confirmed:", int((manifest["review_status"] == "CONFIRMED").sum()))
    print("Backup:", BACKUP_PATH)
    print("Updated manifest:", MANIFEST_PATH)
    print("\nGATE STATUS: MANIFEST CONFIRMED. Stage 0 may be finalized.")


if __name__ == "__main__":
    main()
