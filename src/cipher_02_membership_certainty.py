from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRICTA_CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
ENSEMBLE_DIR = PROJECT_ROOT / "cipher" / "outputs" / "ensemble" / "official"
CIPHER_CONFIG_PATH = (
    PROJECT_ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
)
OUTPUT_DIR = PROJECT_ROOT / "cipher" / "outputs" / "certainty"
AUDIT_DIR = PROJECT_ROOT / "cipher" / "outputs" / "audit"

EXPECTED_FAMILIES = [
    "R0_WARD",
    "R1_PCA85_WARD",
    "R0_KMEANS",
    "R1_PCA85_KMEANS",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_entropy_binary(p1: float, p2: float) -> float:
    probs = np.array([p1, p2], dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return np.nan
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(2.0))


def majority_probability(
    values: np.ndarray, profiles: list[int]
) -> tuple[int, float, dict[int, float]]:
    counts = {profile: int(np.sum(values == profile)) for profile in profiles}
    total = sum(counts.values())
    if total == 0:
        return profiles[0], np.nan, {profile: np.nan for profile in profiles}

    probs = {profile: counts[profile] / total for profile in profiles}
    dominant = max(profiles, key=lambda p: (probs[p], -p))
    return dominant, float(probs[dominant]), probs


def get_oob_member_columns(
    manifest: pd.DataFrame,
    pred_oob: pd.DataFrame,
) -> list[str]:
    member_ids = manifest["member_id"].astype(str).tolist()
    missing = [
        member_id for member_id in member_ids if member_id not in pred_oob.columns
    ]
    if missing:
        raise KeyError(f"Missing member prediction columns: {missing[:5]}")
    return member_ids


def compute_pair_consensus(
    institution_index: int,
    reference_labels: np.ndarray,
    coassignment_oob: np.ndarray,
) -> tuple[float, float]:
    own_profile = reference_labels[institution_index]
    same_mask = reference_labels == own_profile
    other_mask = reference_labels != own_profile

    same_mask[institution_index] = False

    same_values = coassignment_oob[institution_index, same_mask]
    other_values = coassignment_oob[institution_index, other_mask]

    within = (
        float(np.nanmean(same_values)) if np.isfinite(same_values).any() else np.nan
    )
    cross = (
        float(np.nanmean(other_values)) if np.isfinite(other_values).any() else np.nan
    )

    return within, cross


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)

    id_column = fricta_config["id_column"]

    manifest = pd.read_csv(ENSEMBLE_DIR / "member_manifest.csv")
    pred_oob = pd.read_csv(ENSEMBLE_DIR / "member_predictions_oob.csv")
    family_consensus = pd.read_csv(ENSEMBLE_DIR / "family_consensus.csv")

    coassign_oob_df = pd.read_csv(
        ENSEMBLE_DIR / "coassignment_oob_matrix.csv",
        index_col=0,
    )
    coassign_oob = coassign_oob_df.to_numpy(dtype=float)

    labels = pd.read_csv(FINAL_LABELS_PATH)[[id_column, "cluster_id"]].copy()
    labels[id_column] = labels[id_column].astype(str)

    pred_oob["institution_id"] = pred_oob["institution_id"].astype(str)

    merged = pred_oob[["institution_id"]].merge(
        labels.rename(columns={id_column: "institution_id"}),
        on="institution_id",
        how="left",
        validate="one_to_one",
    )

    if merged["cluster_id"].isna().any():
        raise ValueError(
            "Some ensemble institution IDs could not be matched to frozen labels."
        )

    institution_ids = merged["institution_id"].to_numpy()
    reference_labels = merged["cluster_id"].astype(int).to_numpy()
    profiles = sorted(np.unique(reference_labels).tolist())

    if profiles != [1, 2]:
        raise ValueError(f"Expected frozen profiles [1, 2], found {profiles}.")

    if list(coassign_oob_df.index.astype(str)) != institution_ids.tolist():
        raise ValueError(
            "OOB coassignment matrix row order does not match ensemble prediction order."
        )

    if list(coassign_oob_df.columns.astype(str)) != institution_ids.tolist():
        raise ValueError(
            "OOB coassignment matrix column order does not match ensemble prediction order."
        )

    member_columns = get_oob_member_columns(manifest, pred_oob)

    family_members = {
        family: manifest.loc[
            manifest["family"] == family,
            "member_id",
        ]
        .astype(str)
        .tolist()
        for family in EXPECTED_FAMILIES
    }

    missing_families = [
        family for family, members in family_members.items() if len(members) == 0
    ]
    if missing_families:
        raise ValueError(f"Missing ensemble families: {missing_families}")

    core_p = float(cipher_config["certainty"]["core_probability_threshold"])
    core_family = float(cipher_config["certainty"]["core_family_consistency_threshold"])
    boundary_p = float(cipher_config["certainty"]["boundary_probability_threshold"])
    boundary_family = float(
        cipher_config["certainty"]["boundary_family_consistency_threshold"]
    )

    rows = []
    family_rows = []

    for i, institution_id in enumerate(institution_ids):
        all_values = (
            pd.to_numeric(
                pred_oob.loc[i, member_columns],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .to_numpy()
        )

        dominant_profile, dominant_probability, probs = majority_probability(
            all_values,
            profiles,
        )

        p1 = probs[1]
        p2 = probs[2]
        entropy = normalized_entropy_binary(p1, p2)
        margin = float(2.0 * abs(p1 - 0.5))

        within_consensus, cross_leakage = compute_pair_consensus(
            i,
            reference_labels,
            coassign_oob,
        )

        family_dominant_probs = []
        family_reference_probs = []
        family_disagreement_flags = []

        for family in EXPECTED_FAMILIES:
            values = (
                pd.to_numeric(
                    pred_oob.loc[i, family_members[family]],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .to_numpy()
            )

            fam_dom, fam_dom_p, fam_probs = majority_probability(
                values,
                profiles,
            )

            reference_profile = int(reference_labels[i])
            reference_probability = float(fam_probs[reference_profile])

            family_rows.append(
                {
                    "institution_id": institution_id,
                    "reference_profile": reference_profile,
                    "family": family,
                    "n_oob_predictions": int(len(values)),
                    "family_dominant_profile": int(fam_dom),
                    "family_dominant_probability": fam_dom_p,
                    "family_reference_profile_probability": reference_probability,
                    "profile_1_probability": float(fam_probs[1]),
                    "profile_2_probability": float(fam_probs[2]),
                }
            )

            family_dominant_probs.append(fam_dom_p)
            family_reference_probs.append(reference_probability)
            family_disagreement_flags.append(int(fam_dom != dominant_profile))

        family_consistency = float(min(family_reference_probs))
        minimum_family_dominant_probability = float(min(family_dominant_probs))
        family_disagreement_count = int(sum(family_disagreement_flags))

        reference_profile = int(reference_labels[i])
        reference_probability = float(probs[reference_profile])
        ensemble_agrees_with_reference = int(dominant_profile == reference_profile)

        if reference_probability >= core_p and family_consistency >= core_family:
            certainty_class = "CORE"
        elif reference_probability < boundary_p or family_consistency < boundary_family:
            certainty_class = "BOUNDARY"
        else:
            certainty_class = "HALO"

        rows.append(
            {
                "institution_id": institution_id,
                "reference_profile": reference_profile,
                "n_oob_predictions": int(len(all_values)),
                "ensemble_dominant_profile": int(dominant_profile),
                "ensemble_dominant_probability": dominant_probability,
                "reference_profile_probability": reference_probability,
                "profile_1_probability": float(p1),
                "profile_2_probability": float(p2),
                "normalized_entropy": entropy,
                "membership_margin": margin,
                "within_reference_profile_consensus": within_consensus,
                "cross_profile_leakage": cross_leakage,
                "consensus_gap": (
                    float(within_consensus - cross_leakage)
                    if np.isfinite(within_consensus) and np.isfinite(cross_leakage)
                    else np.nan
                ),
                "family_consistency": family_consistency,
                "minimum_family_dominant_probability": minimum_family_dominant_probability,
                "family_disagreement_count": family_disagreement_count,
                "ensemble_agrees_with_reference": ensemble_agrees_with_reference,
                "certainty_class": certainty_class,
            }
        )

    certainty = pd.DataFrame(rows)
    certainty_by_family = pd.DataFrame(family_rows)

    certainty.to_csv(
        OUTPUT_DIR / "institution_certainty.csv",
        index=False,
    )
    certainty_by_family.to_csv(
        OUTPUT_DIR / "certainty_by_family.csv",
        index=False,
    )

    class_table = (
        certainty.groupby(
            ["reference_profile", "certainty_class"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["CORE", "HALO", "BOUNDARY"], fill_value=0)
        .reset_index()
    )
    class_table.to_csv(
        OUTPUT_DIR / "core_halo_boundary_counts.csv",
        index=False,
    )

    profile_summary = (
        certainty.groupby("reference_profile", observed=True)
        .agg(
            n=("institution_id", "count"),
            median_reference_probability=("reference_profile_probability", "median"),
            minimum_reference_probability=("reference_profile_probability", "min"),
            median_family_consistency=("family_consistency", "median"),
            minimum_family_consistency=("family_consistency", "min"),
            median_entropy=("normalized_entropy", "median"),
            median_consensus_gap=("consensus_gap", "median"),
            core_count=("certainty_class", lambda s: int((s == "CORE").sum())),
            halo_count=("certainty_class", lambda s: int((s == "HALO").sum())),
            boundary_count=("certainty_class", lambda s: int((s == "BOUNDARY").sum())),
            ensemble_reference_disagreements=(
                "ensemble_agrees_with_reference",
                lambda s: int((s == 0).sum()),
            ),
        )
        .reset_index()
    )
    profile_summary.to_csv(
        OUTPUT_DIR / "certainty_profile_summary.csv",
        index=False,
    )

    lowest_certainty = certainty.sort_values(
        [
            "reference_profile_probability",
            "family_consistency",
            "membership_margin",
        ],
        ascending=[True, True, True],
    ).head(15)

    lowest_certainty.to_csv(
        OUTPUT_DIR / "lowest_certainty_institutions.csv",
        index=False,
    )

    class_counts = certainty["certainty_class"].value_counts().to_dict()

    family_profile_summary = (
        certainty_by_family.groupby(
            ["reference_profile", "family"],
            observed=True,
        )
        .agg(
            institutions=("institution_id", "count"),
            median_reference_probability=(
                "family_reference_profile_probability",
                "median",
            ),
            minimum_reference_probability=(
                "family_reference_profile_probability",
                "min",
            ),
            median_oob_predictions=("n_oob_predictions", "median"),
        )
        .reset_index()
    )
    family_profile_summary.to_csv(
        OUTPUT_DIR / "family_profile_summary.csv",
        index=False,
    )

    gate_checks = {
        "81_institutions": len(certainty) == 81,
        "all_have_at_least_150_oob_predictions": bool(
            (certainty["n_oob_predictions"] >= 150).all()
        ),
        "all_probabilities_in_0_1": bool(
            (
                certainty[
                    [
                        "ensemble_dominant_probability",
                        "reference_profile_probability",
                        "profile_1_probability",
                        "profile_2_probability",
                        "family_consistency",
                        "minimum_family_dominant_probability",
                    ]
                ]
                .apply(pd.to_numeric)
                .ge(0)
                .all()
                .all()
            )
            and (
                certainty[
                    [
                        "ensemble_dominant_probability",
                        "reference_profile_probability",
                        "profile_1_probability",
                        "profile_2_probability",
                        "family_consistency",
                        "minimum_family_dominant_probability",
                    ]
                ]
                .apply(pd.to_numeric)
                .le(1)
                .all()
                .all()
            )
        ),
        "profile_probabilities_sum_to_1": bool(
            np.allclose(
                certainty["profile_1_probability"] + certainty["profile_2_probability"],
                1.0,
            )
        ),
        "certainty_class_complete": bool(
            certainty["certainty_class"].isin(["CORE", "HALO", "BOUNDARY"]).all()
        ),
        "both_profiles_present": set(certainty["reference_profile"]) == {1, 2},
        "family_rows_complete": len(certainty_by_family) == 81 * 4,
    }

    report = {
        "gate_checks": gate_checks,
        "gate_status": (
            "PASS_STAGE_2_COMPUTATION"
            if all(gate_checks.values())
            else "FAIL_STAGE_2_COMPUTATION"
        ),
        "thresholds": {
            "core_reference_probability": core_p,
            "core_family_consistency": core_family,
            "boundary_reference_probability": boundary_p,
            "boundary_family_consistency": boundary_family,
        },
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "ensemble_reference_disagreements": int(
            (certainty["ensemble_agrees_with_reference"] == 0).sum()
        ),
        "minimum_reference_profile_probability": float(
            certainty["reference_profile_probability"].min()
        ),
        "median_reference_profile_probability": float(
            certainty["reference_profile_probability"].median()
        ),
        "minimum_family_consistency": float(certainty["family_consistency"].min()),
        "median_family_consistency": float(certainty["family_consistency"].median()),
        "minimum_consensus_gap": float(certainty["consensus_gap"].min()),
        "median_consensus_gap": float(certainty["consensus_gap"].median()),
    }

    (OUTPUT_DIR / "certainty_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 2 MEMBERSHIP CERTAINTY ===\n")
    print("Institutions:", len(certainty))
    print(
        "Certainty classes:",
        {key: int(value) for key, value in class_counts.items()},
    )
    print(
        "Ensemble/reference disagreements:",
        report["ensemble_reference_disagreements"],
    )
    print(
        "Reference-profile probability:",
        f"min={report['minimum_reference_profile_probability']:.4f},",
        f"median={report['median_reference_profile_probability']:.4f}",
    )
    print(
        "Family consistency:",
        f"min={report['minimum_family_consistency']:.4f},",
        f"median={report['median_family_consistency']:.4f}",
    )
    print(
        "Consensus gap:",
        f"min={report['minimum_consensus_gap']:.4f},",
        f"median={report['median_consensus_gap']:.4f}",
    )

    print("\n=== PROFILE SUMMARY ===\n")
    print(profile_summary.to_string(index=False))

    print("\n=== FAMILY PROFILE SUMMARY ===\n")
    print(family_profile_summary.to_string(index=False))

    print("\n=== 15 LOWEST-CERTAINTY INSTITUTIONS ===\n")
    print(
        lowest_certainty[
            [
                "institution_id",
                "reference_profile",
                "reference_profile_probability",
                "family_consistency",
                "normalized_entropy",
                "membership_margin",
                "consensus_gap",
                "family_disagreement_count",
                "certainty_class",
            ]
        ].to_string(index=False)
    )

    print("\n=== COMPUTATION CHECKS ===\n")
    for name, passed in gate_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {report['gate_status']}")
    print(
        "Stage 2 results require interpretive review before "
        "core/halo/boundary claims are accepted."
    )


if __name__ == "__main__":
    main()
