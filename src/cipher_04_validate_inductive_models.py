from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FRICTA_CONFIG_PATH = ROOT / "icdm" / "design" / "experiment_config.json"
CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
PRIMARY_MATRIX_PATH = ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
MANIFEST_PATH = (
    ROOT / "cipher" / "outputs" / "ensemble" / "official" / "member_manifest.csv"
)
METRICS_PATH = (
    ROOT / "cipher" / "outputs" / "ensemble" / "official" / "member_metrics.csv"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "inductive_validation"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_centroid_labels(
    Z_all: np.ndarray,
    Z_sample: np.ndarray,
    fitted_labels: np.ndarray,
) -> np.ndarray:
    unique_labels = list(np.unique(fitted_labels))

    centroids = np.vstack(
        [Z_sample[fitted_labels == label].mean(axis=0) for label in unique_labels]
    )

    squared_distances = ((Z_all[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    nearest = np.argmin(squared_distances, axis=1)

    return np.array([unique_labels[idx] for idx in nearest])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)

    id_column = fricta_config["id_column"]
    frozen_features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    primary[id_column] = primary[id_column].astype(str)

    required_columns = [id_column] + frozen_features

    missing = [column for column in required_columns if column not in primary.columns]

    if missing:
        raise KeyError(f"Primary matrix missing columns: {missing}")

    if primary[id_column].duplicated().any():
        raise ValueError("Duplicate institution IDs in primary matrix.")

    if manifest["member_id"].duplicated().any():
        raise ValueError("Duplicate member IDs in ensemble manifest.")

    if metrics["member_id"].duplicated().any():
        raise ValueError("Duplicate member IDs in ensemble metrics.")

    data = primary.set_index(id_column)

    ward_manifest = manifest[manifest["algorithm"] == "ward"].copy()

    if len(ward_manifest) != 500:
        raise ValueError(f"Expected 500 Ward members; found {len(ward_manifest)}.")

    fidelity_threshold = float(cipher_config["inductive_extension"]["minimum_fidelity"])

    minimum_retained_ward_fraction = float(
        cipher_config["inductive_extension"]["minimum_retained_ward_fraction"]
    )

    minimum_counterfactual_ensemble_members = int(
        cipher_config["inductive_extension"]["minimum_counterfactual_ensemble_members"]
    )

    pca_threshold = float(cipher_config["ensemble"]["pca_variance_threshold"])

    validation_rows = []

    for _, row in ward_manifest.iterrows():
        member_id = str(row["member_id"])
        representation = str(row["representation"])

        sampled_ids = json.loads(row["sampled_institution_ids_json"])

        feature_names = json.loads(row["feature_names_json"])

        missing_ids = [
            institution_id
            for institution_id in sampled_ids
            if institution_id not in data.index
        ]

        if missing_ids:
            raise KeyError(
                f"{member_id}: sampled IDs missing from primary matrix: "
                f"{missing_ids[:5]}"
            )

        if not set(feature_names).issubset(set(frozen_features)):
            raise ValueError(f"{member_id}: manifest contains non-frozen features.")

        X_all = data.loc[
            :,
            feature_names,
        ].to_numpy(dtype=float)

        X_sample = data.loc[
            sampled_ids,
            feature_names,
        ].to_numpy(dtype=float)

        scaler = StandardScaler()
        Z_sample_raw = scaler.fit_transform(X_sample)
        Z_all_raw = scaler.transform(X_all)

        if representation == "PCA85":
            pca = PCA(
                n_components=pca_threshold,
                svd_solver="full",
                random_state=int(row["seed"]),
            )
            Z_sample = pca.fit_transform(Z_sample_raw)
            Z_all = pca.transform(Z_all_raw)
            pca_components = int(pca.n_components_)
            pca_variance = float(pca.explained_variance_ratio_.sum())
        elif representation == "RAW_STANDARDIZED":
            Z_sample = Z_sample_raw
            Z_all = Z_all_raw
            pca_components = np.nan
            pca_variance = np.nan
        else:
            raise ValueError(f"{member_id}: unknown representation {representation}")

        ward = AgglomerativeClustering(
            n_clusters=2,
            linkage="ward",
        )
        fitted_labels = ward.fit_predict(Z_sample)

        extension_all = nearest_centroid_labels(
            Z_all,
            Z_sample,
            fitted_labels,
        )

        sample_positions = [
            data.index.get_loc(institution_id) for institution_id in sampled_ids
        ]

        extension_sample = extension_all[sample_positions]

        fidelity = float(np.mean(extension_sample == fitted_labels))

        stored_match = metrics.loc[
            metrics["member_id"] == member_id,
            "ward_or_native_extension_fidelity_on_sample",
        ]

        if len(stored_match) != 1:
            raise ValueError(
                f"{member_id}: stored fidelity metric missing or duplicated."
            )

        stored_fidelity = float(stored_match.iloc[0])

        absolute_difference = abs(fidelity - stored_fidelity)

        validation_rows.append(
            {
                "member_id": member_id,
                "family": row["family"],
                "representation": representation,
                "sample_size": len(sampled_ids),
                "feature_count": len(feature_names),
                "pca_components": pca_components,
                "pca_explained_variance": pca_variance,
                "stored_extension_fidelity": stored_fidelity,
                "recomputed_extension_fidelity": fidelity,
                "absolute_difference": absolute_difference,
                "passes_fidelity_threshold": (fidelity >= fidelity_threshold),
            }
        )

    validation = pd.DataFrame(validation_rows)

    validation.to_csv(
        OUTPUT_DIR / "ward_extension_fidelity.csv",
        index=False,
    )

    maximum_recompute_difference = float(validation["absolute_difference"].max())

    if maximum_recompute_difference > 1e-12:
        raise ValueError(
            "Recomputed Ward fidelity does not reproduce "
            f"Stage 1 values. Max difference: {maximum_recompute_difference}"
        )

    retained_ward = validation[validation["passes_fidelity_threshold"]][
        "member_id"
    ].tolist()

    excluded_ward = validation[~validation["passes_fidelity_threshold"]][
        "member_id"
    ].tolist()

    counterfactual_manifest = manifest.merge(
        metrics[
            [
                "member_id",
                "ward_or_native_extension_fidelity_on_sample",
            ]
        ],
        on="member_id",
        how="left",
        validate="one_to_one",
    ).copy()

    counterfactual_manifest["eligible_for_counterfactual_ensemble"] = np.where(
        counterfactual_manifest["algorithm"].eq("kmeans"),
        True,
        counterfactual_manifest["member_id"].isin(retained_ward),
    )

    counterfactual_manifest["eligibility_reason"] = np.select(
        [
            counterfactual_manifest["algorithm"].eq("kmeans"),
            counterfactual_manifest["member_id"].isin(retained_ward),
        ],
        [
            "NATIVE_INDUCTIVE_PREDICTION",
            "WARD_NEAREST_CENTROID_FIDELITY_PASS",
        ],
        default=("WARD_NEAREST_CENTROID_FIDELITY_BELOW_THRESHOLD"),
    )

    counterfactual_manifest.to_csv(
        OUTPUT_DIR / "counterfactual_ensemble_manifest.csv",
        index=False,
    )

    eligible = counterfactual_manifest[
        counterfactual_manifest["eligible_for_counterfactual_ensemble"]
    ].copy()

    family_counts = eligible["family"].value_counts().sort_index().to_dict()

    ward_total = int((counterfactual_manifest["algorithm"] == "ward").sum())

    ward_retained = int(len(retained_ward))

    ward_retained_fraction = ward_retained / ward_total

    eligible_total = int(len(eligible))

    all_four_families_present = len(family_counts) == 4 and all(
        value > 0 for value in family_counts.values()
    )

    checks = {
        "ward_members_recomputed_500": (len(validation) == 500),
        "recomputed_fidelity_exactly_matches_stage1": (
            maximum_recompute_difference <= 1e-12
        ),
        "ward_retained_fraction_at_least_frozen_minimum": (
            ward_retained_fraction >= minimum_retained_ward_fraction
        ),
        "counterfactual_ensemble_at_least_frozen_minimum": (
            eligible_total >= minimum_counterfactual_ensemble_members
        ),
        "all_four_families_represented": (all_four_families_present),
        "all_kmeans_members_retained": (
            int((eligible["algorithm"] == "kmeans").sum()) == 500
        ),
    }

    report = {
        "fidelity_threshold": fidelity_threshold,
        "minimum_retained_ward_fraction": (minimum_retained_ward_fraction),
        "minimum_counterfactual_ensemble_members": (
            minimum_counterfactual_ensemble_members
        ),
        "ward_members_total": ward_total,
        "ward_members_retained": ward_retained,
        "ward_members_excluded": len(excluded_ward),
        "ward_retained_fraction": (ward_retained_fraction),
        "counterfactual_ensemble_members": (eligible_total),
        "family_counts": {str(key): int(value) for key, value in family_counts.items()},
        "maximum_fidelity_recompute_difference": (maximum_recompute_difference),
        "fidelity_distribution": {
            "minimum": float(validation["recomputed_extension_fidelity"].min()),
            "q025": float(validation["recomputed_extension_fidelity"].quantile(0.025)),
            "median": float(validation["recomputed_extension_fidelity"].median()),
            "q975": float(validation["recomputed_extension_fidelity"].quantile(0.975)),
            "maximum": float(validation["recomputed_extension_fidelity"].max()),
        },
        "checks": checks,
        "gate_status": ("PASS_STAGE_4" if all(checks.values()) else "FAIL_STAGE_4"),
    }

    (OUTPUT_DIR / "inductive_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    excluded_df = validation[~validation["passes_fidelity_threshold"]].sort_values(
        "recomputed_extension_fidelity"
    )

    excluded_df.to_csv(
        OUTPUT_DIR / "excluded_ward_members.csv",
        index=False,
    )

    print("\n=== CIPHER STAGE 4 — INDUCTIVE EXTENSION AUDIT ===\n")

    print(
        "Ward members recomputed:",
        len(validation),
    )
    print(
        "Maximum difference vs Stage 1 stored fidelity:",
        f"{maximum_recompute_difference:.12f}",
    )

    print("\nWard fidelity distribution:")
    for key, value in report["fidelity_distribution"].items():
        print(f"  {key}: {value:.4f}")

    print(
        "\nFidelity threshold:",
        fidelity_threshold,
    )
    print(
        "Ward retained:",
        ward_retained,
        "of",
        ward_total,
        f"({ward_retained_fraction:.4%})",
    )
    print(
        "Ward excluded:",
        len(excluded_ward),
    )

    print(
        "\nCounterfactual ensemble members:",
        eligible_total,
    )

    print(
        "Eligible family counts:",
        report["family_counts"],
    )

    print("\n=== EXCLUDED WARD MEMBERS ===\n")

    if len(excluded_df):
        print(
            excluded_df[
                [
                    "member_id",
                    "family",
                    "representation",
                    "recomputed_extension_fidelity",
                ]
            ].to_string(index=False)
        )
    else:
        print("None.")

    print("\n=== GATE CHECKS ===\n")

    for key, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {key}")

    print(f"\nGATE STATUS: {report['gate_status']}")

    if report["gate_status"] == "PASS_STAGE_4":
        print("Counterfactual ensemble frozen for Stage 5.")


if __name__ == "__main__":
    main()
