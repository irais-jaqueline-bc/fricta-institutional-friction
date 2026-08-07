from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

STAGE6_FREEZE_PATH = (
    ROOT / "cipher" / "design" / "stage6_ensemble_robustness_freeze.json"
)
CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
FRICTA_CONFIG_PATH = ROOT / "icdm" / "design" / "experiment_config.json"

PRIMARY_MATRIX_PATH = ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
FINAL_LABELS_PATH = (
    ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)

OFFICIAL_MANIFEST_PATH = (
    ROOT / "cipher" / "outputs" / "ensemble" / "official" / "member_manifest.csv"
)
CF_MANIFEST_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "inductive_validation"
    / "counterfactual_ensemble_manifest.csv"
)
PREDICTIONS_ALL_PATH = (
    ROOT / "cipher" / "outputs" / "ensemble" / "official" / "member_predictions_all.csv"
)
WARD_FIDELITY_PATH = (
    ROOT / "cipher" / "outputs" / "inductive_validation" / "ward_extension_fidelity.csv"
)
EXCLUDED_WARD_PATH = (
    ROOT / "cipher" / "outputs" / "inductive_validation" / "excluded_ward_members.csv"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "ensemble" / "stage6_reconstruction_audit"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"

EXPECTED_ELIGIBLE_FAMILY_COUNTS = {
    "R0_KMEANS": 250,
    "R0_WARD": 244,
    "R1_PCA85_KMEANS": 250,
    "R1_PCA85_WARD": 240,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_list(value: str) -> list[Any]:
    obj = json.loads(value)
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list, found {type(obj).__name__}")
    return obj


def parse_label_mapping(value: str) -> dict[int, int]:
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise ValueError("label_mapping_json must decode to an object.")
    return {int(k): int(v) for k, v in obj.items()}


def apply_mapping(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    try:
        return np.array([mapping[int(label)] for label in labels], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"Raw cluster label {exc.args[0]} missing from frozen label mapping."
        ) from exc


def family_from_row(row: pd.Series) -> str:
    return str(row["family"])


def fit_member_and_predict(
    row: pd.Series,
    data_by_id: pd.DataFrame,
    all_ids: list[str],
    kmeans_n_init: int,
    pca_threshold: float,
) -> dict[str, Any]:
    member_id = str(row["member_id"])
    algorithm = str(row["algorithm"]).upper()
    representation = str(row["representation"]).upper()
    seed = int(row["seed"])

    feature_names = [
        str(item) for item in parse_json_list(str(row["feature_names_json"]))
    ]
    sampled_ids = [
        str(item) for item in parse_json_list(str(row["sampled_institution_ids_json"]))
    ]

    if len(feature_names) != int(row["feature_count"]):
        raise ValueError(
            f"{member_id}: feature_count does not match feature_names_json."
        )
    if len(sampled_ids) != int(row["sample_size"]):
        raise ValueError(f"{member_id}: sample_size does not match sampled IDs.")

    X_sample = data_by_id.loc[
        sampled_ids,
        feature_names,
    ].to_numpy(dtype=float)

    X_all = data_by_id.loc[
        all_ids,
        feature_names,
    ].to_numpy(dtype=float)

    scaler = StandardScaler()
    Z_sample_scaled = scaler.fit_transform(X_sample)
    Z_all_scaled = scaler.transform(X_all)

    pca_components = 0
    pca_explained_variance = 1.0
    pca = None

    if (
        representation
        in {
            "R1",
            "R1_PCA85",
            "PCA85",
        }
        or "PCA" in representation
    ):
        pca = PCA(
            n_components=pca_threshold,
            svd_solver="full",
            random_state=seed,
        )
        Z_sample = pca.fit_transform(Z_sample_scaled)
        Z_all = pca.transform(Z_all_scaled)
        pca_components = int(pca.n_components_)
        pca_explained_variance = float(np.sum(pca.explained_variance_ratio_))
    else:
        Z_sample = Z_sample_scaled
        Z_all = Z_all_scaled

    mapping = parse_label_mapping(str(row["label_mapping_json"]))

    if algorithm == "KMEANS":
        model = KMeans(
            n_clusters=2,
            n_init=kmeans_n_init,
            random_state=seed,
        )
        raw_sample_labels = model.fit_predict(Z_sample)
        raw_all_predictions = model.predict(Z_all)

        aligned_sample_labels = apply_mapping(
            raw_sample_labels,
            mapping,
        )
        aligned_all_predictions = apply_mapping(
            raw_all_predictions,
            mapping,
        )

        extension_fidelity = 1.0

    elif algorithm in {
        "WARD",
        "HAC_WARD",
        "AGGLOMERATIVE_WARD",
    }:
        model = AgglomerativeClustering(
            n_clusters=2,
            linkage="ward",
        )
        raw_sample_labels = model.fit_predict(Z_sample)

        raw_values = sorted(np.unique(raw_sample_labels).tolist())

        raw_centroids = {
            int(raw_label): Z_sample[raw_sample_labels == raw_label].mean(axis=0)
            for raw_label in raw_values
        }

        centroid_matrix = np.vstack(
            [raw_centroids[int(raw_label)] for raw_label in raw_values]
        )

        distances_all = np.sqrt(
            (
                (
                    Z_all[:, None, :]
                    - centroid_matrix[
                        None,
                        :,
                        :,
                    ]
                )
                ** 2
            ).sum(axis=2)
        )

        nearest_all = np.argmin(
            distances_all,
            axis=1,
        )
        raw_all_predictions = np.array(
            [raw_values[index] for index in nearest_all],
            dtype=int,
        )

        distances_sample = np.sqrt(
            (
                (
                    Z_sample[:, None, :]
                    - centroid_matrix[
                        None,
                        :,
                        :,
                    ]
                )
                ** 2
            ).sum(axis=2)
        )
        nearest_sample = np.argmin(
            distances_sample,
            axis=1,
        )
        raw_sample_extension = np.array(
            [raw_values[index] for index in nearest_sample],
            dtype=int,
        )

        aligned_sample_labels = apply_mapping(
            raw_sample_labels,
            mapping,
        )
        aligned_sample_extension = apply_mapping(
            raw_sample_extension,
            mapping,
        )
        aligned_all_predictions = apply_mapping(
            raw_all_predictions,
            mapping,
        )

        extension_fidelity = float(
            np.mean(aligned_sample_extension == aligned_sample_labels)
        )

    else:
        raise ValueError(f"{member_id}: unsupported algorithm {algorithm}")

    return {
        "member_id": member_id,
        "family": family_from_row(row),
        "algorithm": algorithm,
        "representation": representation,
        "feature_count": len(feature_names),
        "sample_size": len(sampled_ids),
        "pca_components_recomputed": pca_components,
        "pca_explained_variance_recomputed": pca_explained_variance,
        "extension_fidelity_recomputed": extension_fidelity,
        "all_predictions": aligned_all_predictions,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage6_freeze = load_json(STAGE6_FREEZE_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    fricta_config = load_json(FRICTA_CONFIG_PATH)

    if stage6_freeze.get("gate_status") != "PASS_STAGE_6A_DESIGN_FREEZE":
        raise ValueError("Stage 6A design freeze has not passed.")

    id_column = fricta_config["id_column"]

    official_manifest = pd.read_csv(OFFICIAL_MANIFEST_PATH)
    cf_manifest = pd.read_csv(CF_MANIFEST_PATH)
    stored_predictions = pd.read_csv(PREDICTIONS_ALL_PATH)
    ward_fidelity = pd.read_csv(WARD_FIDELITY_PATH)
    excluded_ward = pd.read_csv(EXCLUDED_WARD_PATH)

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[
        [
            id_column,
            "cluster_id",
        ]
    ]

    primary[id_column] = primary[id_column].astype(str)
    labels[id_column] = labels[id_column].astype(str)
    stored_predictions["institution_id"] = stored_predictions["institution_id"].astype(
        str
    )

    data = primary.merge(
        labels,
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 institutions; found {len(data)}.")

    data_by_id = data.set_index(id_column)
    all_ids = data[id_column].astype(str).tolist()

    official_manifest["member_id"] = official_manifest["member_id"].astype(str)
    cf_manifest["member_id"] = cf_manifest["member_id"].astype(str)

    # -----------------------------
    # Static artifact checks
    # -----------------------------
    shared_columns = [
        "member_id",
        "family",
        "algorithm",
        "representation",
        "accepted_index_within_family",
        "attempt_within_family",
        "seed",
        "sample_size",
        "oob_size",
        "feature_count",
        "feature_names_json",
        "sampled_institution_ids_json",
        "oob_institution_ids_json",
        "pca_components",
        "pca_explained_variance",
        "label_mapping_json",
        "alignment_contingency_json",
    ]

    official_sorted = (
        official_manifest[shared_columns]
        .sort_values("member_id")
        .reset_index(drop=True)
    )

    cf_sorted = (
        cf_manifest[shared_columns].sort_values("member_id").reset_index(drop=True)
    )

    manifests_match = len(official_sorted) == len(cf_sorted) and official_sorted.equals(
        cf_sorted
    )

    eligibility_series = (
        cf_manifest["eligible_for_counterfactual_ensemble"]
        .map(
            lambda value: (
                value
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() in {"true", "1", "yes"}
            )
        )
        .astype(bool)
    )

    eligible = cf_manifest[eligibility_series].copy()

    eligible_family_counts = eligible["family"].value_counts().to_dict()

    excluded = cf_manifest[~eligibility_series].copy()

    excluded_all_ward = bool(
        excluded["algorithm"].astype(str).str.upper().str.contains("WARD").all()
    )

    stored_member_columns = [
        column for column in stored_predictions.columns if column != "institution_id"
    ]

    static_checks = {
        "official_manifest_has_1000_members": (len(official_manifest) == 1000),
        "counterfactual_manifest_has_1000_members": (len(cf_manifest) == 1000),
        "manifest_member_definitions_match_exactly": (manifests_match),
        "exactly_984_counterfactual_eligible_members": (len(eligible) == 984),
        "exactly_16_excluded_members": (len(excluded) == 16),
        "all_16_excluded_members_are_ward": (excluded_all_ward),
        "eligible_family_counts_match_stage4": (
            all(
                int(
                    eligible_family_counts.get(
                        family,
                        0,
                    )
                )
                == expected
                for family, expected in EXPECTED_ELIGIBLE_FAMILY_COUNTS.items()
            )
        ),
        "stored_prediction_matrix_has_81_institutions": (len(stored_predictions) == 81),
        "stored_prediction_matrix_has_1000_members": (
            len(stored_member_columns) == 1000
        ),
        "stored_prediction_member_ids_match_manifest": (
            set(stored_member_columns)
            == set(official_manifest["member_id"].astype(str))
        ),
        "ward_fidelity_file_has_500_members": (len(ward_fidelity) == 500),
        "excluded_ward_file_has_16_members": (len(excluded_ward) == 16),
    }

    print("\n=== CIPHER STAGE 6B — ENSEMBLE RECONSTRUCTION AUDIT ===\n")

    print("Static artifact checks:")
    for name, passed in static_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(static_checks.values()):
        print("\nGATE STATUS: FAIL_STAGE_6B_STATIC_ARTIFACT_AUDIT")
        raise SystemExit(1)

    # Config defaults are only fallbacks; current frozen design is known to use these.
    ensemble_cfg = cipher_config.get(
        "ensemble",
        {},
    )

    kmeans_n_init = int(
        ensemble_cfg.get(
            "kmeans_n_init",
            25,
        )
    )
    pca_threshold = float(
        ensemble_cfg.get(
            "pca_variance_threshold",
            0.85,
        )
    )

    stored_by_id = stored_predictions.set_index("institution_id").loc[all_ids]

    ward_fidelity_lookup = (
        ward_fidelity.set_index("member_id")["recomputed_extension_fidelity"]
        .astype(float)
        .to_dict()
    )

    eligibility_lookup = dict(
        zip(
            cf_manifest["member_id"].astype(str),
            eligibility_series.astype(bool),
        )
    )

    audit_rows = []

    total_prediction_mismatches = 0
    exact_prediction_members = 0
    pca_component_mismatches = 0
    pca_variance_mismatches = 0
    fidelity_mismatches = 0

    started = time.perf_counter()

    for index, row in official_manifest.iterrows():
        reconstructed = fit_member_and_predict(
            row=row,
            data_by_id=data_by_id,
            all_ids=all_ids,
            kmeans_n_init=kmeans_n_init,
            pca_threshold=pca_threshold,
        )

        member_id = reconstructed["member_id"]

        stored = stored_by_id[member_id].to_numpy()

        stored_numeric = pd.to_numeric(
            pd.Series(stored),
            errors="coerce",
        ).to_numpy()

        if np.isnan(stored_numeric).any():
            raise ValueError(
                f"{member_id}: stored member_predictions_all contains NaN/non-numeric values."
            )

        stored_numeric = stored_numeric.astype(int)
        recomputed_predictions = reconstructed["all_predictions"].astype(int)

        mismatch_count = int(np.sum(stored_numeric != recomputed_predictions))

        prediction_exact = mismatch_count == 0

        total_prediction_mismatches += mismatch_count
        exact_prediction_members += int(prediction_exact)

        stored_pca_components_raw = row["pca_components"]
        is_pca_member = "PCA" in str(row["representation"]).upper()

        if pd.isna(stored_pca_components_raw):
            stored_pca_components = 0
            pca_components_match = (
                not is_pca_member and reconstructed["pca_components_recomputed"] == 0
            )
        else:
            stored_pca_components = int(stored_pca_components_raw)
            pca_components_match = (
                stored_pca_components == reconstructed["pca_components_recomputed"]
            )

        if not pca_components_match:
            pca_component_mismatches += 1

        stored_pca_variance_raw = row["pca_explained_variance"]

        if pd.isna(stored_pca_variance_raw):
            stored_pca_variance = np.nan
            variance_abs_diff = 0.0 if not is_pca_member else np.inf
            pca_variance_match = not is_pca_member
        else:
            stored_pca_variance = float(stored_pca_variance_raw)
            variance_abs_diff = abs(
                stored_pca_variance - reconstructed["pca_explained_variance_recomputed"]
            )
            pca_variance_match = variance_abs_diff <= 1e-10

        if not pca_variance_match:
            pca_variance_mismatches += 1

        if "WARD" in str(row["algorithm"]).upper():
            stored_stage4_fidelity = float(ward_fidelity_lookup[member_id])
        else:
            stored_stage4_fidelity = 1.0

        fidelity_abs_diff = abs(
            stored_stage4_fidelity - reconstructed["extension_fidelity_recomputed"]
        )

        fidelity_match = fidelity_abs_diff <= 1e-12

        if not fidelity_match:
            fidelity_mismatches += 1

        audit_rows.append(
            {
                "member_id": member_id,
                "family": reconstructed["family"],
                "algorithm": reconstructed["algorithm"],
                "representation": reconstructed["representation"],
                "eligible_for_counterfactual_ensemble": bool(
                    eligibility_lookup[member_id]
                ),
                "prediction_mismatch_count_81": (mismatch_count),
                "prediction_exact_match": (prediction_exact),
                "stored_pca_components": (stored_pca_components),
                "recomputed_pca_components": (
                    reconstructed["pca_components_recomputed"]
                ),
                "pca_components_match": (pca_components_match),
                "stored_pca_explained_variance": (stored_pca_variance),
                "recomputed_pca_explained_variance": (
                    reconstructed["pca_explained_variance_recomputed"]
                ),
                "pca_variance_absolute_difference": (variance_abs_diff),
                "pca_variance_match": (pca_variance_match),
                "stored_stage4_extension_fidelity": (stored_stage4_fidelity),
                "recomputed_extension_fidelity": (
                    reconstructed["extension_fidelity_recomputed"]
                ),
                "fidelity_absolute_difference": (fidelity_abs_diff),
                "fidelity_match": (fidelity_match),
            }
        )

        if (index + 1) % 100 == 0:
            print(
                f"Reconstructed {index + 1:04d}/1000 members...",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    audit_df = pd.DataFrame(audit_rows)

    audit_df.to_csv(
        OUTPUT_DIR / "member_reconstruction_audit.csv",
        index=False,
    )

    eligible_audit = audit_df[
        audit_df["eligible_for_counterfactual_ensemble"].astype(bool)
    ].copy()

    family_summary = (
        eligible_audit.groupby(
            "family",
            sort=True,
        )
        .agg(
            eligible_members=(
                "member_id",
                "size",
            ),
            exact_prediction_members=(
                "prediction_exact_match",
                "sum",
            ),
            total_prediction_mismatches=(
                "prediction_mismatch_count_81",
                "sum",
            ),
            pca_component_mismatches=(
                "pca_components_match",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            pca_variance_mismatches=(
                "pca_variance_match",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            fidelity_mismatches=(
                "fidelity_match",
                lambda values: int((~values.astype(bool)).sum()),
            ),
        )
        .reset_index()
    )

    family_summary.to_csv(
        OUTPUT_DIR / "eligible_family_reconstruction_summary.csv",
        index=False,
    )

    eligible_member_ids = set(eligible["member_id"].astype(str))
    excluded_member_ids = set(excluded["member_id"].astype(str))
    excluded_file_ids = set(excluded_ward["member_id"].astype(str))

    dynamic_checks = {
        "all_1000_members_reconstructed": (len(audit_df) == 1000),
        "all_1000_member_predictions_match_stored_81x1000_matrix": (
            exact_prediction_members == 1000 and total_prediction_mismatches == 0
        ),
        "all_pca_component_counts_reproduce": (pca_component_mismatches == 0),
        "all_pca_explained_variance_values_reproduce": (pca_variance_mismatches == 0),
        "all_extension_fidelity_values_reproduce": (fidelity_mismatches == 0),
        "eligible_member_set_is_exactly_984": (len(eligible_member_ids) == 984),
        "excluded_member_set_matches_stage4_excluded_file": (
            excluded_member_ids == excluded_file_ids
        ),
        "all_984_eligible_members_have_exact_prediction_reconstruction": bool(
            eligible_audit["prediction_exact_match"].astype(bool).all()
        ),
        "all_984_eligible_members_have_exact_fidelity_reconstruction": bool(
            eligible_audit["fidelity_match"].astype(bool).all()
        ),
    }

    report = {
        "kmeans_n_init": kmeans_n_init,
        "pca_variance_threshold": pca_threshold,
        "members_reconstructed": int(len(audit_df)),
        "eligible_members": int(len(eligible_audit)),
        "exact_prediction_members": int(exact_prediction_members),
        "total_prediction_mismatches": int(total_prediction_mismatches),
        "pca_component_mismatches": int(pca_component_mismatches),
        "pca_variance_mismatches": int(pca_variance_mismatches),
        "fidelity_mismatches": int(fidelity_mismatches),
        "elapsed_seconds": float(elapsed),
        "static_checks": static_checks,
        "dynamic_checks": dynamic_checks,
        "gate_status": (
            "PASS_STAGE_6B_RECONSTRUCTION_AUDIT"
            if all(static_checks.values()) and all(dynamic_checks.values())
            else "FAIL_STAGE_6B_RECONSTRUCTION_AUDIT"
        ),
    }

    (OUTPUT_DIR / "stage6b_reconstruction_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (AUDIT_DIR / "stage6b_reconstruction_audit.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== ELIGIBLE FAMILY RECONSTRUCTION SUMMARY ===\n")
    print(family_summary.to_string(index=False))

    print("\n=== DYNAMIC RECONSTRUCTION CHECKS ===\n")
    for name, passed in dynamic_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(
        "\nPrediction mismatch cells:",
        total_prediction_mismatches,
    )
    print(
        "PCA component mismatches:",
        pca_component_mismatches,
    )
    print(
        "PCA variance mismatches:",
        pca_variance_mismatches,
    )
    print(
        "Extension fidelity mismatches:",
        fidelity_mismatches,
    )
    print(
        "Elapsed seconds:",
        f"{elapsed:.2f}",
    )

    print(f"\nGATE STATUS: {report['gate_status']}")

    if report["gate_status"] == "PASS_STAGE_6B_RECONSTRUCTION_AUDIT":
        print(
            "The 984-member counterfactual ensemble prediction engine is "
            "reconstructed and audited. Stage 6C may begin after review."
        )
    else:
        print(
            "Do not evaluate counterfactuals. Review reconstruction mismatches first."
        )


if __name__ == "__main__":
    main()
