from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = PROJECT_ROOT / "cipher" / "outputs" / "ensemble" / "smoke"

EXPECTED_FAMILIES = {
    "R0_WARD": 2,
    "R1_PCA85_WARD": 2,
    "R0_KMEANS": 2,
    "R1_PCA85_KMEANS": 2,
}


def read_required_csv(name: str) -> pd.DataFrame:
    path = SMOKE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing smoke output: {path}")
    return pd.read_csv(path)


def read_optional_csv(name: str) -> pd.DataFrame:
    path = SMOKE_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    report_path = SMOKE_DIR / "ensemble_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing smoke report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = read_required_csv("member_manifest.csv")
    metrics = read_required_csv("member_metrics.csv")
    pred_all = read_required_csv("member_predictions_all.csv")
    pred_oob = read_required_csv("member_predictions_oob.csv")
    family_consensus = read_required_csv("family_consensus.csv")
    coassign = read_required_csv("coassignment_matrix.csv")
    coassign_oob = read_required_csv("coassignment_oob_matrix.csv")
    coassign_oob_counts = read_required_csv("coassignment_oob_counts.csv")
    rejected = read_optional_csv("rejected_attempts.csv")

    checks: list[tuple[str, bool]] = []

    checks.append(("report_mode_smoke", report.get("mode") == "smoke"))
    checks.append(("accepted_members_8", report.get("accepted_members") == 8))
    checks.append(("manifest_rows_8", len(manifest) == 8))
    checks.append(("metrics_rows_8", len(metrics) == 8))
    checks.append(("member_ids_unique", manifest["member_id"].is_unique))
    checks.append(
        (
            "manifest_metrics_member_ids_match",
            set(manifest["member_id"]) == set(metrics["member_id"]),
        )
    )

    family_counts = manifest["family"].value_counts().to_dict()
    checks.append(("family_counts_exact", family_counts == EXPECTED_FAMILIES))

    checks.append(("sample_size_65", bool((manifest["sample_size"] == 65).all())))
    checks.append(("oob_size_16", bool((manifest["oob_size"] == 16).all())))
    checks.append(("feature_count_11", bool((manifest["feature_count"] == 11).all())))

    pca_rows = manifest["representation"] == "PCA85"
    raw_rows = manifest["representation"] == "RAW_STANDARDIZED"

    checks.append(
        (
            "pca_components_present",
            bool(manifest.loc[pca_rows, "pca_components"].notna().all()),
        )
    )
    checks.append(
        (
            "pca_variance_at_least_085",
            bool(
                (manifest.loc[pca_rows, "pca_explained_variance"] >= 0.85 - 1e-12).all()
            ),
        )
    )
    checks.append(
        (
            "raw_pca_fields_empty",
            bool(manifest.loc[raw_rows, "pca_components"].isna().all()),
        )
    )

    finite_metric_columns = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "reference_ari_on_sample",
        "ward_or_native_extension_fidelity_on_sample",
        "minimum_sampled_cluster_size",
    ]
    checks.append(
        (
            "all_metrics_finite",
            bool(
                np.isfinite(metrics[finite_metric_columns].to_numpy(dtype=float)).all()
            ),
        )
    )
    checks.append(
        (
            "minimum_cluster_size_at_least_5",
            bool((metrics["minimum_sampled_cluster_size"] >= 5).all()),
        )
    )

    ward_mask = metrics["family"].str.contains("WARD")
    kmeans_mask = metrics["family"].str.contains("KMEANS")

    ward_fidelity_min = float(
        metrics.loc[
            ward_mask,
            "ward_or_native_extension_fidelity_on_sample",
        ].min()
    )
    checks.append(("ward_extension_fidelity_at_least_095", ward_fidelity_min >= 0.95))
    checks.append(
        (
            "kmeans_native_fidelity_exact_1",
            bool(
                np.isclose(
                    metrics.loc[
                        kmeans_mask,
                        "ward_or_native_extension_fidelity_on_sample",
                    ],
                    1.0,
                ).all()
            ),
        )
    )

    member_columns = manifest["member_id"].tolist()
    expected_prediction_columns = {"institution_id", *member_columns}
    checks.append(
        (
            "all_prediction_columns_match",
            set(pred_all.columns) == expected_prediction_columns,
        )
    )
    checks.append(
        (
            "oob_prediction_columns_match",
            set(pred_oob.columns) == expected_prediction_columns,
        )
    )
    checks.append(("prediction_rows_81", len(pred_all) == 81))
    checks.append(("oob_prediction_rows_81", len(pred_oob) == 81))

    non_null_all_values = pd.unique(pred_all[member_columns].to_numpy().ravel())
    non_null_all_values = {
        int(value) for value in non_null_all_values if pd.notna(value)
    }
    checks.append(("all_predictions_only_profiles_1_2", non_null_all_values == {1, 2}))

    oob_counts_by_member = pred_oob[member_columns].notna().sum(axis=0)
    checks.append(
        (
            "exactly_16_oob_predictions_per_member",
            bool((oob_counts_by_member == 16).all()),
        )
    )

    checks.append(("family_consensus_rows_324", len(family_consensus) == 81 * 4))

    def square_matrix_from_csv(frame: pd.DataFrame) -> np.ndarray:
        return frame.drop(columns=[frame.columns[0]]).to_numpy(dtype=float)

    coassign_matrix = square_matrix_from_csv(coassign)
    checks.append(("coassignment_shape_81x81", coassign_matrix.shape == (81, 81)))
    checks.append(
        (
            "coassignment_symmetric",
            bool(np.allclose(coassign_matrix, coassign_matrix.T, equal_nan=True)),
        )
    )
    checks.append(
        (
            "coassignment_diagonal_1",
            bool(np.allclose(np.diag(coassign_matrix), 1.0)),
        )
    )
    checks.append(
        (
            "coassignment_range_0_1",
            bool(
                np.nanmin(coassign_matrix) >= 0.0 and np.nanmax(coassign_matrix) <= 1.0
            ),
        )
    )

    oob_matrix = square_matrix_from_csv(coassign_oob)
    oob_count_matrix = square_matrix_from_csv(coassign_oob_counts)
    checks.append(("oob_coassignment_shape_81x81", oob_matrix.shape == (81, 81)))
    checks.append(
        (
            "oob_count_matrix_symmetric",
            bool(np.allclose(oob_count_matrix, oob_count_matrix.T)),
        )
    )
    checks.append(
        (
            "oob_coassignment_symmetric",
            bool(np.allclose(oob_matrix, oob_matrix.T, equal_nan=True)),
        )
    )

    failed_checks = [name for name, passed in checks if not passed]

    print("\n=== CIPHER STAGE 1 SMOKE AUDIT ===\n")
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\n=== METRIC SUMMARY ===\n")
    summary_columns = [
        "family",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "reference_ari_on_sample",
        "ward_or_native_extension_fidelity_on_sample",
        "minimum_sampled_cluster_size",
    ]
    print(metrics[summary_columns].to_string(index=False))

    print("\n=== FAMILY SUMMARY ===\n")
    grouped = (
        metrics.groupby("family")
        .agg(
            members=("member_id", "count"),
            silhouette_median=("silhouette", "median"),
            reference_ari_median=("reference_ari_on_sample", "median"),
            extension_fidelity_min=(
                "ward_or_native_extension_fidelity_on_sample",
                "min",
            ),
            minimum_cluster_size=("minimum_sampled_cluster_size", "min"),
        )
        .reset_index()
    )
    print(grouped.to_string(index=False))

    print("\nRejected attempts:", len(rejected))
    print(
        "OOB predictions per institution:",
        f"min={pred_oob[member_columns].notna().sum(axis=1).min()},",
        f"median={pred_oob[member_columns].notna().sum(axis=1).median():.1f},",
        f"max={pred_oob[member_columns].notna().sum(axis=1).max()}",
    )
    print(f"Ward extension fidelity minimum: {ward_fidelity_min:.4f}")

    if failed_checks:
        print("\nGATE STATUS: FAIL_SMOKE_AUDIT")
        print("Failed checks:", ", ".join(failed_checks))
        raise SystemExit(1)

    print("\nGATE STATUS: PASS_SMOKE_AUDIT")
    print(
        "The official 1,000-member run may be authorized after reviewing these metrics."
    )


if __name__ == "__main__":
    main()
