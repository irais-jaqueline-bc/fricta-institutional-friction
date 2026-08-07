from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
PRIMARY_PATH = PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
AUGMENTED_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_sensitivity_augmented_raw.csv"
)
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
ALL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "all_candidate_labels.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "sensitivity"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    for path in [
        CONFIG_PATH,
        PRIMARY_PATH,
        AUGMENTED_PATH,
        FINAL_LABELS_PATH,
        ALL_LABELS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    primary = pd.read_csv(PRIMARY_PATH)
    augmented = pd.read_csv(AUGMENTED_PATH)
    final_labels = pd.read_csv(FINAL_LABELS_PATH)
    all_labels = pd.read_csv(ALL_LABELS_PATH)

    id_column = config["id_column"]
    primary_features = config["primary_features"]
    sensitivity_features = config["sensitivity_only_features"]

    required_primary = [id_column] + primary_features
    required_augmented = [id_column] + primary_features + sensitivity_features

    missing_primary = [c for c in required_primary if c not in primary.columns]
    missing_augmented = [c for c in required_augmented if c not in augmented.columns]

    if missing_primary:
        raise KeyError("Missing primary columns:\n- " + "\n- ".join(missing_primary))

    if missing_augmented:
        raise KeyError(
            "Missing augmented columns:\n- " + "\n- ".join(missing_augmented)
        )

    aligned = (
        primary[required_primary]
        .merge(
            augmented[[id_column] + sensitivity_features],
            on=id_column,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            final_labels[[id_column, "cluster_id"]],
            on=id_column,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            all_labels,
            on=id_column,
            how="inner",
            validate="one_to_one",
        )
    )

    if len(aligned) != len(primary):
        raise ValueError("Institution IDs do not align across sensitivity inputs.")

    numeric_columns = primary_features + sensitivity_features
    aligned[numeric_columns] = aligned[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if aligned[numeric_columns].isna().any().any():
        missing = aligned[numeric_columns].isna().sum()
        raise ValueError(
            "Sensitivity matrix contains missing values:\n"
            + missing[missing > 0].to_string()
        )

    return config, aligned, primary_features, sensitivity_features


def fit_hac(X: np.ndarray, k: int = 2) -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            metric="euclidean",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            affinity="euclidean",
        )

    return model.fit_predict(X)


def transform_matrix(
    X: np.ndarray,
    *,
    standardize: bool,
    use_pca: bool,
    pca_threshold: float = 0.85,
) -> tuple[np.ndarray, int]:
    transformed = X.copy()

    if standardize:
        transformed = StandardScaler().fit_transform(transformed)

    if use_pca:
        full_pca = PCA(svd_solver="full")
        full_pca.fit(transformed)

        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, pca_threshold) + 1)

        transformed = PCA(
            n_components=n_components,
            svd_solver="full",
        ).fit_transform(transformed)

        return transformed, n_components

    return transformed, transformed.shape[1]


def evaluate_solution(
    *,
    analysis_id: str,
    X_transformed: np.ndarray,
    labels: np.ndarray,
    reference_labels: np.ndarray,
    dimensions: int,
    feature_count: int,
    notes: str,
) -> dict:
    counts = pd.Series(labels).value_counts()

    return {
        "analysis_id": analysis_id,
        "adjusted_rand_index_vs_final": float(
            adjusted_rand_score(reference_labels, labels)
        ),
        "normalized_mutual_information_vs_final": float(
            normalized_mutual_info_score(reference_labels, labels)
        ),
        "silhouette": float(silhouette_score(X_transformed, labels)),
        "feature_count": int(feature_count),
        "representation_dimensions": int(dimensions),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
        "cluster_sizes": ";".join(
            f"{int(cluster)}:{int(size)}"
            for cluster, size in counts.sort_index().items()
        ),
        "notes": notes,
    }


def run_core_sensitivities(
    aligned: pd.DataFrame,
    primary_features: list[str],
    sensitivity_features: list[str],
) -> pd.DataFrame:
    reference = aligned["cluster_id"].to_numpy(dtype=int)
    X_primary = aligned[primary_features].to_numpy(dtype=float)
    X_augmented = aligned[primary_features + sensitivity_features].to_numpy(dtype=float)

    specifications = [
        {
            "analysis_id": "BASELINE_RECOMPUTED_PCA85_HAC_K2",
            "X": X_primary,
            "standardize": True,
            "use_pca": True,
            "threshold": 0.85,
            "feature_count": len(primary_features),
            "notes": "Recomputed selected pipeline.",
        },
        {
            "analysis_id": "NO_PCA_STANDARDIZED_HAC_K2",
            "X": X_primary,
            "standardize": True,
            "use_pca": False,
            "threshold": 0.85,
            "feature_count": len(primary_features),
            "notes": "Tests whether PCA changes assignments.",
        },
        {
            "analysis_id": "RAW_0_1_NO_STANDARDIZATION_HAC_K2",
            "X": X_primary,
            "standardize": False,
            "use_pca": False,
            "threshold": 0.85,
            "feature_count": len(primary_features),
            "notes": "Tests sensitivity to StandardScaler.",
        },
        {
            "analysis_id": "AUGMENTED_15_PCA85_HAC_K2",
            "X": X_augmented,
            "standardize": True,
            "use_pca": True,
            "threshold": 0.85,
            "feature_count": len(primary_features) + len(sensitivity_features),
            "notes": (
                "Adds previous implementation and perceived utility; "
                "does not add validation-only or metadata variables."
            ),
        },
        {
            "analysis_id": "AUGMENTED_15_NO_PCA_HAC_K2",
            "X": X_augmented,
            "standardize": True,
            "use_pca": False,
            "threshold": 0.85,
            "feature_count": len(primary_features) + len(sensitivity_features),
            "notes": "Augmented sensitivity matrix without PCA.",
        },
    ]

    rows = []

    for specification in specifications:
        transformed, dimensions = transform_matrix(
            specification["X"],
            standardize=specification["standardize"],
            use_pca=specification["use_pca"],
            pca_threshold=specification["threshold"],
        )
        labels = fit_hac(transformed, k=2)

        rows.append(
            evaluate_solution(
                analysis_id=specification["analysis_id"],
                X_transformed=transformed,
                labels=labels,
                reference_labels=reference,
                dimensions=dimensions,
                feature_count=specification["feature_count"],
                notes=specification["notes"],
            )
        )

    return pd.DataFrame(rows)


def run_leave_one_feature_out(
    aligned: pd.DataFrame,
    primary_features: list[str],
) -> pd.DataFrame:
    reference = aligned["cluster_id"].to_numpy(dtype=int)
    rows = []

    for removed_feature in primary_features:
        retained = [
            feature for feature in primary_features if feature != removed_feature
        ]

        X = aligned[retained].to_numpy(dtype=float)
        transformed, dimensions = transform_matrix(
            X,
            standardize=True,
            use_pca=True,
            pca_threshold=0.85,
        )
        labels = fit_hac(transformed, k=2)

        result = evaluate_solution(
            analysis_id=f"LEAVE_OUT__{removed_feature}",
            X_transformed=transformed,
            labels=labels,
            reference_labels=reference,
            dimensions=dimensions,
            feature_count=len(retained),
            notes="Leave-one-primary-feature-out PCA85 HAC-Ward k=2.",
        )
        result["removed_feature"] = removed_feature
        rows.append(result)

    output = (
        pd.DataFrame(rows)
        .sort_values(
            "adjusted_rand_index_vs_final",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return output


def run_pca_threshold_sensitivity(
    aligned: pd.DataFrame,
    primary_features: list[str],
) -> pd.DataFrame:
    reference = aligned["cluster_id"].to_numpy(dtype=int)
    X = aligned[primary_features].to_numpy(dtype=float)
    rows = []

    for threshold in [0.80, 0.85, 0.90, 0.95]:
        transformed, dimensions = transform_matrix(
            X,
            standardize=True,
            use_pca=True,
            pca_threshold=threshold,
        )
        labels = fit_hac(transformed, k=2)

        result = evaluate_solution(
            analysis_id=f"PCA_THRESHOLD_{int(threshold * 100)}",
            X_transformed=transformed,
            labels=labels,
            reference_labels=reference,
            dimensions=dimensions,
            feature_count=len(primary_features),
            notes="Exploratory PCA retention-threshold sensitivity.",
        )
        result["pca_threshold"] = threshold
        rows.append(result)

    return pd.DataFrame(rows)


def run_algorithmic_reference(
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    reference = aligned["cluster_id"].to_numpy(dtype=int)

    candidate_ids = [
        "R1_PCA_85__KMEANS__K2",
        "R0_STANDARDIZED__KMEANS__K2",
        "R1_PCA_85__KMEANS__K5",
    ]

    rows = []

    for candidate_id in candidate_ids:
        if candidate_id not in aligned.columns:
            continue

        labels = aligned[candidate_id].to_numpy(dtype=int)
        counts = pd.Series(labels).value_counts()

        rows.append(
            {
                "candidate_id": candidate_id,
                "adjusted_rand_index_vs_final": float(
                    adjusted_rand_score(reference, labels)
                ),
                "normalized_mutual_information_vs_final": float(
                    normalized_mutual_info_score(reference, labels)
                ),
                "minimum_cluster_size": int(counts.min()),
                "maximum_cluster_size": int(counts.max()),
                "cluster_sizes": ";".join(
                    f"{int(cluster)}:{int(size)}"
                    for cluster, size in counts.sort_index().items()
                ),
                "interpretation": (
                    "Algorithmic or granularity alternative; not the selected model."
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    config, aligned, primary_features, sensitivity_features = load_inputs()

    core = run_core_sensitivities(
        aligned,
        primary_features,
        sensitivity_features,
    )
    leave_one_out = run_leave_one_feature_out(
        aligned,
        primary_features,
    )
    pca_thresholds = run_pca_threshold_sensitivity(
        aligned,
        primary_features,
    )
    algorithmic = run_algorithmic_reference(aligned)

    core.to_csv(
        OUTPUT_DIR / "sensitivity_summary.csv",
        index=False,
    )
    leave_one_out.to_csv(
        OUTPUT_DIR / "leave_one_feature_out.csv",
        index=False,
    )
    pca_thresholds.to_csv(
        OUTPUT_DIR / "pca_threshold_sensitivity.csv",
        index=False,
    )
    algorithmic.to_csv(
        OUTPUT_DIR / "algorithmic_reference_comparison.csv",
        index=False,
    )

    report = {
        "status": "SENSITIVITY_COMPLETE",
        "selected_model": "R1_PCA_85__HAC_WARD__K2",
        "core_sensitivity_min_ari": float(core["adjusted_rand_index_vs_final"].min()),
        "leave_one_feature_out_min_ari": float(
            leave_one_out["adjusted_rand_index_vs_final"].min()
        ),
        "leave_one_feature_out_median_ari": float(
            leave_one_out["adjusted_rand_index_vs_final"].median()
        ),
        "pca_threshold_min_ari": float(
            pca_thresholds["adjusted_rand_index_vs_final"].min()
        ),
        "any_sensitivity_cluster_below_5": bool(
            (core["minimum_cluster_size"] < 5).any()
            or (leave_one_out["minimum_cluster_size"] < 5).any()
            or (pca_thresholds["minimum_cluster_size"] < 5).any()
        ),
        "interpretation_rule": (
            "ARI and cluster-size changes are descriptive robustness checks. "
            "Sensitivity analyses do not replace the frozen model-selection rule."
        ),
        "generated_files": [
            "icdm/outputs/sensitivity/sensitivity_summary.csv",
            "icdm/outputs/sensitivity/leave_one_feature_out.csv",
            "icdm/outputs/sensitivity/pca_threshold_sensitivity.csv",
            "icdm/outputs/sensitivity/algorithmic_reference_comparison.csv",
            "icdm/outputs/sensitivity/sensitivity_report.json",
        ],
    }

    (OUTPUT_DIR / "sensitivity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== CORE SENSITIVITY SUMMARY ===\n")
    print(
        core[
            [
                "analysis_id",
                "adjusted_rand_index_vs_final",
                "normalized_mutual_information_vs_final",
                "silhouette",
                "representation_dimensions",
                "minimum_cluster_size",
                "maximum_cluster_size",
            ]
        ].to_string(
            index=False,
            formatters={
                "adjusted_rand_index_vs_final": lambda value: f"{value:.4f}",
                "normalized_mutual_information_vs_final": lambda value: f"{value:.4f}",
                "silhouette": lambda value: f"{value:.4f}",
            },
        )
    )

    print("\n=== LEAVE-ONE-FEATURE-OUT SUMMARY ===\n")
    print(
        leave_one_out[
            [
                "removed_feature",
                "adjusted_rand_index_vs_final",
                "silhouette",
                "representation_dimensions",
                "minimum_cluster_size",
                "maximum_cluster_size",
            ]
        ].to_string(
            index=False,
            formatters={
                "adjusted_rand_index_vs_final": lambda value: f"{value:.4f}",
                "silhouette": lambda value: f"{value:.4f}",
            },
        )
    )

    print("\n=== PCA THRESHOLD SENSITIVITY ===\n")
    print(
        pca_thresholds[
            [
                "pca_threshold",
                "representation_dimensions",
                "adjusted_rand_index_vs_final",
                "silhouette",
                "minimum_cluster_size",
                "maximum_cluster_size",
            ]
        ].to_string(
            index=False,
            formatters={
                "pca_threshold": lambda value: f"{value:.2f}",
                "adjusted_rand_index_vs_final": lambda value: f"{value:.4f}",
                "silhouette": lambda value: f"{value:.4f}",
            },
        )
    )

    print("\n=== ALGORITHMIC REFERENCE COMPARISON ===\n")
    print(
        algorithmic.to_string(
            index=False,
            formatters={
                "adjusted_rand_index_vs_final": lambda value: f"{value:.4f}",
                "normalized_mutual_information_vs_final": lambda value: f"{value:.4f}",
            },
        )
    )

    print(
        "\nGATE STATUS: SENSITIVITY COMPLETE. "
        "Next step is figures and paper-ready tables."
    )


if __name__ == "__main__":
    main()
