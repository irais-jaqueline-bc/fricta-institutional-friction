from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
PRIMARY_PATH = PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
VALIDATION_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "features" / "validation_variables_all_rows.csv"
)
ALIGNMENT_PATH = (
    PROJECT_ROOT
    / "icdm"
    / "outputs"
    / "alignment"
    / "cluster_archetype_assignments.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "interpretability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CV_SPLITS = 5
CV_REPEATS = 10
PERMUTATION_REPEATS = 20


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró la configuración: {CONFIG_PATH}")

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_data(
    config: dict,
) -> tuple[pd.DataFrame, list[str], str]:
    for path in [
        PRIMARY_PATH,
        FINAL_LABELS_PATH,
        VALIDATION_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró un archivo requerido: {path}")

    id_column = config["id_column"]
    features = config["primary_features"]

    primary = pd.read_csv(PRIMARY_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)

    required_primary = [id_column] + features
    missing_primary = [
        column for column in required_primary if column not in primary.columns
    ]

    if missing_primary:
        raise KeyError(
            "Faltan columnas en X_primary_raw.csv:\n- " + "\n- ".join(missing_primary)
        )

    required_label_columns = [
        id_column,
        "cluster_id",
    ]

    missing_label_columns = [
        column for column in required_label_columns if column not in labels.columns
    ]

    if missing_label_columns:
        raise KeyError(
            "Faltan columnas en final_cluster_labels.csv:\n- "
            + "\n- ".join(missing_label_columns)
        )

    merged = primary[required_primary].merge(
        labels[required_label_columns],
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(primary):
        raise ValueError("No todas las instituciones tienen cluster final.")

    if merged[features].isna().any().any():
        raise ValueError("Hay valores faltantes en las features primarias.")

    cluster_values = sorted(merged["cluster_id"].unique().tolist())

    if cluster_values != [1, 2]:
        raise ValueError(
            "Este script espera exactamente los clusters 1 y 2. "
            f"Encontrados: {cluster_values}"
        )

    return merged, features, id_column


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    n = len(values)

    order = np.argsort(values)
    ranked = values[order]

    adjusted_ranked = np.empty(n, dtype=float)
    running_min = 1.0

    for index in range(n - 1, -1, -1):
        rank = index + 1
        adjusted = ranked[index] * n / rank
        running_min = min(running_min, adjusted)
        adjusted_ranked[index] = min(running_min, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked

    return pd.Series(
        adjusted,
        index=p_values.index,
    )


def cliffs_delta(
    higher_group: np.ndarray,
    lower_group: np.ndarray,
) -> float:
    comparisons = higher_group[:, None] - lower_group[None, :]

    greater = np.count_nonzero(comparisons > 0)
    lower = np.count_nonzero(comparisons < 0)

    denominator = comparisons.size

    if denominator == 0:
        return np.nan

    return float((greater - lower) / denominator)


def hedges_g(
    cluster_2: np.ndarray,
    cluster_1: np.ndarray,
) -> float:
    n2 = len(cluster_2)
    n1 = len(cluster_1)

    variance_2 = np.var(cluster_2, ddof=1)
    variance_1 = np.var(cluster_1, ddof=1)

    pooled_variance = (((n2 - 1) * variance_2) + ((n1 - 1) * variance_1)) / (
        n2 + n1 - 2
    )

    if pooled_variance <= 0:
        return 0.0

    cohen_d = (np.mean(cluster_2) - np.mean(cluster_1)) / np.sqrt(pooled_variance)

    correction = 1 - (3 / (4 * (n2 + n1) - 9))

    return float(cohen_d * correction)


def build_cluster_profiles(
    data: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_means = data[features].mean()
    overall_stds = data[features].std(ddof=0)

    rows = []

    for cluster_id, group in data.groupby(
        "cluster_id",
        sort=True,
    ):
        for feature in features:
            mean_value = float(group[feature].mean())
            median_value = float(group[feature].median())
            std_value = float(group[feature].std(ddof=1))

            overall_std = float(overall_stds[feature])

            mean_z = (
                (mean_value - float(overall_means[feature])) / overall_std
                if overall_std > 0
                else 0.0
            )

            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(group)),
                    "feature": feature,
                    "mean": mean_value,
                    "median": median_value,
                    "std": std_value,
                    "sample_overall_mean": float(overall_means[feature]),
                    "cluster_mean_z_vs_sample": mean_z,
                }
            )

    long_table = pd.DataFrame(rows)

    wide_table = (
        long_table.pivot(
            index="feature",
            columns="cluster_id",
            values="mean",
        )
        .rename(
            columns={
                1: "cluster_1_mean",
                2: "cluster_2_mean",
            }
        )
        .reset_index()
    )

    long_table.to_csv(
        OUTPUT_DIR / "cluster_feature_profiles_long.csv",
        index=False,
    )

    wide_table.to_csv(
        OUTPUT_DIR / "cluster_feature_profiles_wide.csv",
        index=False,
    )

    return long_table, wide_table


def build_feature_contrasts(
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []

    cluster_1 = data.loc[data["cluster_id"] == 1]

    cluster_2 = data.loc[data["cluster_id"] == 2]

    for feature in features:
        values_1 = cluster_1[feature].to_numpy(dtype=float)

        values_2 = cluster_2[feature].to_numpy(dtype=float)

        statistic, p_value = mannwhitneyu(
            values_2,
            values_1,
            alternative="two-sided",
            method="auto",
        )

        delta = cliffs_delta(
            values_2,
            values_1,
        )

        g = hedges_g(
            values_2,
            values_1,
        )

        rows.append(
            {
                "feature": feature,
                "cluster_1_n": len(values_1),
                "cluster_2_n": len(values_2),
                "cluster_1_mean": float(np.mean(values_1)),
                "cluster_2_mean": float(np.mean(values_2)),
                "mean_difference_cluster_2_minus_1": float(
                    np.mean(values_2) - np.mean(values_1)
                ),
                "cluster_1_median": float(np.median(values_1)),
                "cluster_2_median": float(np.median(values_2)),
                "mann_whitney_u": float(statistic),
                "mann_whitney_p": float(p_value),
                "cliffs_delta_cluster_2_vs_1": delta,
                "absolute_cliffs_delta": abs(delta),
                "hedges_g_cluster_2_vs_1": g,
                "absolute_hedges_g": abs(g),
                "direction": (
                    "Cluster 2 higher friction"
                    if np.mean(values_2) > np.mean(values_1)
                    else "Cluster 1 higher friction"
                ),
            }
        )

    contrasts = pd.DataFrame(rows)

    contrasts["mann_whitney_q_bh"] = benjamini_hochberg(contrasts["mann_whitney_p"])

    contrasts = contrasts.sort_values(
        [
            "absolute_cliffs_delta",
            "absolute_hedges_g",
        ],
        ascending=False,
    ).reset_index(drop=True)

    contrasts.insert(
        0,
        "rank",
        np.arange(1, len(contrasts) + 1),
    )

    contrasts.to_csv(
        OUTPUT_DIR / "feature_contrasts.csv",
        index=False,
    )

    return contrasts


def build_top_indicator_table(
    profile_long: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    rows = []

    for cluster_id in [1, 2]:
        group = profile_long.loc[profile_long["cluster_id"] == cluster_id].copy()

        high = group.sort_values(
            "cluster_mean_z_vs_sample",
            ascending=False,
        ).head(top_n)

        low = group.sort_values(
            "cluster_mean_z_vs_sample",
            ascending=True,
        ).head(top_n)

        for rank, row in enumerate(
            high.itertuples(index=False),
            start=1,
        ):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "indicator_direction": ("HIGHER_FRICTION_THAN_SAMPLE"),
                    "rank": rank,
                    "feature": row.feature,
                    "mean": row.mean,
                    "cluster_mean_z_vs_sample": (row.cluster_mean_z_vs_sample),
                }
            )

        for rank, row in enumerate(
            low.itertuples(index=False),
            start=1,
        ):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "indicator_direction": ("LOWER_FRICTION_THAN_SAMPLE"),
                    "rank": rank,
                    "feature": row.feature,
                    "mean": row.mean,
                    "cluster_mean_z_vs_sample": (row.cluster_mean_z_vs_sample),
                }
            )

    top_indicators = pd.DataFrame(rows)

    top_indicators.to_csv(
        OUTPUT_DIR / "cluster_top_indicators.csv",
        index=False,
    )

    return top_indicators


def validate_implementation_difficulty(
    data: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame:
    validation = pd.read_csv(VALIDATION_PATH)

    if id_column not in validation.columns:
        raise KeyError(f"Falta {id_column} en validation_variables_all_rows.csv")

    candidate_columns = [column for column in validation.columns if column != id_column]

    if len(candidate_columns) != 1:
        raise ValueError(
            "Se esperaba una sola variable de validación. "
            f"Encontradas: {candidate_columns}"
        )

    feature = candidate_columns[0]

    aligned = data[[id_column, "cluster_id"]].merge(
        validation[[id_column, feature]],
        on=id_column,
        how="left",
        validate="one_to_one",
    )

    aligned[feature] = pd.to_numeric(
        aligned[feature],
        errors="coerce",
    )

    cluster_1 = (
        aligned.loc[
            aligned["cluster_id"] == 1,
            feature,
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    cluster_2 = (
        aligned.loc[
            aligned["cluster_id"] == 2,
            feature,
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    statistic, p_value = mannwhitneyu(
        cluster_2,
        cluster_1,
        alternative="two-sided",
        method="auto",
    )

    delta = cliffs_delta(
        cluster_2,
        cluster_1,
    )

    g = hedges_g(
        cluster_2,
        cluster_1,
    )

    result = pd.DataFrame(
        [
            {
                "validation_feature": feature,
                "cluster_1_complete_n": len(cluster_1),
                "cluster_2_complete_n": len(cluster_2),
                "cluster_1_mean": float(np.mean(cluster_1)),
                "cluster_2_mean": float(np.mean(cluster_2)),
                "mean_difference_cluster_2_minus_1": float(
                    np.mean(cluster_2) - np.mean(cluster_1)
                ),
                "cluster_1_median": float(np.median(cluster_1)),
                "cluster_2_median": float(np.median(cluster_2)),
                "mann_whitney_u": float(statistic),
                "mann_whitney_p": float(p_value),
                "cliffs_delta_cluster_2_vs_1": delta,
                "hedges_g_cluster_2_vs_1": g,
                "imputation_used": False,
                "interpretation": (
                    "External descriptive criterion only; " "not used to form clusters."
                ),
            }
        ]
    )

    result.to_csv(
        OUTPUT_DIR / "implementation_difficulty_validation.csv",
        index=False,
    )

    return result


def run_surrogate_cv(
    data: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data[features].to_numpy(dtype=float)
    y = data["cluster_id"].to_numpy(dtype=int)

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_SEED,
    )

    metric_rows = []
    importance_rows = []

    for split_index, (train_index, test_index) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=(RANDOM_SEED + split_index),
            n_jobs=-1,
        )

        model.fit(
            X[train_index],
            y[train_index],
        )

        predictions = model.predict(X[test_index])

        baseline_balanced_accuracy = balanced_accuracy_score(
            y[test_index],
            predictions,
        )

        metric_rows.append(
            {
                "split": split_index,
                "accuracy": accuracy_score(
                    y[test_index],
                    predictions,
                ),
                "balanced_accuracy": (baseline_balanced_accuracy),
                "macro_f1": f1_score(
                    y[test_index],
                    predictions,
                    average="macro",
                ),
                "test_size": len(test_index),
                "test_cluster_1_n": int(np.sum(y[test_index] == 1)),
                "test_cluster_2_n": int(np.sum(y[test_index] == 2)),
            }
        )

        rng = np.random.default_rng(RANDOM_SEED + split_index)

        for feature_index, feature in enumerate(features):
            drops = []

            for repeat in range(PERMUTATION_REPEATS):
                permuted_test = X[test_index].copy()

                shuffled_values = permuted_test[
                    :,
                    feature_index,
                ].copy()

                rng.shuffle(shuffled_values)

                permuted_test[
                    :,
                    feature_index,
                ] = shuffled_values

                permuted_predictions = model.predict(permuted_test)

                permuted_score = balanced_accuracy_score(
                    y[test_index],
                    permuted_predictions,
                )

                drops.append(baseline_balanced_accuracy - permuted_score)

            importance_rows.append(
                {
                    "split": split_index,
                    "feature": feature,
                    "importance_mean_within_split": float(np.mean(drops)),
                    "importance_std_within_split": float(
                        np.std(
                            drops,
                            ddof=1,
                        )
                    ),
                }
            )

    fold_metrics = pd.DataFrame(metric_rows)

    fold_metrics.to_csv(
        OUTPUT_DIR / "surrogate_cv_fold_metrics.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "cv_splits": CV_SPLITS,
                "cv_repeats": CV_REPEATS,
                "total_test_folds": len(fold_metrics),
                "accuracy_mean": float(fold_metrics["accuracy"].mean()),
                "accuracy_std": float(fold_metrics["accuracy"].std(ddof=1)),
                "balanced_accuracy_mean": float(
                    fold_metrics["balanced_accuracy"].mean()
                ),
                "balanced_accuracy_std": float(
                    fold_metrics["balanced_accuracy"].std(ddof=1)
                ),
                "macro_f1_mean": float(fold_metrics["macro_f1"].mean()),
                "macro_f1_std": float(fold_metrics["macro_f1"].std(ddof=1)),
                "surrogate_interpretation_only": True,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "surrogate_cv_metrics.csv",
        index=False,
    )

    importance_raw = pd.DataFrame(importance_rows)

    importance_summary = (
        importance_raw.groupby(
            "feature",
            as_index=False,
        )
        .agg(
            permutation_importance_mean=(
                "importance_mean_within_split",
                "mean",
            ),
            permutation_importance_std=(
                "importance_mean_within_split",
                "std",
            ),
            positive_importance_fraction=(
                "importance_mean_within_split",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
        )
        .sort_values(
            "permutation_importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(importance_summary) + 1,
        ),
    )

    importance_raw.to_csv(
        OUTPUT_DIR / "permutation_importance_by_split.csv",
        index=False,
    )

    importance_summary.to_csv(
        OUTPUT_DIR / "permutation_importance.csv",
        index=False,
    )

    return summary, importance_summary


def save_enriched_assignments(
    data: pd.DataFrame,
    id_column: str,
) -> None:
    enriched = data.copy()

    if ALIGNMENT_PATH.exists():
        alignment = pd.read_csv(ALIGNMENT_PATH)

        additional_columns = [
            column
            for column in alignment.columns
            if column
            not in {
                id_column,
                "cluster_id",
                "cluster_label_zero_based",
            }
        ]

        if additional_columns:
            enriched = enriched.merge(
                alignment[[id_column] + additional_columns],
                on=id_column,
                how="left",
                validate="one_to_one",
            )

    enriched.to_csv(
        OUTPUT_DIR / "institution_assignments_enriched.csv",
        index=False,
    )


def main() -> None:
    config = load_config()
    data, features, id_column = load_data(config)

    profile_long, _ = build_cluster_profiles(
        data,
        features,
    )

    contrasts = build_feature_contrasts(
        data,
        features,
    )

    top_indicators = build_top_indicator_table(
        profile_long,
        top_n=5,
    )

    validation = validate_implementation_difficulty(
        data,
        id_column,
    )

    surrogate_summary, importance = run_surrogate_cv(
        data,
        features,
    )

    save_enriched_assignments(
        data,
        id_column,
    )

    macro_f1 = float(
        surrogate_summary.loc[
            0,
            "macro_f1_mean",
        ]
    )

    report = {
        "status": "EMPIRICAL_INTERPRETATION_COMPLETE",
        "clusters": {
            str(cluster_id): int(size)
            for cluster_id, size in (
                data["cluster_id"].value_counts().sort_index().items()
            )
        },
        "feature_contrast_direction": (
            "Positive values indicate higher friction " "in Cluster 2 than Cluster 1."
        ),
        "top_discriminating_features": (
            contrasts.head(5)[
                [
                    "feature",
                    "mean_difference_cluster_2_minus_1",
                    "cliffs_delta_cluster_2_vs_1",
                    "mann_whitney_q_bh",
                ]
            ].to_dict(orient="records")
        ),
        "surrogate": {
            "model": "RandomForestClassifier",
            "purpose": (
                "Descriptive reconstruction of cluster labels; "
                "not outcome prediction or external validation."
            ),
            "macro_f1_mean": macro_f1,
            "balanced_accuracy_mean": float(
                surrogate_summary.loc[
                    0,
                    "balanced_accuracy_mean",
                ]
            ),
            "shap_threshold": 0.75,
            "shap_eligible": bool(macro_f1 >= 0.75),
            "shap_run": False,
        },
        "implementation_difficulty_validation": (validation.iloc[0].to_dict()),
        "cluster_names_assigned": False,
        "generated_files": [
            "icdm/outputs/interpretability/cluster_feature_profiles_long.csv",
            "icdm/outputs/interpretability/cluster_feature_profiles_wide.csv",
            "icdm/outputs/interpretability/feature_contrasts.csv",
            "icdm/outputs/interpretability/cluster_top_indicators.csv",
            "icdm/outputs/interpretability/implementation_difficulty_validation.csv",
            "icdm/outputs/interpretability/surrogate_cv_fold_metrics.csv",
            "icdm/outputs/interpretability/surrogate_cv_metrics.csv",
            "icdm/outputs/interpretability/permutation_importance_by_split.csv",
            "icdm/outputs/interpretability/permutation_importance.csv",
            "icdm/outputs/interpretability/institution_assignments_enriched.csv",
            "icdm/outputs/interpretability/interpretability_report.json",
        ],
    }

    (OUTPUT_DIR / "interpretability_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== EMPIRICAL PROFILE SUMMARY ===\n")

    print(
        data["cluster_id"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster_id")
        .reset_index(name="size")
        .to_string(index=False)
    )

    print("\n=== TOP FEATURE CONTRASTS " "(CLUSTER 2 MINUS CLUSTER 1) ===\n")

    print(
        contrasts.head(13)[
            [
                "rank",
                "feature",
                "cluster_1_mean",
                "cluster_2_mean",
                "mean_difference_cluster_2_minus_1",
                "cliffs_delta_cluster_2_vs_1",
                "hedges_g_cluster_2_vs_1",
                "mann_whitney_q_bh",
            ]
        ].to_string(
            index=False,
            formatters={
                "cluster_1_mean": (lambda value: f"{value:.3f}"),
                "cluster_2_mean": (lambda value: f"{value:.3f}"),
                "mean_difference_cluster_2_minus_1": (lambda value: f"{value:.3f}"),
                "cliffs_delta_cluster_2_vs_1": (lambda value: f"{value:.3f}"),
                "hedges_g_cluster_2_vs_1": (lambda value: f"{value:.3f}"),
                "mann_whitney_q_bh": (lambda value: f"{value:.5f}"),
            },
        )
    )

    print("\n=== TOP RELATIVE INDICATORS BY CLUSTER ===\n")

    print(
        top_indicators.loc[
            top_indicators["indicator_direction"] == "HIGHER_FRICTION_THAN_SAMPLE"
        ][
            [
                "cluster_id",
                "rank",
                "feature",
                "mean",
                "cluster_mean_z_vs_sample",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean": (lambda value: f"{value:.3f}"),
                "cluster_mean_z_vs_sample": (lambda value: f"{value:.3f}"),
            },
        )
    )

    print("\n=== IMPLEMENTATION DIFFICULTY VALIDATION ===\n")

    print(
        validation.to_string(
            index=False,
            formatters={
                "cluster_1_mean": (lambda value: f"{value:.3f}"),
                "cluster_2_mean": (lambda value: f"{value:.3f}"),
                "mean_difference_cluster_2_minus_1": (lambda value: f"{value:.3f}"),
                "cliffs_delta_cluster_2_vs_1": (lambda value: f"{value:.3f}"),
                "hedges_g_cluster_2_vs_1": (lambda value: f"{value:.3f}"),
                "mann_whitney_p": (lambda value: f"{value:.5f}"),
            },
        )
    )

    print("\n=== SURROGATE MODEL SUMMARY ===\n")

    print(
        surrogate_summary.to_string(
            index=False,
            formatters={
                "accuracy_mean": (lambda value: f"{value:.4f}"),
                "accuracy_std": (lambda value: f"{value:.4f}"),
                "balanced_accuracy_mean": (lambda value: f"{value:.4f}"),
                "balanced_accuracy_std": (lambda value: f"{value:.4f}"),
                "macro_f1_mean": (lambda value: f"{value:.4f}"),
                "macro_f1_std": (lambda value: f"{value:.4f}"),
            },
        )
    )

    print("\n=== TOP PERMUTATION IMPORTANCE ===\n")

    print(
        importance.head(10).to_string(
            index=False,
            formatters={
                "permutation_importance_mean": (lambda value: f"{value:.4f}"),
                "permutation_importance_std": (lambda value: f"{value:.4f}"),
                "positive_importance_fraction": (lambda value: f"{value:.3f}"),
            },
        )
    )

    print(
        "\nGATE STATUS: EMPIRICAL INTERPRETATION COMPLETE. "
        "Review contrasts before assigning profile names."
    )


if __name__ == "__main__":
    main()
