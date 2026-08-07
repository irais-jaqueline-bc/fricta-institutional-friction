from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
PRIMARY_PATH = PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
PCA_PATH = PROJECT_ROOT / "icdm" / "outputs" / "pca" / "pca_scores.csv"
OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "clustering"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró experiment_config.json: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_representations(
    config: dict,
) -> tuple[pd.Series, dict[str, np.ndarray], dict[str, list[str]]]:
    if not PRIMARY_PATH.exists():
        raise FileNotFoundError(f"No se encontró X_primary_raw.csv: {PRIMARY_PATH}")
    if not PCA_PATH.exists():
        raise FileNotFoundError(f"No se encontró pca_scores.csv: {PCA_PATH}")

    id_column = config["id_column"]
    primary_features = config["primary_features"]

    primary = pd.read_csv(PRIMARY_PATH)
    pca_scores = pd.read_csv(PCA_PATH)

    required_primary = [id_column] + primary_features
    missing_primary = [
        column for column in required_primary if column not in primary.columns
    ]
    if missing_primary:
        raise KeyError(
            "Faltan columnas en X_primary_raw.csv:\n- " + "\n- ".join(missing_primary)
        )

    pc_columns = [column for column in pca_scores.columns if column.startswith("PC")]

    if id_column not in pca_scores.columns:
        raise KeyError(f"Falta {id_column} en pca_scores.csv.")
    if not pc_columns:
        raise ValueError("No se encontraron componentes PC en pca_scores.csv.")

    if primary[id_column].duplicated().any():
        raise ValueError("Hay IDs duplicados en X_primary_raw.csv.")
    if pca_scores[id_column].duplicated().any():
        raise ValueError("Hay IDs duplicados en pca_scores.csv.")

    aligned = primary[[id_column] + primary_features].merge(
        pca_scores[[id_column] + pc_columns],
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) != len(primary) or len(aligned) != len(pca_scores):
        raise ValueError("Los IDs de la matriz primaria y PCA no coinciden.")

    X_raw = aligned[primary_features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    X_pca = aligned[pc_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X_raw.isna().any().any():
        raise ValueError("R0 contiene valores faltantes o no numéricos.")
    if X_pca.isna().any().any():
        raise ValueError("R1 contiene valores faltantes o no numéricos.")

    scaler = StandardScaler()
    X_r0 = scaler.fit_transform(X_raw)

    representations = {
        "R0_STANDARDIZED": X_r0,
        "R1_PCA_85": X_pca.to_numpy(dtype=float),
    }
    feature_names = {
        "R0_STANDARDIZED": primary_features,
        "R1_PCA_85": pc_columns,
    }

    return aligned[id_column].copy(), representations, feature_names


def fit_hac(X: np.ndarray, k: int) -> np.ndarray:
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


def calculate_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict:
    observed_clusters = np.unique(labels)
    if len(observed_clusters) < 2:
        raise ValueError("La solución produjo menos de dos clusters.")

    counts = pd.Series(labels).value_counts()

    return {
        "observed_clusters": int(len(observed_clusters)),
        "silhouette": float(silhouette_score(X, labels, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
    }


def append_sizes(
    size_rows: list[dict],
    candidate_id: str,
    representation: str,
    algorithm: str,
    k: int,
    labels: np.ndarray,
) -> None:
    for cluster, size in pd.Series(labels).value_counts().sort_index().items():
        size_rows.append(
            {
                "candidate_id": candidate_id,
                "representation": representation,
                "algorithm": algorithm,
                "k_requested": k,
                "cluster": int(cluster),
                "size": int(size),
            }
        )


def run_candidates(
    ids: pd.Series,
    representations: dict[str, np.ndarray],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    size_rows = []
    label_columns = {config["id_column"]: ids.to_numpy()}

    k_values = config["clustering"]["k_values"]
    minimum_allowed = config["minimum_cluster_size"]
    seed = config["random_seed"]

    for representation_name, X in representations.items():
        for k in k_values:
            # K-Means
            candidate_id = f"{representation_name}__KMEANS__K{k}"
            model = KMeans(
                n_clusters=k,
                n_init=config["clustering"]["kmeans"]["n_init"],
                random_state=seed,
            )
            labels = model.fit_predict(X)
            metrics = calculate_metrics(X, labels)

            metric_rows.append(
                {
                    "candidate_id": candidate_id,
                    "representation": representation_name,
                    "algorithm": "KMEANS",
                    "k_requested": k,
                    "n_samples": int(X.shape[0]),
                    "n_dimensions": int(X.shape[1]),
                    **metrics,
                    "bic": np.nan,
                    "aic": np.nan,
                    "inertia": float(model.inertia_),
                    "converged": True,
                    "eligible_min_cluster_size": (
                        metrics["minimum_cluster_size"] >= minimum_allowed
                    ),
                }
            )
            label_columns[candidate_id] = labels
            append_sizes(
                size_rows,
                candidate_id,
                representation_name,
                "KMEANS",
                k,
                labels,
            )

            # HAC-Ward
            candidate_id = f"{representation_name}__HAC_WARD__K{k}"
            labels = fit_hac(X, k)
            metrics = calculate_metrics(X, labels)

            metric_rows.append(
                {
                    "candidate_id": candidate_id,
                    "representation": representation_name,
                    "algorithm": "HAC_WARD",
                    "k_requested": k,
                    "n_samples": int(X.shape[0]),
                    "n_dimensions": int(X.shape[1]),
                    **metrics,
                    "bic": np.nan,
                    "aic": np.nan,
                    "inertia": np.nan,
                    "converged": True,
                    "eligible_min_cluster_size": (
                        metrics["minimum_cluster_size"] >= minimum_allowed
                    ),
                }
            )
            label_columns[candidate_id] = labels
            append_sizes(
                size_rows,
                candidate_id,
                representation_name,
                "HAC_WARD",
                k,
                labels,
            )

            # Gaussian Mixture Model
            candidate_id = f"{representation_name}__GMM_DIAG__K{k}"
            model = GaussianMixture(
                n_components=k,
                covariance_type=config["clustering"]["gmm"]["covariance_type"],
                n_init=config["clustering"]["gmm"]["n_init"],
                reg_covar=config["clustering"]["gmm"]["reg_covar"],
                random_state=seed,
            )
            labels = model.fit_predict(X)
            metrics = calculate_metrics(X, labels)

            metric_rows.append(
                {
                    "candidate_id": candidate_id,
                    "representation": representation_name,
                    "algorithm": "GMM_DIAG",
                    "k_requested": k,
                    "n_samples": int(X.shape[0]),
                    "n_dimensions": int(X.shape[1]),
                    **metrics,
                    "bic": float(model.bic(X)),
                    "aic": float(model.aic(X)),
                    "inertia": np.nan,
                    "converged": bool(model.converged_),
                    "eligible_min_cluster_size": (
                        metrics["minimum_cluster_size"] >= minimum_allowed
                    ),
                }
            )
            label_columns[candidate_id] = labels
            append_sizes(
                size_rows,
                candidate_id,
                representation_name,
                "GMM_DIAG",
                k,
                labels,
            )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(size_rows),
        pd.DataFrame(label_columns),
    )


def save_outputs(
    metrics: pd.DataFrame,
    sizes: pd.DataFrame,
    labels_wide: pd.DataFrame,
    config: dict,
    feature_names: dict[str, list[str]],
) -> None:
    labels_long = labels_wide.melt(
        id_vars=[config["id_column"]],
        var_name="candidate_id",
        value_name="cluster",
    )

    metrics = metrics.sort_values(
        [
            "eligible_min_cluster_size",
            "representation",
            "algorithm",
            "k_requested",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    metrics.to_csv(
        OUTPUT_DIR / "all_candidate_metrics.csv",
        index=False,
    )
    sizes.to_csv(
        OUTPUT_DIR / "cluster_size_report.csv",
        index=False,
    )
    labels_wide.to_csv(
        OUTPUT_DIR / "all_candidate_labels.csv",
        index=False,
    )
    labels_long.to_csv(
        OUTPUT_DIR / "all_candidate_labels_long.csv",
        index=False,
    )

    inventory = {
        "status": "FULL_DATA_CLUSTER_SEARCH_COMPLETE",
        "candidate_count": int(len(metrics)),
        "representations": {
            name: {
                "dimensions": len(features),
                "features": features,
            }
            for name, features in feature_names.items()
        },
        "algorithms": ["KMEANS", "HAC_WARD", "GMM_DIAG"],
        "k_values": config["clustering"]["k_values"],
        "minimum_cluster_size_rule": config["minimum_cluster_size"],
        "eligible_candidates": int(metrics["eligible_min_cluster_size"].sum()),
        "ineligible_candidates": int((~metrics["eligible_min_cluster_size"]).sum()),
        "important_note": (
            "This stage does not select a winner. "
            "Candidate selection requires resampling stability."
        ),
    }

    (OUTPUT_DIR / "candidate_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_summary(metrics: pd.DataFrame) -> None:
    print("\n=== CLUSTER SEARCH SUMMARY ===\n")
    print(f"Candidate solutions: {len(metrics)}")
    print(
        "Eligible by minimum cluster size: "
        f"{int(metrics['eligible_min_cluster_size'].sum())}"
    )
    print(
        "Ineligible by minimum cluster size: "
        f"{int((~metrics['eligible_min_cluster_size']).sum())}"
    )

    print("\n=== ALL CANDIDATES ===\n")
    display = metrics[
        [
            "candidate_id",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "bic",
            "minimum_cluster_size",
            "maximum_cluster_size",
            "converged",
            "eligible_min_cluster_size",
        ]
    ].copy()

    print(
        display.to_string(
            index=False,
            formatters={
                "silhouette": lambda value: f"{value:.4f}",
                "davies_bouldin": lambda value: f"{value:.4f}",
                "calinski_harabasz": lambda value: f"{value:.2f}",
                "bic": (lambda value: "" if pd.isna(value) else f"{value:.2f}"),
            },
        )
    )

    print("\n=== TOP ELIGIBLE BY SILHOUETTE ===\n")
    eligible = metrics.loc[metrics["eligible_min_cluster_size"]].copy()

    if eligible.empty:
        print("No eligible candidates.")
    else:
        top = eligible.sort_values(
            "silhouette",
            ascending=False,
        ).head(10)

        print(
            top[
                [
                    "candidate_id",
                    "silhouette",
                    "davies_bouldin",
                    "calinski_harabasz",
                    "minimum_cluster_size",
                ]
            ].to_string(
                index=False,
                formatters={
                    "silhouette": lambda value: f"{value:.4f}",
                    "davies_bouldin": lambda value: f"{value:.4f}",
                    "calinski_harabasz": lambda value: f"{value:.2f}",
                },
            )
        )

    print(
        "\nGATE STATUS: FULL-DATA CANDIDATE SEARCH COMPLETE. "
        "Do not select a winner before stability analysis."
    )


def main() -> None:
    config = load_config()
    ids, representations, feature_names = load_representations(config)

    metrics, sizes, labels_wide = run_candidates(
        ids=ids,
        representations=representations,
        config=config,
    )

    save_outputs(
        metrics=metrics,
        sizes=sizes,
        labels_wide=labels_wide,
        config=config,
        feature_names=feature_names,
    )

    print_summary(metrics)


if __name__ == "__main__":
    main()
