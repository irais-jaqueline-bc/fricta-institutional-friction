from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
PRIMARY_PATH = PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
METRICS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "all_candidate_metrics.csv"
)
LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "all_candidate_labels.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "stability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FRICTA clustering stability analysis."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the frozen iteration count. Final analysis should use 1000.",
    )
    return parser.parse_args()


def load_inputs():
    for path in [CONFIG_PATH, PRIMARY_PATH, METRICS_PATH, LABELS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    primary = pd.read_csv(PRIMARY_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    labels = pd.read_csv(LABELS_PATH)

    id_column = config["id_column"]
    features = config["primary_features"]

    if primary[id_column].duplicated().any():
        raise ValueError("Duplicate institution IDs in primary matrix.")

    if labels[id_column].duplicated().any():
        raise ValueError("Duplicate institution IDs in label matrix.")

    merged = primary[[id_column] + features].merge(
        labels,
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(primary) or len(merged) != len(labels):
        raise ValueError("Institution IDs do not align across input files.")

    X = merged[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Primary matrix contains missing/non-numeric values.")

    return config, merged[id_column].copy(), X, metrics, merged


def build_shortlist(metrics: pd.DataFrame) -> pd.DataFrame:
    eligible = metrics.loc[metrics["eligible_min_cluster_size"].astype(bool)].copy()

    selections: dict[str, set[str]] = {}

    for (representation, algorithm), group in eligible.groupby(
        ["representation", "algorithm"],
        sort=True,
    ):
        rules = {
            "BEST_SILHOUETTE": group.loc[group["silhouette"].idxmax(), "candidate_id"],
            "BEST_DAVIES_BOULDIN": group.loc[
                group["davies_bouldin"].idxmin(), "candidate_id"
            ],
            "BEST_CALINSKI_HARABASZ": group.loc[
                group["calinski_harabasz"].idxmax(), "candidate_id"
            ],
        }

        if algorithm == "GMM_DIAG":
            valid_bic = group.dropna(subset=["bic"])
            if not valid_bic.empty:
                rules["BEST_BIC_WITHIN_REPRESENTATION"] = valid_bic.loc[
                    valid_bic["bic"].idxmin(), "candidate_id"
                ]

        for reason, candidate_id in rules.items():
            selections.setdefault(candidate_id, set()).add(reason)

    selected_ids = sorted(selections)

    shortlist = eligible.loc[eligible["candidate_id"].isin(selected_ids)].copy()

    shortlist["shortlist_reasons"] = shortlist["candidate_id"].map(
        lambda candidate_id: ";".join(sorted(selections[candidate_id]))
    )

    shortlist = shortlist.sort_values(
        ["representation", "algorithm", "k_requested"]
    ).reset_index(drop=True)

    return shortlist


def candidate_seed(base_seed: int, candidate_id: str) -> int:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return (base_seed + offset) % (2**32 - 1)


def prepare_representation(
    X_subset: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, int]:
    standardized = StandardScaler().fit_transform(X_subset)

    if representation == "R0_STANDARDIZED":
        return standardized, standardized.shape[1]

    if representation == "R1_PCA_85":
        full_pca = PCA(svd_solver="full")
        full_pca.fit(standardized)

        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, 0.85) + 1)

        pca = PCA(n_components=n_components, svd_solver="full")
        transformed = pca.fit_transform(standardized)

        return transformed, n_components

    raise ValueError(f"Unknown representation: {representation}")


def fit_candidate(
    X: np.ndarray,
    algorithm: str,
    k: int,
    config: dict,
    random_state: int,
) -> np.ndarray:
    if algorithm == "KMEANS":
        model = KMeans(
            n_clusters=k,
            n_init=config["clustering"]["kmeans"]["n_init"],
            random_state=random_state,
        )
        return model.fit_predict(X)

    if algorithm == "HAC_WARD":
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

    if algorithm == "GMM_DIAG":
        model = GaussianMixture(
            n_components=k,
            covariance_type=config["clustering"]["gmm"]["covariance_type"],
            n_init=config["clustering"]["gmm"]["n_init"],
            reg_covar=config["clustering"]["gmm"]["reg_covar"],
            random_state=random_state,
        )
        return model.fit_predict(X)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def matched_cluster_jaccards(
    reference_labels: np.ndarray,
    resampled_labels: np.ndarray,
) -> list[dict]:
    reference_clusters = np.sort(np.unique(reference_labels))
    resampled_clusters = np.sort(np.unique(resampled_labels))

    matrix = np.zeros(
        (len(reference_clusters), len(resampled_clusters)),
        dtype=float,
    )

    for i, reference_cluster in enumerate(reference_clusters):
        reference_members = reference_labels == reference_cluster

        for j, resampled_cluster in enumerate(resampled_clusters):
            resampled_members = resampled_labels == resampled_cluster

            intersection = np.logical_and(
                reference_members,
                resampled_members,
            ).sum()

            union = np.logical_or(
                reference_members,
                resampled_members,
            ).sum()

            matrix[i, j] = float(intersection / union) if union > 0 else 0.0

    row_indices, column_indices = linear_sum_assignment(-matrix)

    results = []

    for row, column in zip(row_indices, column_indices):
        results.append(
            {
                "reference_cluster": int(reference_clusters[row]),
                "matched_resample_cluster": int(resampled_clusters[column]),
                "jaccard": float(matrix[row, column]),
            }
        )

    return results


def update_consensus(
    sampled_indices: np.ndarray,
    labels: np.ndarray,
    co_sampled: np.ndarray,
    co_clustered: np.ndarray,
) -> None:
    ix = np.ix_(sampled_indices, sampled_indices)
    co_sampled[ix] += 1

    same_cluster = labels[:, None] == labels[None, :]
    co_clustered[ix] += same_cluster.astype(np.int32)


def summarize_candidate(
    candidate_id: str,
    distribution: pd.DataFrame,
    clusterwise: pd.DataFrame,
) -> dict:
    ari = distribution["ari"]
    mean_jaccard = distribution["mean_matched_jaccard"]

    summary = {
        "candidate_id": candidate_id,
        "successful_iterations": int(len(distribution)),
        "ari_mean": float(ari.mean()),
        "ari_median": float(ari.median()),
        "ari_std": float(ari.std(ddof=1)),
        "ari_p025": float(ari.quantile(0.025)),
        "ari_p975": float(ari.quantile(0.975)),
        "mean_matched_jaccard_mean": float(mean_jaccard.mean()),
        "mean_matched_jaccard_median": float(mean_jaccard.median()),
        "minimum_resample_cluster_size_median": float(
            distribution["minimum_resample_cluster_size"].median()
        ),
        "minimum_resample_cluster_size_min": int(
            distribution["minimum_resample_cluster_size"].min()
        ),
        "pca_components_median": (
            float(distribution["representation_dimensions"].median())
        ),
        "pca_components_min": int(distribution["representation_dimensions"].min()),
        "pca_components_max": int(distribution["representation_dimensions"].max()),
        "clusterwise_jaccard_min_mean": float(
            clusterwise.groupby("reference_cluster")["jaccard"].mean().min()
        ),
    }

    return summary


def consensus_to_long(
    candidate_id: str,
    ids: pd.Series,
    co_sampled: np.ndarray,
    co_clustered: np.ndarray,
) -> pd.DataFrame:
    rows = []
    n = len(ids)

    for i in range(n):
        for j in range(i, n):
            denominator = int(co_sampled[i, j])
            numerator = int(co_clustered[i, j])

            rows.append(
                {
                    "candidate_id": candidate_id,
                    "institution_a": ids.iloc[i],
                    "institution_b": ids.iloc[j],
                    "co_sample_count": denominator,
                    "co_cluster_count": numerator,
                    "consensus": (
                        float(numerator / denominator) if denominator > 0 else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config, ids, X_df, metrics, merged_labels = load_inputs()

    frozen_iterations = int(config["stability"]["iterations"])
    iterations = args.iterations or frozen_iterations

    if iterations <= 0:
        raise ValueError("Iterations must be positive.")

    sample_fraction = float(config["stability"]["sample_fraction"])
    sample_size = int(round(len(X_df) * sample_fraction))
    base_seed = int(config["random_seed"])

    shortlist = build_shortlist(metrics)
    shortlist.to_csv(
        OUTPUT_DIR / "stability_candidate_shortlist.csv",
        index=False,
    )

    print("\n=== STABILITY SHORTLIST ===\n")
    print(
        shortlist[
            [
                "candidate_id",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "bic",
                "shortlist_reasons",
            ]
        ].to_string(
            index=False,
            formatters={
                "silhouette": lambda value: f"{value:.4f}",
                "davies_bouldin": lambda value: f"{value:.4f}",
                "calinski_harabasz": lambda value: f"{value:.2f}",
                "bic": lambda value: "" if pd.isna(value) else f"{value:.2f}",
            },
        )
    )

    print(
        f"\nRunning {iterations} subsamples per candidate "
        f"with sample size {sample_size}/{len(X_df)}.\n"
    )

    all_distribution_frames = []
    all_clusterwise_frames = []
    all_consensus_frames = []
    summary_rows = []

    X_all = X_df.to_numpy(dtype=float)

    for candidate_number, candidate in shortlist.iterrows():
        candidate_id = candidate["candidate_id"]
        representation = candidate["representation"]
        algorithm = candidate["algorithm"]
        k = int(candidate["k_requested"])

        if candidate_id not in merged_labels.columns:
            raise KeyError(f"Candidate labels not found for {candidate_id}")

        full_reference = merged_labels[candidate_id].to_numpy(dtype=int)

        rng = np.random.default_rng(candidate_seed(base_seed, candidate_id))

        co_sampled = np.zeros(
            (len(X_df), len(X_df)),
            dtype=np.int32,
        )
        co_clustered = np.zeros(
            (len(X_df), len(X_df)),
            dtype=np.int32,
        )

        distribution_rows = []
        clusterwise_rows = []

        print(f"[{candidate_number + 1}/{len(shortlist)}] " f"{candidate_id}")

        for iteration in range(iterations):
            sampled_indices = np.sort(
                rng.choice(
                    len(X_df),
                    size=sample_size,
                    replace=False,
                )
            )

            X_subset = X_all[sampled_indices]
            reference_subset = full_reference[sampled_indices]

            transformed, dimensions = prepare_representation(
                X_subset,
                representation,
            )

            labels_subset = fit_candidate(
                transformed,
                algorithm,
                k,
                config,
                random_state=(base_seed + iteration + candidate_number * iterations),
            )

            ari = float(
                adjusted_rand_score(
                    reference_subset,
                    labels_subset,
                )
            )

            matched = matched_cluster_jaccards(
                reference_subset,
                labels_subset,
            )

            jaccard_values = [item["jaccard"] for item in matched]

            counts = pd.Series(labels_subset).value_counts()

            distribution_rows.append(
                {
                    "candidate_id": candidate_id,
                    "iteration": iteration + 1,
                    "ari": ari,
                    "mean_matched_jaccard": float(np.mean(jaccard_values)),
                    "minimum_matched_jaccard": float(np.min(jaccard_values)),
                    "minimum_resample_cluster_size": int(counts.min()),
                    "maximum_resample_cluster_size": int(counts.max()),
                    "representation_dimensions": int(dimensions),
                }
            )

            for item in matched:
                clusterwise_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "iteration": iteration + 1,
                        **item,
                    }
                )

            update_consensus(
                sampled_indices,
                labels_subset,
                co_sampled,
                co_clustered,
            )

            if (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
                print(f"  completed {iteration + 1}/{iterations}")

        distribution_df = pd.DataFrame(distribution_rows)
        clusterwise_df = pd.DataFrame(clusterwise_rows)

        all_distribution_frames.append(distribution_df)
        all_clusterwise_frames.append(clusterwise_df)

        all_consensus_frames.append(
            consensus_to_long(
                candidate_id,
                ids,
                co_sampled,
                co_clustered,
            )
        )

        summary = summarize_candidate(
            candidate_id,
            distribution_df,
            clusterwise_df,
        )

        summary.update(
            {
                "representation": representation,
                "algorithm": algorithm,
                "k_requested": k,
                "full_data_silhouette": float(candidate["silhouette"]),
                "full_data_davies_bouldin": float(candidate["davies_bouldin"]),
                "full_data_calinski_harabasz": float(candidate["calinski_harabasz"]),
                "full_data_minimum_cluster_size": int(
                    candidate["minimum_cluster_size"]
                ),
                "shortlist_reasons": candidate["shortlist_reasons"],
            }
        )

        summary_rows.append(summary)

    distributions = pd.concat(
        all_distribution_frames,
        ignore_index=True,
    )
    clusterwise = pd.concat(
        all_clusterwise_frames,
        ignore_index=True,
    )
    consensus = pd.concat(
        all_consensus_frames,
        ignore_index=True,
    )
    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        [
            "ari_median",
            "mean_matched_jaccard_median",
            "full_data_silhouette",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary.insert(
        0,
        "stability_rank",
        np.arange(1, len(summary) + 1),
    )

    distributions.to_csv(
        OUTPUT_DIR / "stability_distributions.csv",
        index=False,
    )
    clusterwise.to_csv(
        OUTPUT_DIR / "clusterwise_stability.csv",
        index=False,
    )
    consensus.to_csv(
        OUTPUT_DIR / "consensus_matrices_long.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "stability_summary.csv",
        index=False,
    )

    run_report = {
        "status": "STABILITY_COMPLETE",
        "iterations_per_candidate": iterations,
        "frozen_iterations": frozen_iterations,
        "used_frozen_iteration_count": iterations == frozen_iterations,
        "sample_fraction": sample_fraction,
        "sample_size": sample_size,
        "candidate_count": int(len(shortlist)),
        "shortlist_rule": (
            "Within each representation-algorithm group, take the union of "
            "best Silhouette, best Davies-Bouldin, best Calinski-Harabasz, "
            "and for GMM best BIC."
        ),
        "pca_rule_inside_resampling": (
            "Refit StandardScaler and PCA in each subsample; retain the "
            "minimum number of PCs reaching at least 85% cumulative variance."
        ),
        "selection_warning": (
            "Stability ranking is not yet the final model-selection decision."
        ),
    }

    (OUTPUT_DIR / "stability_run_report.json").write_text(
        json.dumps(
            run_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== STABILITY SUMMARY ===\n")
    print(
        summary[
            [
                "stability_rank",
                "candidate_id",
                "ari_median",
                "ari_p025",
                "ari_p975",
                "mean_matched_jaccard_median",
                "clusterwise_jaccard_min_mean",
                "minimum_resample_cluster_size_min",
            ]
        ].to_string(
            index=False,
            formatters={
                "ari_median": lambda value: f"{value:.4f}",
                "ari_p025": lambda value: f"{value:.4f}",
                "ari_p975": lambda value: f"{value:.4f}",
                "mean_matched_jaccard_median": lambda value: f"{value:.4f}",
                "clusterwise_jaccard_min_mean": lambda value: f"{value:.4f}",
            },
        )
    )

    print(
        "\nGATE STATUS: STABILITY COMPLETE. "
        "Do not select the final model until reviewing this summary."
    )


if __name__ == "__main__":
    main()
