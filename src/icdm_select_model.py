from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
METRICS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "all_candidate_metrics.csv"
)
LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "all_candidate_labels.csv"
)
STABILITY_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "stability" / "stability_summary.csv"
)
CONSENSUS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "stability" / "consensus_matrices_long.csv"
)

CLUSTERING_DIR = PROJECT_ROOT / "icdm" / "outputs" / "clustering"
CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    for path in [
        CONFIG_PATH,
        METRICS_PATH,
        LABELS_PATH,
        STABILITY_PATH,
        CONSENSUS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS_PATH)
    labels = pd.read_csv(LABELS_PATH)
    stability = pd.read_csv(STABILITY_PATH)
    consensus = pd.read_csv(CONSENSUS_PATH)

    id_column = config["id_column"]

    if id_column not in labels.columns:
        raise KeyError(f"Missing {id_column} in all_candidate_labels.csv")

    if labels[id_column].duplicated().any():
        raise ValueError("Duplicate institution IDs in label table.")

    return config, metrics, labels, stability, consensus


def build_diagnostics(metrics: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    merged = stability.merge(
        metrics,
        on=["candidate_id", "representation", "algorithm", "k_requested"],
        how="left",
        validate="one_to_one",
        suffixes=("_stability", "_full"),
    )

    merged["resampling_min_cluster_ok"] = (
        merged["minimum_resample_cluster_size_min"] >= 5
    )
    merged["full_data_min_cluster_ok"] = merged["minimum_cluster_size"] >= 5
    merged["selection_eligible"] = (
        merged["resampling_min_cluster_ok"] & merged["full_data_min_cluster_ok"]
    )

    return merged


def pairwise_candidate_ari(
    labels: pd.DataFrame, candidate_ids: list[str]
) -> pd.DataFrame:
    rows = []

    for candidate_a, candidate_b in combinations(candidate_ids, 2):
        ari = adjusted_rand_score(
            labels[candidate_a].to_numpy(),
            labels[candidate_b].to_numpy(),
        )
        rows.append(
            {
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "adjusted_rand_index": float(ari),
                "partition_equivalent_at_0_95": bool(ari >= 0.95),
            }
        )

    return pd.DataFrame(rows)


def consensus_diagnostics(
    labels: pd.DataFrame,
    consensus: pd.DataFrame,
    id_column: str,
    candidate_ids: list[str],
) -> pd.DataFrame:
    rows = []
    labels_indexed = labels.set_index(id_column)

    for candidate_id in candidate_ids:
        candidate_consensus = consensus.loc[
            consensus["candidate_id"] == candidate_id
        ].copy()

        candidate_consensus = candidate_consensus.loc[
            candidate_consensus["institution_a"] != candidate_consensus["institution_b"]
        ].copy()

        label_map = labels_indexed[candidate_id].to_dict()

        candidate_consensus["cluster_a"] = candidate_consensus["institution_a"].map(
            label_map
        )
        candidate_consensus["cluster_b"] = candidate_consensus["institution_b"].map(
            label_map
        )
        candidate_consensus["same_full_data_cluster"] = (
            candidate_consensus["cluster_a"] == candidate_consensus["cluster_b"]
        )

        within = candidate_consensus.loc[
            candidate_consensus["same_full_data_cluster"], "consensus"
        ]
        between = candidate_consensus.loc[
            ~candidate_consensus["same_full_data_cluster"], "consensus"
        ]

        rows.append(
            {
                "candidate_id": candidate_id,
                "within_cluster_consensus_mean": float(within.mean()),
                "within_cluster_consensus_median": float(within.median()),
                "within_cluster_consensus_p025": float(within.quantile(0.025)),
                "between_cluster_consensus_mean": float(between.mean()),
                "between_cluster_consensus_median": float(between.median()),
                "between_cluster_consensus_p975": float(between.quantile(0.975)),
                "consensus_separation": float(within.mean() - between.mean()),
            }
        )

    return pd.DataFrame(rows)


def choose_model(
    diagnostics: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[str, dict]:
    eligible = diagnostics.loc[diagnostics["selection_eligible"]].copy()

    if eligible.empty:
        raise ValueError("No eligible candidates remain after stability checks.")

    best_median = eligible["ari_median"].max()

    median_tied = eligible.loc[
        np.isclose(
            eligible["ari_median"],
            best_median,
            atol=1e-12,
            rtol=0,
        )
    ].copy()

    decision = {
        "maximum_ari_median": float(best_median),
        "median_tied_candidates": median_tied["candidate_id"].tolist(),
        "partition_equivalence_threshold": 0.95,
    }

    if len(median_tied) == 1:
        selected = median_tied.iloc[0]
        decision["rule_used"] = "Unique highest median ARI."
        return selected["candidate_id"], decision

    tied_ids = median_tied["candidate_id"].tolist()
    pairwise = pairwise_candidate_ari(labels, tied_ids)

    all_equivalent = (
        not pairwise.empty and pairwise["partition_equivalent_at_0_95"].all()
    )

    decision["tied_pairwise_ari"] = pairwise.to_dict(orient="records")
    decision["tied_partitions_equivalent"] = bool(all_equivalent)

    if all_equivalent:
        selected = median_tied.sort_values(
            ["silhouette", "davies_bouldin", "calinski_harabasz", "k_requested"],
            ascending=[False, True, False, True],
        ).iloc[0]
        decision["rule_used"] = (
            "Top-stability candidates produced partition-equivalent labels "
            "(ARI >= 0.95). Selected the representation with stronger full-data "
            "internal separation; retained the alternative as robustness evidence."
        )
        return selected["candidate_id"], decision

    selected = median_tied.sort_values(
        [
            "ari_p025",
            "clusterwise_jaccard_min_mean",
            "silhouette",
            "k_requested",
        ],
        ascending=[False, False, False, True],
    ).iloc[0]

    decision["rule_used"] = (
        "Median ARI tie with non-equivalent partitions. Selected by stronger "
        "lower-tail stability, then weakest-cluster Jaccard, then Silhouette, "
        "then fewer clusters."
    )
    return selected["candidate_id"], decision


def write_rationale(
    selected_id: str,
    selected_row: pd.Series,
    decision: dict,
    consensus_summary: pd.DataFrame,
) -> None:
    selected_consensus = consensus_summary.loc[
        consensus_summary["candidate_id"] == selected_id
    ].iloc[0]

    tied_candidates = ", ".join(decision["median_tied_candidates"])

    text = f"""# Final Model-Selection Rationale

## Selected model

`{selected_id}`

- Representation: `{selected_row['representation']}`
- Algorithm: `{selected_row['algorithm']}`
- k: `{int(selected_row['k_requested'])}`
- Full-data Silhouette: `{selected_row['silhouette']:.4f}`
- Full-data Davies-Bouldin: `{selected_row['davies_bouldin']:.4f}`
- Full-data Calinski-Harabasz: `{selected_row['calinski_harabasz']:.2f}`
- Median resampling ARI: `{selected_row['ari_median']:.4f}`
- 95% empirical ARI interval: `[{selected_row['ari_p025']:.4f}, {selected_row['ari_p975']:.4f}]`
- Weakest cluster mean matched Jaccard: `{selected_row['clusterwise_jaccard_min_mean']:.4f}`
- Minimum cluster size observed during resampling: `{int(selected_row['minimum_resample_cluster_size_min'])}`
- Mean within-cluster consensus: `{selected_consensus['within_cluster_consensus_mean']:.4f}`
- Mean between-cluster consensus: `{selected_consensus['between_cluster_consensus_mean']:.4f}`

## Decision rule

{decision['rule_used']}

The candidates tied at the maximum median ARI were: {tied_candidates}.

## Interpretation boundary

This result supports a stable empirical partition in the current sample. It does not
establish causal classes, universal institutional types, or external ground truth.
FRICTA v1 archetypes will be compared only after model selection and were not used
to choose the clustering solution.

## Rejected solution families

Candidates producing clusters smaller than five institutions during resampling are
not retained for the primary solution. Lower-stability alternatives may remain as
sensitivity analyses but are not the primary empirical partition.
"""

    (CLUSTERING_DIR / "model_selection_rationale.md").write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    config, metrics, labels, stability, consensus = load_inputs()
    id_column = config["id_column"]

    diagnostics = build_diagnostics(metrics, stability)
    shortlisted_ids = diagnostics["candidate_id"].tolist()

    pairwise_all = pairwise_candidate_ari(labels, shortlisted_ids)
    consensus_summary = consensus_diagnostics(
        labels,
        consensus,
        id_column,
        shortlisted_ids,
    )

    selected_id, decision = choose_model(diagnostics, labels)

    selected_row = diagnostics.loc[diagnostics["candidate_id"] == selected_id].iloc[0]

    top_ids = diagnostics.sort_values("stability_rank").head(5)["candidate_id"].tolist()
    pairwise_top = pairwise_candidate_ari(labels, top_ids)

    final_labels = labels[[id_column, selected_id]].copy()
    final_labels = final_labels.rename(
        columns={selected_id: "cluster_label_zero_based"}
    )
    final_labels["cluster_id"] = final_labels["cluster_label_zero_based"] + 1

    cluster_sizes = (
        final_labels["cluster_id"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster_id")
        .reset_index(name="size")
    )

    selected_model = {
        "status": "FINAL_MODEL_SELECTED",
        "candidate_id": selected_id,
        "representation": selected_row["representation"],
        "algorithm": selected_row["algorithm"],
        "k": int(selected_row["k_requested"]),
        "full_data_metrics": {
            "silhouette": float(selected_row["silhouette"]),
            "davies_bouldin": float(selected_row["davies_bouldin"]),
            "calinski_harabasz": float(selected_row["calinski_harabasz"]),
            "minimum_cluster_size": int(selected_row["minimum_cluster_size"]),
            "maximum_cluster_size": int(selected_row["maximum_cluster_size"]),
        },
        "stability": {
            "ari_median": float(selected_row["ari_median"]),
            "ari_p025": float(selected_row["ari_p025"]),
            "ari_p975": float(selected_row["ari_p975"]),
            "clusterwise_jaccard_min_mean": float(
                selected_row["clusterwise_jaccard_min_mean"]
            ),
            "minimum_resample_cluster_size_min": int(
                selected_row["minimum_resample_cluster_size_min"]
            ),
        },
        "cluster_sizes": cluster_sizes.to_dict(orient="records"),
        "decision": decision,
        "legacy_profiles_used_for_selection": False,
    }

    diagnostics.to_csv(
        CLUSTERING_DIR / "model_selection_diagnostics.csv",
        index=False,
    )
    pairwise_all.to_csv(
        CLUSTERING_DIR / "shortlist_pairwise_ari.csv",
        index=False,
    )
    pairwise_top.to_csv(
        CLUSTERING_DIR / "top5_pairwise_ari.csv",
        index=False,
    )
    consensus_summary.to_csv(
        CLUSTERING_DIR / "consensus_diagnostics.csv",
        index=False,
    )
    final_labels.to_csv(
        CLUSTERING_DIR / "final_cluster_labels.csv",
        index=False,
    )
    cluster_sizes.to_csv(
        CLUSTERING_DIR / "final_cluster_sizes.csv",
        index=False,
    )
    (CLUSTERING_DIR / "selected_model.json").write_text(
        json.dumps(selected_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_rationale(
        selected_id,
        selected_row,
        decision,
        consensus_summary,
    )

    print("\n=== MODEL SELECTION SUMMARY ===\n")
    print(f"Selected candidate: {selected_id}")
    print(f"Representation: {selected_row['representation']}")
    print(f"Algorithm: {selected_row['algorithm']}")
    print(f"k: {int(selected_row['k_requested'])}")
    print(f"Silhouette: {selected_row['silhouette']:.4f}")
    print(
        "Median ARI [2.5%, 97.5%]: "
        f"{selected_row['ari_median']:.4f} "
        f"[{selected_row['ari_p025']:.4f}, "
        f"{selected_row['ari_p975']:.4f}]"
    )
    print(
        "Weakest-cluster mean Jaccard: "
        f"{selected_row['clusterwise_jaccard_min_mean']:.4f}"
    )

    print("\n=== FINAL CLUSTER SIZES ===\n")
    print(cluster_sizes.to_string(index=False))

    print("\n=== TOP-CANDIDATE PAIRWISE ARI ===\n")
    print(
        pairwise_top.to_string(
            index=False,
            formatters={"adjusted_rand_index": lambda value: f"{value:.4f}"},
        )
    )

    print("\n=== DECISION RULE ===\n")
    print(decision["rule_used"])

    print(
        "\nGATE STATUS: FINAL EMPIRICAL MODEL SELECTED. "
        "Next step is theory alignment and profile interpretation."
    )


if __name__ == "__main__":
    main()
