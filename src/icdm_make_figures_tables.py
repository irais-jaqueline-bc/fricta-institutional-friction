from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PCA_SCORES_PATH = PROJECT_ROOT / "icdm" / "outputs" / "pca" / "pca_scores.csv"
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
STABILITY_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "stability" / "stability_summary.csv"
)
CONTRASTS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "interpretability" / "feature_contrasts.csv"
)
ALIGNMENT_COUNTS_PATH = (
    PROJECT_ROOT
    / "icdm"
    / "outputs"
    / "alignment"
    / "cluster_archetype_contingency_counts.csv"
)
ALIGNMENT_ROWS_PATH = (
    PROJECT_ROOT
    / "icdm"
    / "outputs"
    / "alignment"
    / "cluster_archetype_row_proportions.csv"
)
SELECTED_MODEL_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "selected_model.json"
)
SENSITIVITY_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "sensitivity" / "sensitivity_summary.csv"
)
LOFO_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "sensitivity" / "leave_one_feature_out.csv"
)
PCA_THRESHOLD_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "sensitivity" / "pca_threshold_sensitivity.csv"
)

FIGURE_DIR = PROJECT_ROOT / "icdm" / "figures"
TABLE_DIR = PROJECT_ROOT / "icdm" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_LABELS = {
    "device_constraint": "Device availability",
    "internet_stability_constraint": "Internet stability",
    "digital_tool_variety_constraint": "Digital-tool variety",
    "recording_system_constraint": "Recording system",
    "admin_time_load_constraint": "Administrative time load",
    "administrative_disorganization_constraint": "Administrative disorganization",
    "system_change_resistance_constraint": "Resistance to system change",
    "digital_usage_constraint_score": "Digital usage",
    "time_constraint_score": "Time constraints",
    "staffing_constraint_score": "Staffing constraints",
    "training_deficit_score": "Training deficit",
    "resource_constraint_score": "Resource constraints",
    "willingness_constraint_score": "Willingness to adopt",
}


PROFILE_NAMES = {
    1: "Organizational-Capacity Friction",
    2: "Infrastructure-Bottleneck",
}


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))


def wrap_labels(values: list[str], width: int = 25) -> list[str]:
    return ["\n".join(textwrap.wrap(str(value), width=width)) for value in values]


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_pca_clusters() -> None:
    pca = pd.read_csv(PCA_SCORES_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)

    id_column = [
        column
        for column in labels.columns
        if column
        not in {
            "cluster_label_zero_based",
            "cluster_id",
        }
    ][0]

    merged = pca.merge(
        labels[[id_column, "cluster_id"]],
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    for cluster_id, group in merged.groupby("cluster_id", sort=True):
        ax.scatter(
            group["PC1"],
            group["PC2"],
            label=(
                f"Cluster {cluster_id}: "
                f"{PROFILE_NAMES.get(cluster_id, '')} "
                f"(n={len(group)})"
            ),
            alpha=0.80,
            s=42,
        )

    ax.set_xlabel("PC1 (44.60% explained variance)")
    ax.set_ylabel("PC2 (21.94% explained variance)")
    ax.set_title("Empirical institutional profiles in PCA space")
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)

    save_figure(fig, "figure_pca_cluster_scatter")


def short_candidate_name(candidate_id: str) -> str:
    return (
        candidate_id.replace("R0_STANDARDIZED", "R0")
        .replace("R1_PCA_85", "R1")
        .replace("HAC_WARD", "HAC")
        .replace("GMM_DIAG", "GMM")
        .replace("__", " · ")
    )


def figure_stability() -> None:
    stability = pd.read_csv(STABILITY_PATH).copy()

    stability = stability.sort_values(
        "ari_median",
        ascending=True,
    ).reset_index(drop=True)

    y = np.arange(len(stability))

    lower = (stability["ari_median"] - stability["ari_p025"]).clip(lower=0)

    upper = (stability["ari_p975"] - stability["ari_median"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(8.3, 6.4))

    ax.errorbar(
        stability["ari_median"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        wrap_labels(
            [short_candidate_name(value) for value in stability["candidate_id"]],
            width=28,
        ),
        fontsize=8,
    )
    ax.set_xlabel("Adjusted Rand Index across 1,000 subsamples")
    ax.set_title("Resampling stability of shortlisted clustering solutions")
    ax.set_xlim(-0.15, 1.05)
    ax.axvline(0, linewidth=0.7)

    save_figure(fig, "figure_stability_intervals")


def figure_feature_contrasts() -> None:
    contrasts = pd.read_csv(CONTRASTS_PATH).copy()

    contrasts["feature_label"] = (
        contrasts["feature"].map(FEATURE_LABELS).fillna(contrasts["feature"])
    )

    contrasts = contrasts.sort_values(
        "cliffs_delta_cluster_2_vs_1",
        ascending=True,
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))

    ax.barh(
        np.arange(len(contrasts)),
        contrasts["cliffs_delta_cluster_2_vs_1"],
    )

    ax.set_yticks(np.arange(len(contrasts)))
    ax.set_yticklabels(
        wrap_labels(
            contrasts["feature_label"].tolist(),
            width=28,
        ),
        fontsize=8,
    )

    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Cliff's delta (Cluster 2 minus Cluster 1)")
    ax.set_title("Indicators distinguishing the two empirical profiles")

    save_figure(fig, "figure_feature_contrasts")


def figure_alignment() -> None:
    proportions = pd.read_csv(
        ALIGNMENT_ROWS_PATH,
        index_col=0,
    )

    proportions.index = [f"Cluster {int(index)}" for index in proportions.index]

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    proportions.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.72,
    )

    ax.set_xlabel("Empirical cluster")
    ax.set_ylabel("Proportion within cluster")
    ax.set_title("Structural alignment with FRICTA v1 archetypes")
    ax.set_ylim(0, 1)
    ax.legend(
        title="FRICTA v1 archetype",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    save_figure(fig, "figure_fricta_alignment")


def write_markdown_table(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> None:
    path.write_text(
        dataframe.to_markdown(index=index),
        encoding="utf-8",
    )


def table_model_selection() -> None:
    selected = json.loads(SELECTED_MODEL_PATH.read_text(encoding="utf-8"))

    stability = selected["stability"]
    full = selected["full_data_metrics"]

    table = pd.DataFrame(
        [
            {
                "Selected specification": selected["candidate_id"],
                "Representation": selected["representation"],
                "Algorithm": selected["algorithm"],
                "k": selected["k"],
                "Silhouette": full["silhouette"],
                "Davies-Bouldin": full["davies_bouldin"],
                "Calinski-Harabasz": full["calinski_harabasz"],
                "Median ARI": stability["ari_median"],
                "ARI 2.5%": stability["ari_p025"],
                "ARI 97.5%": stability["ari_p975"],
                "Weakest-cluster mean Jaccard": stability[
                    "clusterwise_jaccard_min_mean"
                ],
                "Cluster sizes": "; ".join(
                    f"{item['cluster_id']}:{item['size']}"
                    for item in selected["cluster_sizes"]
                ),
            }
        ]
    )

    table.to_csv(
        TABLE_DIR / "table_model_selection.csv",
        index=False,
    )

    write_markdown_table(
        table.round(4),
        TABLE_DIR / "table_model_selection.md",
    )


def table_cluster_profiles() -> None:
    contrasts = pd.read_csv(CONTRASTS_PATH).copy()

    table = contrasts[
        [
            "feature",
            "cluster_1_mean",
            "cluster_2_mean",
            "mean_difference_cluster_2_minus_1",
            "cliffs_delta_cluster_2_vs_1",
            "hedges_g_cluster_2_vs_1",
            "mann_whitney_q_bh",
        ]
    ].copy()

    table["feature"] = table["feature"].map(FEATURE_LABELS).fillna(table["feature"])

    table = table.rename(
        columns={
            "feature": "Indicator",
            "cluster_1_mean": "Organizational-Capacity mean",
            "cluster_2_mean": "Infrastructure-Bottleneck mean",
            "mean_difference_cluster_2_minus_1": "Difference C2-C1",
            "cliffs_delta_cluster_2_vs_1": "Cliff's delta",
            "hedges_g_cluster_2_vs_1": "Hedges' g",
            "mann_whitney_q_bh": "BH-adjusted q",
        }
    )

    table.to_csv(
        TABLE_DIR / "table_cluster_profiles.csv",
        index=False,
    )

    write_markdown_table(
        table.round(4),
        TABLE_DIR / "table_cluster_profiles.md",
    )


def table_alignment() -> None:
    counts = pd.read_csv(
        ALIGNMENT_COUNTS_PATH,
        index_col=0,
    )

    rows = []

    for cluster_id, row in counts.iterrows():
        total = int(row.sum())

        for archetype, count in row.items():
            rows.append(
                {
                    "Empirical cluster": int(cluster_id),
                    "Profile name": PROFILE_NAMES.get(
                        int(cluster_id),
                        "",
                    ),
                    "FRICTA v1 archetype": archetype,
                    "Count": int(count),
                    "Within-cluster proportion": (
                        float(count / total) if total > 0 else np.nan
                    ),
                }
            )

    table = pd.DataFrame(rows)

    table.to_csv(
        TABLE_DIR / "table_fricta_alignment.csv",
        index=False,
    )

    write_markdown_table(
        table.round(4),
        TABLE_DIR / "table_fricta_alignment.md",
    )


def table_sensitivity() -> None:
    core = pd.read_csv(SENSITIVITY_PATH)
    lofo = pd.read_csv(LOFO_PATH)
    thresholds = pd.read_csv(PCA_THRESHOLD_PATH)

    rows = []

    for row in core.itertuples(index=False):
        rows.append(
            {
                "Sensitivity family": "Core specification",
                "Analysis": row.analysis_id,
                "ARI vs final": row.adjusted_rand_index_vs_final,
                "Silhouette": row.silhouette,
                "Minimum cluster size": row.minimum_cluster_size,
                "Maximum cluster size": row.maximum_cluster_size,
            }
        )

    for row in lofo.itertuples(index=False):
        rows.append(
            {
                "Sensitivity family": "Leave-one-feature-out",
                "Analysis": f"Remove {row.removed_feature}",
                "ARI vs final": row.adjusted_rand_index_vs_final,
                "Silhouette": row.silhouette,
                "Minimum cluster size": row.minimum_cluster_size,
                "Maximum cluster size": row.maximum_cluster_size,
            }
        )

    for row in thresholds.itertuples(index=False):
        rows.append(
            {
                "Sensitivity family": "PCA threshold",
                "Analysis": f"{int(row.pca_threshold * 100)}% retained variance",
                "ARI vs final": row.adjusted_rand_index_vs_final,
                "Silhouette": row.silhouette,
                "Minimum cluster size": row.minimum_cluster_size,
                "Maximum cluster size": row.maximum_cluster_size,
            }
        )

    table = pd.DataFrame(rows)

    table.to_csv(
        TABLE_DIR / "table_sensitivity.csv",
        index=False,
    )

    write_markdown_table(
        table.round(4),
        TABLE_DIR / "table_sensitivity.md",
    )


def write_figure_manifest() -> None:
    manifest = pd.DataFrame(
        [
            {
                "figure": "figure_pca_cluster_scatter",
                "recommended_use": "Main text",
                "message": (
                    "Shows the selected two-profile partition "
                    "in the dominant PCA space."
                ),
            },
            {
                "figure": "figure_stability_intervals",
                "recommended_use": "Main text or appendix",
                "message": (
                    "Shows why HAC-Ward k=2 was selected over "
                    "less stable alternatives."
                ),
            },
            {
                "figure": "figure_feature_contrasts",
                "recommended_use": "Main text",
                "message": (
                    "Shows the substantive indicators separating "
                    "the two empirical profiles."
                ),
            },
            {
                "figure": "figure_fricta_alignment",
                "recommended_use": "Main text or appendix",
                "message": (
                    "Shows structural concordance with the original "
                    "FRICTA v1 archetypes."
                ),
            },
        ]
    )

    manifest.to_csv(
        TABLE_DIR / "figure_manifest.csv",
        index=False,
    )

    write_markdown_table(
        manifest,
        TABLE_DIR / "figure_manifest.md",
    )


def main() -> None:
    require_files(
        [
            PCA_SCORES_PATH,
            FINAL_LABELS_PATH,
            STABILITY_PATH,
            CONTRASTS_PATH,
            ALIGNMENT_COUNTS_PATH,
            ALIGNMENT_ROWS_PATH,
            SELECTED_MODEL_PATH,
            SENSITIVITY_PATH,
            LOFO_PATH,
            PCA_THRESHOLD_PATH,
        ]
    )

    figure_pca_clusters()
    figure_stability()
    figure_feature_contrasts()
    figure_alignment()

    table_model_selection()
    table_cluster_profiles()
    table_alignment()
    table_sensitivity()
    write_figure_manifest()

    print("\n=== FIGURES GENERATED ===\n")
    for stem in [
        "figure_pca_cluster_scatter",
        "figure_stability_intervals",
        "figure_feature_contrasts",
        "figure_fricta_alignment",
    ]:
        print(f"- icdm/figures/{stem}.png")
        print(f"- icdm/figures/{stem}.pdf")

    print("\n=== TABLES GENERATED ===\n")
    for stem in [
        "table_model_selection",
        "table_cluster_profiles",
        "table_fricta_alignment",
        "table_sensitivity",
        "figure_manifest",
    ]:
        print(f"- icdm/tables/{stem}.csv")
        print(f"- icdm/tables/{stem}.md")

    print(
        "\nGATE STATUS: FIGURES AND PAPER-READY TABLES COMPLETE. "
        "Next step is drafting the five-page manuscript."
    )


if __name__ == "__main__":
    main()
