from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
ARCHETYPES_PATH = PROJECT_ROOT / "data" / "processed" / "friction_archetypes.csv"
OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "alignment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERMUTATIONS = 10_000


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró la configuración: {CONFIG_PATH}")

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def detect_archetype_column(df: pd.DataFrame) -> str:
    candidates = [
        "friction_archetype",
        "archetype",
        "friction_profile",
        "profile",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        "No se encontró una columna de arquetipo. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def load_and_align(config: dict) -> tuple[pd.DataFrame, str]:
    for path in [FINAL_LABELS_PATH, ARCHETYPES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró un archivo requerido: {path}")

    id_column = config["id_column"]

    final_labels = pd.read_csv(FINAL_LABELS_PATH)
    archetypes = pd.read_csv(ARCHETYPES_PATH)

    if id_column not in final_labels.columns:
        raise KeyError(f"Falta {id_column} en final_cluster_labels.csv")

    if id_column not in archetypes.columns:
        raise KeyError(f"Falta {id_column} en friction_archetypes.csv")

    archetype_column = detect_archetype_column(archetypes)

    if final_labels[id_column].duplicated().any():
        raise ValueError("Hay IDs duplicados en final_cluster_labels.csv.")

    if archetypes[id_column].duplicated().any():
        raise ValueError("Hay IDs duplicados en friction_archetypes.csv.")

    required_cluster_columns = [
        id_column,
        "cluster_id",
        "cluster_label_zero_based",
    ]

    missing_cluster_columns = [
        column
        for column in required_cluster_columns
        if column not in final_labels.columns
    ]

    if missing_cluster_columns:
        raise KeyError(
            "Faltan columnas de cluster:\n- " + "\n- ".join(missing_cluster_columns)
        )

    aligned = final_labels[required_cluster_columns].merge(
        archetypes[[id_column, archetype_column]],
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) != len(final_labels):
        missing_ids = sorted(set(final_labels[id_column]) - set(aligned[id_column]))

        raise ValueError(
            "No todos los clusters tienen un arquetipo FRICTA v1. "
            f"IDs faltantes: {missing_ids}"
        )

    if aligned[archetype_column].isna().any():
        raise ValueError("Hay arquetipos FRICTA v1 faltantes.")

    return aligned, archetype_column


def contingency_tables(
    aligned: pd.DataFrame,
    archetype_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        aligned["cluster_id"],
        aligned[archetype_column],
        margins=False,
    )

    counts.index.name = "cluster_id"

    row_proportions = counts.div(
        counts.sum(axis=1),
        axis=0,
    )

    column_proportions = counts.div(
        counts.sum(axis=0),
        axis=1,
    )

    counts.to_csv(OUTPUT_DIR / "cluster_archetype_contingency_counts.csv")

    row_proportions.to_csv(OUTPUT_DIR / "cluster_archetype_row_proportions.csv")

    column_proportions.to_csv(OUTPUT_DIR / "archetype_cluster_column_proportions.csv")

    return counts, row_proportions, column_proportions


def association_metrics(
    aligned: pd.DataFrame,
    archetype_column: str,
    counts: pd.DataFrame,
) -> dict:
    cluster_labels = aligned["cluster_id"].to_numpy()
    archetype_labels = aligned[archetype_column].astype(str).to_numpy()

    ari = float(
        adjusted_rand_score(
            archetype_labels,
            cluster_labels,
        )
    )

    nmi = float(
        normalized_mutual_info_score(
            archetype_labels,
            cluster_labels,
        )
    )

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        archetype_labels,
        cluster_labels,
    )

    chi2, chi2_p, _, _ = chi2_contingency(counts.to_numpy())

    n = int(counts.to_numpy().sum())
    min_dimension = min(
        counts.shape[0] - 1,
        counts.shape[1] - 1,
    )

    cramers_v = (
        float(np.sqrt(chi2 / (n * min_dimension))) if min_dimension > 0 else np.nan
    )

    modal_concentration = float(counts.max(axis=1).sum() / n)

    return {
        "n_institutions": n,
        "n_empirical_clusters": int(aligned["cluster_id"].nunique()),
        "n_legacy_archetypes": int(aligned[archetype_column].nunique()),
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "chi_square": float(chi2),
        "chi_square_p_value": float(chi2_p),
        "cramers_v": cramers_v,
        "cluster_modal_archetype_concentration": (modal_concentration),
    }


def permutation_test(
    aligned: pd.DataFrame,
    archetype_column: str,
    observed_ari: float,
    observed_nmi: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    cluster_labels = aligned["cluster_id"].to_numpy()
    archetype_labels = aligned[archetype_column].astype(str).to_numpy()

    permuted_ari = np.empty(
        PERMUTATIONS,
        dtype=float,
    )

    permuted_nmi = np.empty(
        PERMUTATIONS,
        dtype=float,
    )

    for iteration in range(PERMUTATIONS):
        shuffled = rng.permutation(archetype_labels)

        permuted_ari[iteration] = adjusted_rand_score(
            shuffled,
            cluster_labels,
        )

        permuted_nmi[iteration] = normalized_mutual_info_score(
            shuffled,
            cluster_labels,
        )

    ari_p = float(
        (np.count_nonzero(permuted_ari >= observed_ari) + 1) / (PERMUTATIONS + 1)
    )

    nmi_p = float(
        (np.count_nonzero(permuted_nmi >= observed_nmi) + 1) / (PERMUTATIONS + 1)
    )

    distribution = pd.DataFrame(
        {
            "iteration": np.arange(
                1,
                PERMUTATIONS + 1,
            ),
            "permuted_ari": permuted_ari,
            "permuted_nmi": permuted_nmi,
        }
    )

    distribution.to_csv(
        OUTPUT_DIR / "alignment_permutation_distribution.csv",
        index=False,
    )

    return {
        "permutations": PERMUTATIONS,
        "random_seed": seed,
        "ari_empirical_p_value": ari_p,
        "nmi_empirical_p_value": nmi_p,
        "permuted_ari_mean": float(permuted_ari.mean()),
        "permuted_ari_p975": float(np.quantile(permuted_ari, 0.975)),
        "permuted_nmi_mean": float(permuted_nmi.mean()),
        "permuted_nmi_p975": float(np.quantile(permuted_nmi, 0.975)),
    }


def modal_archetype_outputs(
    aligned: pd.DataFrame,
    archetype_column: str,
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []

    for cluster_id, row in counts.iterrows():
        modal_archetype = row.idxmax()
        modal_count = int(row.max())
        cluster_size = int(row.sum())

        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "modal_fricta_v1_archetype": (modal_archetype),
                "modal_count": modal_count,
                "modal_proportion": float(modal_count / cluster_size),
                "number_of_archetypes_present": int((row > 0).sum()),
            }
        )

    modal_summary = pd.DataFrame(summary_rows)

    modal_map = dict(
        zip(
            modal_summary["cluster_id"],
            modal_summary["modal_fricta_v1_archetype"],
        )
    )

    cases = aligned.copy()

    cases["cluster_modal_fricta_v1_archetype"] = cases["cluster_id"].map(modal_map)

    cases["is_cluster_modal_archetype"] = (
        cases[archetype_column] == cases["cluster_modal_fricta_v1_archetype"]
    )

    non_modal = cases.loc[~cases["is_cluster_modal_archetype"]].copy()

    modal_summary.to_csv(
        OUTPUT_DIR / "cluster_modal_archetype_summary.csv",
        index=False,
    )

    non_modal.to_csv(
        OUTPUT_DIR / "non_modal_archetype_cases.csv",
        index=False,
    )

    aligned.to_csv(
        OUTPUT_DIR / "cluster_archetype_assignments.csv",
        index=False,
    )

    return modal_summary, non_modal


def main() -> None:
    config = load_config()
    aligned, archetype_column = load_and_align(config)

    counts, row_proportions, _ = contingency_tables(
        aligned,
        archetype_column,
    )

    metrics = association_metrics(
        aligned,
        archetype_column,
        counts,
    )

    permutation = permutation_test(
        aligned,
        archetype_column,
        metrics["adjusted_rand_index"],
        metrics["normalized_mutual_information"],
        seed=int(config["random_seed"]),
    )

    modal_summary, non_modal = modal_archetype_outputs(
        aligned,
        archetype_column,
        counts,
    )

    summary_table = pd.DataFrame(
        [
            {
                **metrics,
                **permutation,
                "interpretation": (
                    "Structural concordance only; " "FRICTA v1 is not ground truth."
                ),
            }
        ]
    )

    summary_table.to_csv(
        OUTPUT_DIR / "ari_nmi_summary.csv",
        index=False,
    )

    report = {
        "status": "THEORY_ALIGNMENT_COMPLETE",
        "empirical_model": ("R1_PCA_85__HAC_WARD__K2"),
        "legacy_reference": ("FRICTA v1 rule-based archetypes"),
        "archetype_column": archetype_column,
        "metrics": metrics,
        "permutation_test": permutation,
        "interpretation_boundary": (
            "These statistics describe association "
            "and structural concordance. They do not "
            "measure classification accuracy, external "
            "validity, or causal correctness."
        ),
        "legacy_profiles_used_for_model_selection": (False),
        "generated_files": [
            "icdm/outputs/alignment/ari_nmi_summary.csv",
            "icdm/outputs/alignment/cluster_archetype_contingency_counts.csv",
            "icdm/outputs/alignment/cluster_archetype_row_proportions.csv",
            "icdm/outputs/alignment/archetype_cluster_column_proportions.csv",
            "icdm/outputs/alignment/cluster_modal_archetype_summary.csv",
            "icdm/outputs/alignment/non_modal_archetype_cases.csv",
            "icdm/outputs/alignment/cluster_archetype_assignments.csv",
            "icdm/outputs/alignment/alignment_permutation_distribution.csv",
            "icdm/outputs/alignment/theory_alignment_report.json",
        ],
    }

    (OUTPUT_DIR / "theory_alignment_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== THEORY ALIGNMENT SUMMARY ===\n")
    print(f"Institutions: {len(aligned)}")
    print("Empirical clusters: " f"{metrics['n_empirical_clusters']}")
    print("FRICTA v1 archetypes: " f"{metrics['n_legacy_archetypes']}")
    print("Adjusted Rand Index: " f"{metrics['adjusted_rand_index']:.4f}")
    print(
        "Normalized Mutual Information: "
        f"{metrics['normalized_mutual_information']:.4f}"
    )
    print("Cramer's V: " f"{metrics['cramers_v']:.4f}")
    print("Permutation p-value (ARI): " f"{permutation['ari_empirical_p_value']:.5f}")
    print("Permutation p-value (NMI): " f"{permutation['nmi_empirical_p_value']:.5f}")

    print("\n=== CONTINGENCY COUNTS ===\n")
    print(counts.to_string())

    print("\n=== ROW PROPORTIONS BY EMPIRICAL CLUSTER ===\n")
    print(row_proportions.to_string(float_format=lambda value: (f"{value:.3f}")))

    print("\n=== MODAL ARCHETYPE BY CLUSTER ===\n")
    print(
        modal_summary.to_string(
            index=False,
            formatters={"modal_proportion": (lambda value: f"{value:.3f}")},
        )
    )

    print("\nNon-modal cases: " f"{len(non_modal)}")

    print(
        "\nGATE STATUS: THEORY ALIGNMENT COMPLETE. "
        "Next step is empirical profile interpretation."
    )


if __name__ == "__main__":
    main()
