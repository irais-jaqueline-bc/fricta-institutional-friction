import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")
OUTPUT_DIR = os.path.join("data", "processed")

FACTOR_OUTPUT = os.path.join(OUTPUT_DIR, "factor_correlations.csv")
BRANCH_OUTPUT = os.path.join(OUTPUT_DIR, "branch_correlations.csv")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "validation_summary.csv")


TARGET = "AFS_theoretical"


def load_scored_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    return pd.read_csv(filepath)


def compute_factor_correlations(df):
    factor_columns = [
        "device_constraint",
        "internet_stability_constraint",
        "digital_tool_variety_constraint",
        "recording_system_constraint",
        "admin_time_load_constraint",
        "administrative_disorganization_constraint",
        "implementation_difficulty_constraint",
        "system_change_resistance_constraint",
        "time_constraint_score",
        "staffing_constraint_score",
        "resource_constraint_score",
        "digital_usage_constraint_score",
        "training_deficit_score",
        "willingness_constraint_score",
        "perceived_utility_constraint",
        "pilot_openness_constraint",
        "previous_implementation_constraint",
    ]

    results = []

    for col in factor_columns:
        if col in df.columns:
            pearson = df[col].corr(df[TARGET], method="pearson")
            spearman = df[col].corr(df[TARGET], method="spearman")

            results.append(
                {
                    "factor": col,
                    "pearson_correlation_with_AFS": pearson,
                    "spearman_correlation_with_AFS": spearman,
                    "absolute_pearson": abs(pearson),
                }
            )

    return (
        pd.DataFrame(results)
        .sort_values(by="absolute_pearson", ascending=False)
        .reset_index(drop=True)
    )


def compute_branch_correlations(df):
    branch_columns = ["ICI", "OCI", "OLI", "HCARI"]

    results = []

    for col in branch_columns:
        pearson = df[col].corr(df[TARGET], method="pearson")
        spearman = df[col].corr(df[TARGET], method="spearman")

        results.append(
            {
                "branch": col,
                "pearson_correlation_with_AFS": pearson,
                "spearman_correlation_with_AFS": spearman,
                "absolute_pearson": abs(pearson),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="absolute_pearson", ascending=False)
        .reset_index(drop=True)
    )


def create_validation_summary(df, factor_corr, branch_corr):
    top_factor = factor_corr.iloc[0]
    top_branch = branch_corr.iloc[0]

    summary = {
        "n_institutions": len(df),
        "mean_AFS_theoretical": df[TARGET].mean(),
        "std_AFS_theoretical": df[TARGET].std(),
        "min_AFS_theoretical": df[TARGET].min(),
        "max_AFS_theoretical": df[TARGET].max(),
        "strongest_factor": top_factor["factor"],
        "strongest_factor_pearson": top_factor["pearson_correlation_with_AFS"],
        "strongest_branch": top_branch["branch"],
        "strongest_branch_pearson": top_branch["pearson_correlation_with_AFS"],
    }

    return pd.DataFrame([summary])


def main():
    print("[PIPELINE] Ejecutando módulo 'validation.py'...")

    df = load_scored_data(INPUT_FILE)
    print(f"[INFO] Filas cargadas: {len(df)}")

    factor_corr = compute_factor_correlations(df)
    branch_corr = compute_branch_correlations(df)
    summary = create_validation_summary(df, factor_corr, branch_corr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    factor_corr.to_csv(FACTOR_OUTPUT, index=False)
    branch_corr.to_csv(BRANCH_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("[ÉXITO] Validación completada.")
    print(f"[OUTPUT] {FACTOR_OUTPUT}")
    print(f"[OUTPUT] {BRANCH_OUTPUT}")
    print(f"[OUTPUT] {SUMMARY_OUTPUT}")

    print("\n[TOP FACTORS]")
    print(factor_corr.head(10))

    print("\n[BRANCH CORRELATIONS]")
    print(branch_corr)


if __name__ == "__main__":
    main()
