import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")

OUTPUT_FACTORS = os.path.join("data", "processed", "institutional_factor_ranking.csv")

OUTPUT_BRANCHES = os.path.join("data", "processed", "institutional_branch_ranking.csv")

OUTPUT_SUMMARY = os.path.join("data", "processed", "institutional_analysis_summary.csv")


TARGET = "AFS_theoretical"


FACTOR_COLUMNS = [
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


BRANCH_COLUMNS = [
    "ICI",
    "OCI",
    "OLI",
    "HCARI",
]


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")
    return pd.read_csv(filepath)


def validate_required_columns(df):
    required = [TARGET] + BRANCH_COLUMNS

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def compute_factor_ranking(df):
    results = []

    for factor in FACTOR_COLUMNS:
        if factor not in df.columns:
            continue

        pearson = df[factor].corr(df[TARGET], method="pearson")

        spearman = df[factor].corr(df[TARGET], method="spearman")

        mean_value = df[factor].mean()
        std_value = df[factor].std()

        results.append(
            {
                "factor": factor,
                "pearson_correlation_with_AFS": pearson,
                "spearman_correlation_with_AFS": spearman,
                "absolute_pearson": abs(pearson),
                "mean_value": mean_value,
                "std_value": std_value,
            }
        )

    ranking = pd.DataFrame(results)

    ranking = ranking.sort_values(by="absolute_pearson", ascending=False).reset_index(
        drop=True
    )

    return ranking


def compute_branch_ranking(df):
    results = []

    for branch in BRANCH_COLUMNS:
        pearson = df[branch].corr(df[TARGET], method="pearson")

        spearman = df[branch].corr(df[TARGET], method="spearman")

        results.append(
            {
                "branch": branch,
                "pearson_correlation_with_AFS": pearson,
                "spearman_correlation_with_AFS": spearman,
                "absolute_pearson": abs(pearson),
                "mean_score": df[branch].mean(),
                "std_score": df[branch].std(),
            }
        )

    ranking = pd.DataFrame(results)

    ranking = ranking.sort_values(by="absolute_pearson", ascending=False).reset_index(
        drop=True
    )

    return ranking


def generate_summary(df, factor_ranking, branch_ranking):
    top_factor = factor_ranking.iloc[0]
    second_factor = factor_ranking.iloc[1]
    third_factor = factor_ranking.iloc[2]

    top_branch = branch_ranking.iloc[0]

    summary = pd.DataFrame(
        [
            {
                "n_institutions": len(df),
                "target_variable": TARGET,
                "mean_AFS_theoretical": df[TARGET].mean(),
                "std_AFS_theoretical": df[TARGET].std(),
                "top_factor": top_factor["factor"],
                "top_factor_pearson": top_factor["pearson_correlation_with_AFS"],
                "second_factor": second_factor["factor"],
                "second_factor_pearson": second_factor["pearson_correlation_with_AFS"],
                "third_factor": third_factor["factor"],
                "third_factor_pearson": third_factor["pearson_correlation_with_AFS"],
                "top_branch": top_branch["branch"],
                "top_branch_pearson": top_branch["pearson_correlation_with_AFS"],
            }
        ]
    )

    return summary


def save_outputs(factor_ranking, branch_ranking, summary):
    os.makedirs("data/processed", exist_ok=True)

    factor_ranking.to_csv(OUTPUT_FACTORS, index=False)

    branch_ranking.to_csv(OUTPUT_BRANCHES, index=False)

    summary.to_csv(OUTPUT_SUMMARY, index=False)


def main():
    print("[PIPELINE] Running institutional_analysis.py")

    df = load_data(INPUT_FILE)

    print(f"[INFO] Institutions loaded: {len(df)}")

    validate_required_columns(df)

    factor_ranking = compute_factor_ranking(df)

    branch_ranking = compute_branch_ranking(df)

    summary = generate_summary(df, factor_ranking, branch_ranking)

    save_outputs(factor_ranking, branch_ranking, summary)

    print("[SUCCESS] Institutional analysis completed.")

    print("\nTop institutional factors:")
    print(factor_ranking.head(10))

    print("\nBranch ranking:")
    print(branch_ranking)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
