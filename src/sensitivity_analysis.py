import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")

OUTPUT_SCORES = os.path.join("data", "processed", "sensitivity_scores.csv")

OUTPUT_SUMMARY = os.path.join("data", "processed", "sensitivity_summary.csv")

OUTPUT_TOP10 = os.path.join("data", "processed", "sensitivity_top10_overlap.csv")


BRANCHES = ["ICI", "OCI", "OLI", "HCARI"]


WEIGHT_SCHEMES = {
    "AFS_theoretical": {
        "ICI": 0.25,
        "OCI": 0.30,
        "OLI": 0.20,
        "HCARI": 0.25,
    },
    "AFS_equal_weights": {
        "ICI": 0.25,
        "OCI": 0.25,
        "OLI": 0.25,
        "HCARI": 0.25,
    },
    "AFS_organizational_heavy": {
        "ICI": 0.20,
        "OCI": 0.35,
        "OLI": 0.20,
        "HCARI": 0.25,
    },
    "AFS_human_heavy": {
        "ICI": 0.20,
        "OCI": 0.25,
        "OLI": 0.20,
        "HCARI": 0.35,
    },
    "AFS_infrastructure_heavy": {
        "ICI": 0.35,
        "OCI": 0.25,
        "OLI": 0.20,
        "HCARI": 0.20,
    },
}


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")
    return pd.read_csv(filepath)


def validate_required_columns(df):
    missing = [col for col in BRANCHES if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas necesarias: {missing}")


def compute_weighted_scores(df):
    df = df.copy()

    for score_name, weights in WEIGHT_SCHEMES.items():
        df[score_name] = (
            weights["ICI"] * df["ICI"]
            + weights["OCI"] * df["OCI"]
            + weights["OLI"] * df["OLI"]
            + weights["HCARI"] * df["HCARI"]
        )

    return df


def compute_sensitivity_summary(df):
    baseline = "AFS_theoretical"

    alternative_scores = [score for score in WEIGHT_SCHEMES.keys() if score != baseline]

    results = []

    for score in alternative_scores:
        pearson_corr = df[baseline].corr(df[score], method="pearson")
        spearman_corr = df[baseline].corr(df[score], method="spearman")

        mean_absolute_difference = (df[baseline] - df[score]).abs().mean()

        max_absolute_difference = (df[baseline] - df[score]).abs().max()

        results.append(
            {
                "baseline_score": baseline,
                "alternative_score": score,
                "pearson_correlation": pearson_corr,
                "spearman_correlation": spearman_corr,
                "mean_absolute_difference": mean_absolute_difference,
                "max_absolute_difference": max_absolute_difference,
            }
        )

    return pd.DataFrame(results)


def compute_top10_overlap(df):
    baseline = "AFS_theoretical"

    baseline_top10 = set(df.sort_values(by=baseline, ascending=False).head(10).index)

    results = []

    for score in WEIGHT_SCHEMES.keys():
        if score == baseline:
            continue

        alternative_top10 = set(
            df.sort_values(by=score, ascending=False).head(10).index
        )

        overlap_count = len(baseline_top10.intersection(alternative_top10))

        overlap_ratio = overlap_count / 10

        results.append(
            {
                "baseline_score": baseline,
                "alternative_score": score,
                "top10_overlap_count": overlap_count,
                "top10_overlap_ratio": overlap_ratio,
            }
        )

    return pd.DataFrame(results)


def interpret_results(summary_df, top10_df):
    min_corr = summary_df["pearson_correlation"].min()
    min_overlap = top10_df["top10_overlap_ratio"].min()

    if min_corr >= 0.95 and min_overlap >= 0.80:
        interpretation = "The FRICTA score appears highly robust under alternative weighting schemes."
    elif min_corr >= 0.90 and min_overlap >= 0.60:
        interpretation = "The FRICTA score appears moderately robust under alternative weighting schemes."
    else:
        interpretation = "The FRICTA score shows meaningful sensitivity to alternative weighting schemes."

    return interpretation


def save_outputs(df_scores, summary_df, top10_df):
    os.makedirs("data/processed", exist_ok=True)

    df_scores.to_csv(OUTPUT_SCORES, index=False)

    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    top10_df.to_csv(OUTPUT_TOP10, index=False)


def main():
    print("[PIPELINE] Running sensitivity_analysis.py")

    df = load_data(INPUT_FILE)

    print(f"[INFO] Institutions loaded: {len(df)}")

    validate_required_columns(df)

    df_scores = compute_weighted_scores(df)

    summary_df = compute_sensitivity_summary(df_scores)

    top10_df = compute_top10_overlap(df_scores)

    interpretation = interpret_results(summary_df, top10_df)

    save_outputs(df_scores, summary_df, top10_df)

    print("[SUCCESS] Sensitivity analysis completed.")

    print("\nSensitivity Summary:")
    print(summary_df)

    print("\nTop 10 Overlap:")
    print(top10_df)

    print("\nInterpretation:")
    print(interpretation)


if __name__ == "__main__":
    main()
