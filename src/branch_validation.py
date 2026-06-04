import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")

OUTPUT_MATRIX = os.path.join("data", "processed", "branch_correlation_matrix.csv")

OUTPUT_PAIRS = os.path.join("data", "processed", "branch_pairwise_correlations.csv")

OUTPUT_SUMMARY = os.path.join("data", "processed", "branch_validation_summary.csv")


BRANCHES = ["ICI", "OCI", "OLI", "HCARI"]


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")

    return pd.read_csv(filepath)


def compute_branch_correlation_matrix(df):
    return df[BRANCHES].corr(method="pearson")


def compute_pairwise_correlations(df):
    results = []

    for i in range(len(BRANCHES)):
        for j in range(i + 1, len(BRANCHES)):

            branch_a = BRANCHES[i]
            branch_b = BRANCHES[j]

            corr = df[branch_a].corr(df[branch_b], method="pearson")

            results.append(
                {
                    "branch_a": branch_a,
                    "branch_b": branch_b,
                    "pearson_correlation": corr,
                    "absolute_correlation": abs(corr),
                }
            )

    results = pd.DataFrame(results)

    results = results.sort_values(by="absolute_correlation", ascending=False)

    return results


def generate_summary(pairwise_df):

    max_corr = pairwise_df.iloc[0]

    if max_corr["absolute_correlation"] >= 0.90:
        interpretation = "Potential branch redundancy detected."

    elif max_corr["absolute_correlation"] >= 0.70:
        interpretation = (
            "Moderate-to-high association. Branch distinction should be examined."
        )

    else:
        interpretation = "Branches appear reasonably differentiated."

    summary = pd.DataFrame(
        [
            {
                "highest_pair": f"{max_corr['branch_a']} vs {max_corr['branch_b']}",
                "highest_correlation": max_corr["pearson_correlation"],
                "interpretation": interpretation,
            }
        ]
    )

    return summary


def save_outputs(matrix_df, pairwise_df, summary_df):

    matrix_df.to_csv(OUTPUT_MATRIX)

    pairwise_df.to_csv(OUTPUT_PAIRS, index=False)

    summary_df.to_csv(OUTPUT_SUMMARY, index=False)


def main():

    print("[PIPELINE] Running branch_validation.py")

    df = load_data(INPUT_FILE)

    print(f"[INFO] Institutions loaded: {len(df)}")

    matrix_df = compute_branch_correlation_matrix(df)

    pairwise_df = compute_pairwise_correlations(df)

    summary_df = generate_summary(pairwise_df)

    save_outputs(matrix_df, pairwise_df, summary_df)

    print("[SUCCESS] Branch validation completed.")

    print("\nCorrelation Matrix:")
    print(matrix_df)

    print("\nPairwise Correlations:")
    print(pairwise_df)

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()
