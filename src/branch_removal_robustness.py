import os
import pandas as pd

INPUT_FILE = "data/processed/fricta_scored.csv"

OUTPUT_FILE = "data/processed/branch_removal_robustness.csv"

BRANCHES = ["ICI", "OCI", "OLI", "HCARI"]


def effect_size(df, score_col):
    low = df[score_col].quantile(0.25)
    high = df[score_col].quantile(0.75)

    low_group = df[df[score_col] <= low][score_col]
    high_group = df[df[score_col] >= high][score_col]

    diff = high_group.mean() - low_group.mean()

    pooled_std = pd.concat([low_group, high_group]).std()

    return diff / pooled_std


def main():

    print("[PIPELINE] Running branch_removal_robustness.py")

    df = pd.read_csv(INPUT_FILE)

    rows = []

    for removed in BRANCHES:

        remaining = [b for b in BRANCHES if b != removed]

        new_score = df[remaining].mean(axis=1)

        temp = pd.DataFrame()

        temp["score"] = new_score

        robustness = effect_size(temp, "score")

        rows.append(
            {
                "removed_branch": removed,
                "remaining_branches": ",".join(remaining),
                "effect_size": robustness,
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(OUTPUT_FILE, index=False)

    print(result)

    print("\n[SUCCESS] Branch removal robustness completed.")


if __name__ == "__main__":
    main()
