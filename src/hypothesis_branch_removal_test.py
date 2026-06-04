import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")

OUTPUT_FILE = os.path.join("data", "processed", "hypothesis_branch_removal_test.csv")


BRANCHES = ["ICI", "OCI", "OLI", "HCARI"]

DIGITAL_MATURITY_FEATURES = [
    "device_constraint",
    "internet_stability_constraint",
    "digital_tool_variety_constraint",
    "recording_system_constraint",
    "digital_usage_constraint_score",
    "previous_implementation_constraint",
]

HUMAN_CAPACITY_FEATURES = [
    "time_constraint_score",
    "staffing_constraint_score",
    "training_deficit_score",
    "willingness_constraint_score",
]


def compute_effect_size(df, grouping_score, tested_score):
    low_cutoff = df[grouping_score].quantile(0.25)
    high_cutoff = df[grouping_score].quantile(0.75)

    low_group = df[df[grouping_score] <= low_cutoff][tested_score]
    high_group = df[df[grouping_score] >= high_cutoff][tested_score]

    low_mean = low_group.mean()
    high_mean = high_group.mean()
    difference = high_mean - low_mean

    pooled_std = pd.concat([low_group, high_group]).std()

    if pooled_std == 0 or pd.isna(pooled_std):
        standardized_difference = None
    else:
        standardized_difference = difference / pooled_std

    return {
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "raw_difference": difference,
        "standardized_difference": standardized_difference,
        "n_low": len(low_group),
        "n_high": len(high_group),
    }


def main():
    print("[PIPELINE] Running hypothesis_branch_removal_test.py")

    df = pd.read_csv(INPUT_FILE)

    df["digital_maturity_deficit"] = df[DIGITAL_MATURITY_FEATURES].mean(axis=1)
    df["human_capacity_deficit"] = df[HUMAN_CAPACITY_FEATURES].mean(axis=1)

    rows = []

    for removed_branch in BRANCHES:
        remaining_branches = [b for b in BRANCHES if b != removed_branch]

        grouping_score = f"AFS_without_{removed_branch}"
        df[grouping_score] = df[remaining_branches].mean(axis=1)

        digital_result = compute_effect_size(
            df, grouping_score, "digital_maturity_deficit"
        )

        human_result = compute_effect_size(df, grouping_score, "human_capacity_deficit")

        digital_effect = digital_result["standardized_difference"]
        human_effect = human_result["standardized_difference"]

        if digital_effect > human_effect:
            winner = "digital_maturity_deficit"
        elif human_effect > digital_effect:
            winner = "human_capacity_deficit"
        else:
            winner = "tie"

        rows.append(
            {
                "removed_branch": removed_branch,
                "remaining_branches": ",".join(remaining_branches),
                "grouping_score": grouping_score,
                "digital_effect_size": digital_effect,
                "human_effect_size": human_effect,
                "digital_minus_human_effect": digital_effect - human_effect,
                "digital_raw_difference": digital_result["raw_difference"],
                "human_raw_difference": human_result["raw_difference"],
                "n_low": digital_result["n_low"],
                "n_high": digital_result["n_high"],
                "winner": winner,
            }
        )

    result = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(result.to_string(index=False))
    print("\n[SUCCESS] Hypothesis branch-removal test completed.")


if __name__ == "__main__":
    main()
