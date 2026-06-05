import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")

OUTPUT_COMPONENTS = os.path.join(
    "data", "processed", "ici_decomposition_components.csv"
)

OUTPUT_SUMMARY = os.path.join("data", "processed", "ici_decomposition_summary.csv")


TARGET = "AFS_theoretical"

ICI_COMPONENTS = [
    "device_constraint",
    "internet_stability_constraint",
    "digital_tool_variety_constraint",
]

COMPARISON_FEATURES = [
    "human_capacity_deficit",
    "digital_maturity_deficit",
]


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")
    return pd.read_csv(filepath)


def assign_friction_groups(df):
    df = df.copy()

    low_cutoff = df[TARGET].quantile(0.25)
    high_cutoff = df[TARGET].quantile(0.75)

    def classify(value):
        if value <= low_cutoff:
            return "low_friction"
        elif value >= high_cutoff:
            return "high_friction"
        else:
            return "medium_friction"

    df["friction_group"] = df[TARGET].apply(classify)

    return df, low_cutoff, high_cutoff


def compute_effect_size(df, feature):
    low = df[df["friction_group"] == "low_friction"][feature]
    high = df[df["friction_group"] == "high_friction"][feature]

    low_mean = low.mean()
    high_mean = high.mean()
    difference = high_mean - low_mean

    pooled_std = pd.concat([low, high]).std()

    if pooled_std == 0 or pd.isna(pooled_std):
        standardized_difference = None
    else:
        standardized_difference = difference / pooled_std

    return {
        "feature": feature,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "high_minus_low_difference": difference,
        "standardized_difference": standardized_difference,
    }


def main():
    print("[PIPELINE] Running ici_decomposition_analysis.py")

    df = load_data(INPUT_FILE)

    print(f"[INFO] Institutions loaded: {len(df)}")

    df["digital_maturity_deficit"] = df[
        [
            "device_constraint",
            "internet_stability_constraint",
            "digital_tool_variety_constraint",
            "recording_system_constraint",
            "digital_usage_constraint_score",
            "previous_implementation_constraint",
        ]
    ].mean(axis=1)

    df["human_capacity_deficit"] = df[
        [
            "time_constraint_score",
            "staffing_constraint_score",
            "training_deficit_score",
            "willingness_constraint_score",
        ]
    ].mean(axis=1)

    df, low_cutoff, high_cutoff = assign_friction_groups(df)

    results = []

    for feature in ICI_COMPONENTS + COMPARISON_FEATURES:
        results.append(compute_effect_size(df, feature))

    components = (
        pd.DataFrame(results)
        .sort_values(by="standardized_difference", ascending=False)
        .reset_index(drop=True)
    )

    top_component = components.iloc[0]

    summary = pd.DataFrame(
        [
            {
                "n_institutions": len(df),
                "low_friction_cutoff": low_cutoff,
                "high_friction_cutoff": high_cutoff,
                "strongest_ici_or_comparison_feature": top_component["feature"],
                "strongest_standardized_difference": top_component[
                    "standardized_difference"
                ],
                "strongest_raw_difference": top_component["high_minus_low_difference"],
            }
        ]
    )

    os.makedirs("data/processed", exist_ok=True)

    components.to_csv(OUTPUT_COMPONENTS, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("[SUCCESS] ICI decomposition completed.")

    print("\nICI Decomposition Components:")
    print(components.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
