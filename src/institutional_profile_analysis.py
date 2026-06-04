import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")

OUTPUT_GROUPS = os.path.join("data", "processed", "friction_group_profiles.csv")
OUTPUT_DIFFERENCES = os.path.join("data", "processed", "high_vs_low_differences.csv")
OUTPUT_SUMMARY = os.path.join("data", "processed", "institutional_profile_summary.csv")


TARGET = "AFS_theoretical"

FEATURES = [
    "device_constraint",
    "internet_stability_constraint",
    "digital_tool_variety_constraint",
    "recording_system_constraint",
    "admin_time_load_constraint",
    "administrative_disorganization_constraint",
    "digital_usage_constraint_score",
    "implementation_difficulty_constraint",
    "previous_implementation_constraint",
    "time_constraint_score",
    "staffing_constraint_score",
    "training_deficit_score",
    "resource_constraint_score",
    "system_change_resistance_constraint",
    "willingness_constraint_score",
    "perceived_utility_constraint",
    "pilot_openness_constraint",
    "ICI",
    "OCI",
    "OLI",
    "HCARI",
]


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")
    return pd.read_csv(filepath)


def validate_columns(df):
    required = [TARGET] + FEATURES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


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


def compute_group_profiles(df):
    group_profiles = (
        df.groupby("friction_group")[FEATURES + [TARGET]]
        .agg(["mean", "std", "count"])
        .round(4)
    )

    return group_profiles


def compute_high_vs_low_differences(df):
    low = df[df["friction_group"] == "low_friction"]
    high = df[df["friction_group"] == "high_friction"]

    results = []

    for feature in FEATURES:
        low_mean = low[feature].mean()
        high_mean = high[feature].mean()
        difference = high_mean - low_mean

        pooled_std = pd.concat([low[feature], high[feature]]).std()

        if pooled_std == 0 or pd.isna(pooled_std):
            effect_size = None
        else:
            effect_size = difference / pooled_std

        results.append(
            {
                "feature": feature,
                "low_friction_mean": low_mean,
                "high_friction_mean": high_mean,
                "high_minus_low_difference": difference,
                "standardized_difference": effect_size,
                "absolute_difference": abs(difference),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="absolute_difference", ascending=False)
        .reset_index(drop=True)
    )


def generate_summary(df, differences, low_cutoff, high_cutoff):
    top = differences.iloc[0]
    second = differences.iloc[1]
    third = differences.iloc[2]

    summary = pd.DataFrame(
        [
            {
                "n_institutions": len(df),
                "target_variable": TARGET,
                "low_friction_cutoff": low_cutoff,
                "high_friction_cutoff": high_cutoff,
                "low_group_n": len(df[df["friction_group"] == "low_friction"]),
                "medium_group_n": len(df[df["friction_group"] == "medium_friction"]),
                "high_group_n": len(df[df["friction_group"] == "high_friction"]),
                "strongest_distinguishing_feature": top["feature"],
                "strongest_difference": top["high_minus_low_difference"],
                "second_distinguishing_feature": second["feature"],
                "second_difference": second["high_minus_low_difference"],
                "third_distinguishing_feature": third["feature"],
                "third_difference": third["high_minus_low_difference"],
            }
        ]
    )

    return summary


def save_outputs(group_profiles, differences, summary):
    os.makedirs("data/processed", exist_ok=True)

    group_profiles.to_csv(OUTPUT_GROUPS)
    differences.to_csv(OUTPUT_DIFFERENCES, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)


def main():
    print("[PIPELINE] Running institutional_profile_analysis.py")

    df = load_data(INPUT_FILE)

    print(f"[INFO] Institutions loaded: {len(df)}")

    validate_columns(df)

    df, low_cutoff, high_cutoff = assign_friction_groups(df)

    group_profiles = compute_group_profiles(df)
    differences = compute_high_vs_low_differences(df)
    summary = generate_summary(df, differences, low_cutoff, high_cutoff)

    save_outputs(group_profiles, differences, summary)

    print("[SUCCESS] Institutional profile analysis completed.")

    print("\nHigh vs Low Friction Differences:")
    print(differences.head(12))

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
