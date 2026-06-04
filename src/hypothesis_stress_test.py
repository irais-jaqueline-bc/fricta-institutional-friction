import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")

OUTPUT_COMPOSITES = os.path.join(
    "data", "processed", "hypothesis_stress_test_scores.csv"
)

OUTPUT_SUMMARY = os.path.join("data", "processed", "hypothesis_stress_test_summary.csv")

OUTPUT_THRESHOLD_ROBUSTNESS = os.path.join(
    "data", "processed", "threshold_robustness_summary.csv"
)


TARGET = "AFS_theoretical"


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


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")
    return pd.read_csv(filepath)


def validate_columns(df):
    required = [TARGET] + DIGITAL_MATURITY_FEATURES + HUMAN_CAPACITY_FEATURES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def compute_composites(df):
    df = df.copy()

    df["digital_maturity_deficit"] = df[DIGITAL_MATURITY_FEATURES].mean(axis=1)
    df["human_capacity_deficit"] = df[HUMAN_CAPACITY_FEATURES].mean(axis=1)

    return df


def assign_friction_groups(df, low_q, high_q):
    df = df.copy()

    low_cutoff = df[TARGET].quantile(low_q)
    high_cutoff = df[TARGET].quantile(high_q)

    def classify(value):
        if value <= low_cutoff:
            return "low_friction"
        elif value >= high_cutoff:
            return "high_friction"
        else:
            return "medium_friction"

    df["friction_group"] = df[TARGET].apply(classify)

    return df, low_cutoff, high_cutoff


def compute_group_difference(df, score_col):
    low = df[df["friction_group"] == "low_friction"][score_col]
    high = df[df["friction_group"] == "high_friction"][score_col]

    low_mean = low.mean()
    high_mean = high.mean()
    difference = high_mean - low_mean

    pooled_std = pd.concat([low, high]).std()

    if pooled_std == 0 or pd.isna(pooled_std):
        standardized_difference = None
    else:
        standardized_difference = difference / pooled_std

    return {
        "score": score_col,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "high_minus_low_difference": difference,
        "standardized_difference": standardized_difference,
    }


def run_main_stress_test(df):
    grouped, low_cutoff, high_cutoff = assign_friction_groups(df, 0.25, 0.75)

    digital_result = compute_group_difference(grouped, "digital_maturity_deficit")

    human_result = compute_group_difference(grouped, "human_capacity_deficit")

    results = pd.DataFrame([digital_result, human_result])

    digital_effect = digital_result["standardized_difference"]
    human_effect = human_result["standardized_difference"]

    if digital_effect > human_effect:
        interpretation = (
            "Digital maturity deficits distinguish high-friction institutions "
            "more strongly than human-capacity deficits."
        )
    elif human_effect > digital_effect:
        interpretation = (
            "Human-capacity deficits distinguish high-friction institutions "
            "more strongly than digital maturity deficits."
        )
    else:
        interpretation = (
            "Digital maturity and human-capacity deficits show similar separation."
        )

    results["low_friction_cutoff"] = low_cutoff
    results["high_friction_cutoff"] = high_cutoff
    results["interpretation"] = interpretation

    return results


def run_threshold_robustness(df):
    thresholds = [
        (0.20, 0.80),
        (0.25, 0.75),
        (0.30, 0.70),
    ]

    rows = []

    for low_q, high_q in thresholds:
        grouped, low_cutoff, high_cutoff = assign_friction_groups(df, low_q, high_q)

        digital_result = compute_group_difference(grouped, "digital_maturity_deficit")

        human_result = compute_group_difference(grouped, "human_capacity_deficit")

        rows.append(
            {
                "low_quantile": low_q,
                "high_quantile": high_q,
                "low_cutoff": low_cutoff,
                "high_cutoff": high_cutoff,
                "digital_standardized_difference": digital_result[
                    "standardized_difference"
                ],
                "human_standardized_difference": human_result[
                    "standardized_difference"
                ],
                "digital_minus_human_effect": (
                    digital_result["standardized_difference"]
                    - human_result["standardized_difference"]
                ),
                "digital_raw_difference": digital_result["high_minus_low_difference"],
                "human_raw_difference": human_result["high_minus_low_difference"],
                "n_low": len(grouped[grouped["friction_group"] == "low_friction"]),
                "n_high": len(grouped[grouped["friction_group"] == "high_friction"]),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(df, main_summary, threshold_summary):
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_COMPOSITES, index=False)
    main_summary.to_csv(OUTPUT_SUMMARY, index=False)
    threshold_summary.to_csv(OUTPUT_THRESHOLD_ROBUSTNESS, index=False)


def main():
    print("[PIPELINE] Running hypothesis_stress_test.py")

    df = load_data(INPUT_FILE)

    print(f"[INFO] Institutions loaded: {len(df)}")

    validate_columns(df)

    df = compute_composites(df)

    main_summary = run_main_stress_test(df)

    threshold_summary = run_threshold_robustness(df)

    save_outputs(df, main_summary, threshold_summary)

    print("[SUCCESS] Hypothesis stress test completed.")

    print("\nMain Stress Test:")
    print(main_summary)

    print("\nThreshold Robustness:")
    print(threshold_summary)


if __name__ == "__main__":
    main()
