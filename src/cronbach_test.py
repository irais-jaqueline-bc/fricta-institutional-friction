import pandas as pd

df = pd.read_csv("processed/data/fricta_scored.csv")


def cronbach_alpha(data):
    data = data.dropna()

    k = data.shape[1]

    item_variances = data.var(axis=0, ddof=1)

    total_variance = data.sum(axis=1).var(ddof=1)

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

    return alpha


branches = {
    "ICI": [
        "device_constraint",
        "internet_stability_constraint",
        "digital_tool_variety_constraint",
        "resource_constraint_score",
    ],
    "OCI": [
        "administrative_disorganization_constraint",
        "implementation_difficulty_constraint",
        "system_change_resistance_constraint",
    ],
    "OLI": [
        "admin_time_load_constraint",
        "time_constraint_score",
        "staffing_constraint_score",
    ],
    "HCARI": [
        "training_deficit_score",
        "digital_usage_constraint_score",
        "willingness_constraint_score",
    ],
}

for branch, cols in branches.items():
    alpha = cronbach_alpha(df[cols])
    print(branch, round(alpha, 3))
