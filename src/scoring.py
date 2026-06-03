import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "clean_data.csv")
OUTPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")


def load_clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    return pd.read_csv(filepath)


def normalize_direct(series, min_value, max_value):
    return (series - min_value) / (max_value - min_value)


def normalize_reverse(series, min_value, max_value):
    return (max_value - series) / (max_value - min_value)


def create_tool_variety(df):
    tools = df["current_digital_tools"].fillna("")

    df["uses_excel"] = tools.str.contains("Excel", case=False, regex=False).astype(int)
    df["uses_whatsapp"] = tools.str.contains(
        "WhatsApp", case=False, regex=False
    ).astype(int)
    df["uses_google_workspace"] = tools.str.contains(
        "Google", case=False, regex=False
    ).astype(int)
    df["uses_specialized_software"] = tools.str.contains(
        "Software", case=False, regex=False
    ).astype(int)
    df["uses_no_tool"] = tools.str.contains("Ninguna", case=False, regex=False).astype(
        int
    )

    df["digital_tool_variety"] = (
        df["uses_excel"]
        + df["uses_whatsapp"]
        + df["uses_google_workspace"]
        + df["uses_specialized_software"]
    )

    return df


def compute_normalized_constraints(df):
    df = df.copy()

    # Infrastructure: higher raw value = lower friction
    df["device_constraint"] = normalize_reverse(df["available_devices"], 0, 3)
    df["internet_stability_constraint"] = normalize_reverse(
        df["internet_constraint"], 1, 5
    )
    df["digital_tool_variety_constraint"] = normalize_reverse(
        df["digital_tool_variety"], 0, 4
    )
    df["recording_system_constraint"] = normalize_reverse(
        df["information_recording_method"], 1, 4
    )

    # Organizational / administrative
    df["admin_time_load_constraint"] = normalize_direct(
        df["admin_time_load_norm"], 1, 4
    )
    df["administrative_disorganization_constraint"] = normalize_reverse(
        df["admin_disorganization"], 1, 5
    )
    df["implementation_difficulty_constraint"] = normalize_direct(
        df["implementation_difficulty_norm"], 1, 5
    )
    df["system_change_resistance_constraint"] = normalize_direct(
        df["system_change_resistance_norm"], 1, 5
    )

    # Operational load
    df["time_constraint_score"] = normalize_direct(df["time_constraint_norm"], 1, 5)
    df["staffing_constraint_score"] = normalize_direct(
        df["staffing_constraint_norm"], 1, 5
    )
    df["resource_constraint_score"] = normalize_direct(
        df["resource_constraint_norm"], 1, 5
    )

    # Human capacity / adoption readiness
    df["digital_usage_constraint_score"] = normalize_reverse(
        df["digital_usage_constraint"], 1, 4
    )
    df["training_deficit_score"] = normalize_direct(df["training_deficit_norm"], 1, 5)
    df["willingness_constraint_score"] = normalize_reverse(
        df["willingness_constraint"], 1, 5
    )
    df["perceived_utility_constraint"] = normalize_reverse(
        df["perceived_digital_utility_norm"], 1, 4
    )
    df["pilot_openness_constraint"] = normalize_reverse(df["pilot_openness"], 0, 1)
    df["previous_implementation_constraint"] = normalize_reverse(
        df["previous_digital_implementation"], 0, 1
    )

    return df


def compute_branch_scores(df):
    df = df.copy()

    df["ICI"] = df[
        [
            "device_constraint",
            "internet_stability_constraint",
            "digital_tool_variety_constraint",
            "recording_system_constraint",
        ]
    ].mean(axis=1)

    df["OCI"] = df[
        [
            "administrative_disorganization_constraint",
            "implementation_difficulty_constraint",
            "system_change_resistance_constraint",
        ]
    ].mean(axis=1)

    df["OLI"] = df[
        [
            "admin_time_load_constraint",
            "time_constraint_score",
            "staffing_constraint_score",
            "resource_constraint_score",
        ]
    ].mean(axis=1)

    df["HCARI"] = df[
        [
            "digital_usage_constraint_score",
            "training_deficit_score",
            "willingness_constraint_score",
            "perceived_utility_constraint",
            "pilot_openness_constraint",
            "previous_implementation_constraint",
        ]
    ].mean(axis=1)

    return df


def compute_global_scores(df):
    df = df.copy()

    df["AFS_baseline"] = df[["ICI", "OCI", "OLI", "HCARI"]].mean(axis=1)

    df["AFS_theoretical"] = (
        0.25 * df["ICI"] + 0.30 * df["OCI"] + 0.20 * df["OLI"] + 0.25 * df["HCARI"]
    )

    return df


def classify_friction(score):
    if score < 0.25:
        return "Low friction"
    elif score < 0.50:
        return "Moderate friction"
    elif score < 0.75:
        return "High friction"
    else:
        return "Critical friction"


def identify_dominant_constraint(row):
    branches = {
        "Infrastructure": row["ICI"],
        "Organizational": row["OCI"],
        "Operational": row["OLI"],
        "Human-capacity/adoption readiness": row["HCARI"],
    }
    return max(branches, key=branches.get)


def add_diagnostics(df):
    df = df.copy()

    df["friction_category"] = df["AFS_theoretical"].apply(classify_friction)
    df["dominant_constraint"] = df.apply(identify_dominant_constraint, axis=1)

    return df


def save_scored_data(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def main():
    print("[PIPELINE] Ejecutando módulo 'scoring.py'...")

    df = load_clean_data(INPUT_FILE)
    print(f"[INFO] Filas cargadas: {len(df)}")

    df = create_tool_variety(df)
    df = compute_normalized_constraints(df)
    df = compute_branch_scores(df)
    df = compute_global_scores(df)
    df = add_diagnostics(df)

    save_scored_data(df, OUTPUT_FILE)

    print(f"[ÉXITO] Dataset scored exportado en: {OUTPUT_FILE}")
    print(
        "[INFO] Scores generados: ICI, OCI, OLI, HCARI, AFS_baseline, AFS_theoretical"
    )


if __name__ == "__main__":
    main()
