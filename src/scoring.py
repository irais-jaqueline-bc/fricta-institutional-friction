import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "clean_data.csv")
OUTPUT_FILE = os.path.join("data", "processed", "fricta_scored.csv")


def normalize_direct(series, min_value, max_value):
    series = pd.to_numeric(series, errors="coerce")
    return (series - min_value) / (max_value - min_value)


def normalize_reverse(series, min_value, max_value):
    series = pd.to_numeric(series, errors="coerce")
    return (max_value - series) / (max_value - min_value)


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}")
    return pd.read_csv(filepath)


def validate_required_columns(df):
    required = [
        "available_devices_encoded",
        "internet_stability_encoded",
        "registration_system_type_encoded",
        "admin_time_load_encoded",
        "administrative_organization_encoded",
        "digital_usage_frequency_encoded",
        "previous_digital_implementation_encoded",
        "implementation_difficulty_encoded",
        "time_constraint_encoded",
        "staffing_constraint_encoded",
        "training_deficit_encoded",
        "resource_constraint_encoded",
        "system_change_resistance_encoded",
        "perceived_digital_utility_encoded",
        "tool_adoption_willingness_encoded",
        "pilot_openness_encoded",
        "digital_tool_variety",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas requeridas para scoring: {missing}")


def compute_normalized_constraints(df):
    df = df.copy()

    # Infrastructure constraints
    df["device_constraint"] = normalize_reverse(df["available_devices_encoded"], 0, 3)

    df["internet_stability_constraint"] = normalize_reverse(
        df["internet_stability_encoded"], 1, 5
    )

    df["digital_tool_variety_constraint"] = normalize_reverse(
        df["digital_tool_variety"], 0, 5
    )

    # Organizational constraints
    df["recording_system_constraint"] = normalize_reverse(
        df["registration_system_type_encoded"], 1, 4
    )

    df["admin_time_load_constraint"] = normalize_direct(
        df["admin_time_load_encoded"], 1, 4
    )

    df["administrative_disorganization_constraint"] = normalize_reverse(
        df["administrative_organization_encoded"], 1, 5
    )

    df["system_change_resistance_constraint"] = normalize_direct(
        df["system_change_resistance_encoded"], 1, 5
    )

    # Operational / implementation constraints
    df["digital_usage_constraint_score"] = normalize_reverse(
        df["digital_usage_frequency_encoded"], 1, 4
    )

    df["implementation_difficulty_constraint"] = normalize_direct(
        df["implementation_difficulty_encoded"], 1, 5
    )

    df["previous_implementation_constraint"] = 1 - pd.to_numeric(
        df["previous_digital_implementation_encoded"], errors="coerce"
    )

    # Human-capacity / adoption-readiness constraints
    df["time_constraint_score"] = normalize_direct(df["time_constraint_encoded"], 1, 5)

    df["staffing_constraint_score"] = normalize_direct(
        df["staffing_constraint_encoded"], 1, 5
    )

    df["training_deficit_score"] = normalize_direct(
        df["training_deficit_encoded"], 1, 5
    )

    df["resource_constraint_score"] = normalize_direct(
        df["resource_constraint_encoded"], 1, 5
    )

    df["willingness_constraint_score"] = normalize_reverse(
        df["tool_adoption_willingness_encoded"], 1, 5
    )

    df["perceived_utility_constraint"] = normalize_reverse(
        df["perceived_digital_utility_encoded"], 1, 4
    )

    df["pilot_openness_constraint"] = 1 - pd.to_numeric(
        df["pilot_openness_encoded"], errors="coerce"
    )

    return df


def compute_branch_scores(df):
    df = df.copy()

    df["ICI"] = df[
        [
            "device_constraint",
            "internet_stability_constraint",
            "digital_tool_variety_constraint",
        ]
    ].mean(axis=1)

    df["OCI"] = df[
        [
            "recording_system_constraint",
            "admin_time_load_constraint",
            "administrative_disorganization_constraint",
            "system_change_resistance_constraint",
        ]
    ].mean(axis=1)

    df["OLI"] = df[
        [
            "digital_usage_constraint_score",
            "implementation_difficulty_constraint",
            "previous_implementation_constraint",
        ]
    ].mean(axis=1)

    df["HCARI"] = df[
        [
            "time_constraint_score",
            "staffing_constraint_score",
            "training_deficit_score",
            "resource_constraint_score",
            "willingness_constraint_score",
            "perceived_utility_constraint",
            "pilot_openness_constraint",
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


def check_score_quality(df):
    score_cols = [
        "time_constraint_score",
        "staffing_constraint_score",
        "training_deficit_score",
        "resource_constraint_score",
        "ICI",
        "OCI",
        "OLI",
        "HCARI",
        "AFS_theoretical",
    ]

    for col in score_cols:
        if df[col].isna().any():
            print(f"[ADVERTENCIA] Valores NaN detectados en {col}")

        if df[col].std() == 0:
            print(f"[ADVERTENCIA] Columna constante detectada en {col}")


def save_data(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def main():
    print("[PIPELINE] Ejecutando módulo 'scoring.py'...")

    df = load_data(INPUT_FILE)

    print(f"[INFO] Filas cargadas: {len(df)}")

    validate_required_columns(df)

    df = compute_normalized_constraints(df)
    df = compute_branch_scores(df)
    df = compute_global_scores(df)

    check_score_quality(df)

    save_data(df, OUTPUT_FILE)

    print(f"[ÉXITO] Dataset scored exportado en: {OUTPUT_FILE}")
    print(
        "[INFO] Scores generados: ICI, OCI, OLI, HCARI, AFS_baseline, AFS_theoretical"
    )


if __name__ == "__main__":
    main()
