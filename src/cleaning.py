import os
import pandas as pd


RAW_FILE = os.path.join(
    "data", "raw", "Uso de herramientas digitales en casas hogar.csv"
)

PROCESSED_FILE = os.path.join(
    "data", "processed", "fricta_clean_anonymized.csv"
)


def load_raw_data(filepath):
    """Load raw Google Forms CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo en: {filepath}")
    return pd.read_csv(filepath)


def get_column_mapping():
    """Map Google Forms columns to FRICTA canonical variables."""
    return {
        "Estado": "state",
        "¿Qué tipo de institución es?": "institution_type",
        "¿Cuántos niños/niñas atiende la institución?": "children_served",
        "Número aproximado de empleados": "staff_size",
        "¿Cuántas computadoras tienen disponibles?": "available_devices",
        "¿Qué tan estable es su acceso a internet?": "internet_stability",
        "¿Qué herramientas digitales utilizan actualmente?": "digital_tools_used",
        "¿Cómo registran la información de los niños?": "registration_system_type",
        "¿Cuánto tiempo diario se dedica a tareas administrativas?": "admin_time_load",
        "¿Qué tan organizados consideran sus procesos administrativos?": "administrative_organization",
        "¿Con qué frecuencia usan herramientas digitales?": "digital_usage_frequency",
        "¿Han intentado implementar nuevas herramientas digitales?": "previous_digital_implementation",
        "¿Qué tan difícil fue implementarlas?": "implementation_difficulty",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de tiempo]": "time_constraint",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de personal]": "staffing_constraint",
        "¿Qué tanto tienen las siguientes dificultades}[Falta de capacitación]": "training_deficit",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de capacitación]": "training_deficit",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de recursos]": "resource_constraint",
        "¿Qué tan difícil sería cambiar su sistema actual por uno nuevo?": "system_change_resistance",
        "¿Qué tanto ayudaría una herramienta digital bien diseñada?": "perceived_digital_utility",
        "¿Qué tan dispuestos estarían a probar una nueva herramienta?": "tool_adoption_willingness",
        "¿Estarían abiertos a probar una herramienta digital gratuita en su institución?": "pilot_openness",
    }


def clean_text_values(df):
    """Clean text values without removing accents."""
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\n", " ", regex=False)
                .str.strip()
                .replace({"nan": pd.NA, "": pd.NA})
            )

    return df


def apply_methodological_criteria(df):
    """Apply basic inclusion/exclusion criteria."""
    initial_count = len(df)

    df = df.drop_duplicates()

    df = df[df["state"].notna()]
    df = df[df["institution_type"].notna()]

    print(f"[METODOLOGÍA] Registros crudos leídos: {initial_count}")
    print(f"[METODOLOGÍA] Registros válidos finales: {len(df)}")

    return df


def encode_variables(df):
    """Create encoded numeric variables. No 0-1 normalization here."""
    df = df.copy()

    mappings = {
        "available_devices": {
            "0": 0,
            "Ninguna": 0,
            "1-2": 1,
            "3-5": 2,
            "6+": 3,
        },
        "internet_stability": {
            "No hay acceso": 1,
            "Muy inestable": 2,
            "Inestable": 3,
            "Estable": 4,
            "Muy estable": 5,
        },
        "registration_system_type": {
            "Papel": 1,
            "Excel": 2,
            "Mixto": 3,
            "Software": 4,
        },
        "admin_time_load": {
            "Menos de 1 hora": 1,
            "1-3 horas": 2,
            "3-5 horas": 3,
            "5+ horas": 4,
        },
        "administrative_organization": {
            "Muy bajo": 1,
            "Bajo": 2,
            "Medio": 3,
            "Alto": 4,
            "Muy alto": 5,
        },
        "digital_usage_frequency": {
            "Nunca": 1,
            "Rara vez": 2,
            "Semanal": 3,
            "Diario": 4,
        },
        "previous_digital_implementation": {
            "No": 0,
            "Sí": 1,
        },
        "implementation_difficulty": {
            "Muy fácil": 1,
            "Fácil": 2,
            "Moderado": 3,
            "Difícil": 4,
            "Muy difícil": 5,
        },
        "time_constraint": {
            "Nada": 1,
            "Poco": 2,
            "Medio": 3,
            "Alto": 4,
            "Muy alto": 5,
        },
        "staffing_constraint": {
            "Nada": 1,
            "Poco": 2,
            "Medio": 3,
            "Alto": 4,
            "Muy alto": 5,
        },
        "training_deficit": {
            "Nada": 1,
            "Poco": 2,
            "Medio": 3,
            "Alto": 4,
            "Muy alto": 5,
        },
        "resource_constraint": {
            "Nada": 1,
            "Poco": 2,
            "Medio": 3,
            "Alto": 4,
            "Muy alto": 5,
        },
        "system_change_resistance": {
            "Muy fácil": 1,
            "Fácil": 2,
            "Neutral": 3,
            "Difícil": 4,
            "Muy difícil": 5,
        },
        "perceived_digital_utility": {
            "Nada": 1,
            "Poco": 2,
            "Algo": 3,
            "Mucho": 4,
        },
        "tool_adoption_willingness": {
            "Nada dispuestos": 1,
            "Poco dispuestos": 2,
            "Neutrales": 3,
            "Dispuestos": 4,
            "Muy dispuestos": 5,
        },
        "pilot_openness": {
            "No": 0,
            "Tal vez": 0.5,
            "Sí": 1,
        },
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            encoded_col = f"{col}_encoded"
            df[encoded_col] = df[col].map(mapping)

            unknown_values = df.loc[
                df[col].notna() & df[encoded_col].isna(), col
            ].unique()

            if len(unknown_values) > 0:
                print(f"[ADVERTENCIA] Valores no reconocidos en {col}: {unknown_values}")

    return df


def create_tool_binary_columns(df):
    """Create binary variables from the digital tools multiple-choice field."""
    df = df.copy()

    if "digital_tools_used" not in df.columns:
        print("[ADVERTENCIA] No existe la columna digital_tools_used.")
        return df

    tools = df["digital_tools_used"].fillna("")

    df["uses_excel"] = tools.str.contains("Excel", case=False, regex=False).astype(int)
    df["uses_whatsapp"] = tools.str.contains("WhatsApp", case=False, regex=False).astype(int)
    df["uses_google_workspace"] = tools.str.contains("Google Drive / Docs", case=False, regex=False).astype(int)
    df["uses_specialized_software"] = tools.str.contains("Software especializado", case=False, regex=False).astype(int)
    df["uses_other_tool"] = tools.str.contains("Otra", case=False, regex=False).astype(int)
    df["uses_no_tool"] = tools.str.contains("Ninguna", case=False, regex=False).astype(int)

    df["digital_tool_variety"] = (
        df["uses_excel"]
        + df["uses_whatsapp"]
        + df["uses_google_workspace"]
        + df["uses_specialized_software"]
        + df["uses_other_tool"]
    )

    return df


def create_anonymous_ids(df):
    """Shuffle rows and create anonymous institution IDs."""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "institution_id", [f"INST_{i+1:03d}" for i in range(len(df))])
    return df


def main():
    try:
        print("[PIPELINE] Ejecutando módulo 'cleaning.py'...")

        df_raw = load_raw_data(RAW_FILE)

        print(f"[INFO] Filas iniciales: {len(df_raw)}")
        print(f"[INFO] Columnas originales: {list(df_raw.columns)}")

        mapping = get_column_mapping()

        df_mapped = df_raw.rename(columns=mapping)

        valid_cols = [col for col in mapping.values() if col in df_mapped.columns]
        df_projected = df_mapped[valid_cols].copy()

        print("[PRIVACIDAD] Columnas sensibles eliminadas automáticamente por proyección.")
        print(f"[INFO] Columnas conservadas: {valid_cols}")

        df_projected = clean_text_values(df_projected)
        df_filtered = apply_methodological_criteria(df_projected)
        df_encoded = encode_variables(df_filtered)
        df_tools = create_tool_binary_columns(df_encoded)
        df_anonymized = create_anonymous_ids(df_tools)

        print("[PRIVACIDAD] Data shuffling aplicado. Orden temporal roto exitosamente.")

        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        df_anonymized.to_csv(PROCESSED_FILE, index=False)

        print(f"[ÉXITO] Base limpia exportada en: {PROCESSED_FILE}")
        print(f"[INFO] Filas finales: {len(df_anonymized)}")
        print(f"[INFO] Columnas finales: {list(df_anonymized.columns)}")

    except Exception as e:
        print(f"[ERROR] No se pudo completar la limpieza: {str(e)}")


if __name__ == "__main__":
    main()