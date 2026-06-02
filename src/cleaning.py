import os
import pandas as pd


def load_raw_data(filepath):
    """Carga el archivo CSV crudo exportado de Google Forms."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo en: {filepath}")
    return pd.read_csv(filepath)


def get_column_mapping():
    """Diccionario de mapeo oficial alineado con data_dictionary.pdf.

    Traduce las preguntas crudas de Google Forms a variables estándar.
    Se omiten explícitamente Marca de tiempo, correos y columnas de contacto.
    """
    return {
        "Estado": "state",
        "¿Qué tipo de institución es?": "institution_type",
        "¿Cuántos niños/niñas atiende la institución?": "children_served",
        "Número aproximado de empleados": "staff_size",
        "¿Cuántas computadoras tienen disponibles?": "available_devices",
        "¿Qué tan estable es su acceso a internet?": "internet_constraint",
        "¿Qué herramientas digitales utilizan actualmente?": "current_digital_tools",
        "¿Cómo registran la información de los niños?": "information_recording_method",
        "¿Cuánto tiempo diario se dedica a tareas administrativas?": "admin_time_load_norm",
        "¿Qué tan organizados consideran sus procesos administrativos?": "admin_disorganization",
        "¿Con qué frecuencia usan herramientas digitales?": "digital_usage_constraint",
        "¿Han intentado implementar nuevas herramientas digitales?": "previous_digital_implementation",
        "¿Qué tan difícil fue implementarlas?": "implementation_difficulty_norm",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de tiempo]": "time_constraint_norm",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de personal]": "staffing_constraint_norm",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de capacitación]": "training_deficit_norm",
        "¿Qué tanto tienen las siguientes dificultades?[Falta de recursos]": "resource_constraint_norm",
        "¿Qué tan difícil sería cambiar su sistema actual por uno nuevo?": "system_change_resistance_norm",
        "¿Qué tanto ayudaría una herramienta digital bien diseñada?": "perceived_digital_utility_norm",
        "¿Qué tan dispuestos estarían a probar una nueva herramienta?": "willingness_constraint",
        "¿Estarían abiertos a probar una herramienta digital gratuita en su institución?": "pilot_openness",
    }


def transform_responses_to_raw_values(df):
    """Mapea las respuestas cualitativas de Google Forms a los 'Raw Values'

    numéricos enteros estipulados en la taxonomía oficial de FRICTA.
    """
    df_transformed = df.copy()

    # SECTION 2 — INFRASTRUCTURE MAPPINGS
    devices_map = {"Ninguna": 0, "1-2": 1, "3-5": 2, "6+": 3}

    internet_map = {
        "Muy inestable": 1,
        "Inestable": 2,
        "Medio": 3,
        "Estable": 4,
        "Muy estable": 5,
    }

    resource_map = {
        "Poco": 1,
        "Medio": 2,
        "Alto": 3,
        "Mucho": 4,
        "Muy alto": 4,
        "Crítico": 5,
    }

    # SECTION 3 — ADMINISTRATIVE PROCESSES MAPPINGS
    registration_map = {
        "Papel": 1,
        "Mixto": 2,
        "Excel": 3,
        "Software": 4,
        "Software especializado": 4,
    }

    admin_time_map = {
        "Menos de 1 hora": 1,
        "1-3 horas": 2,
        "3-5 horas": 3,
        "5+ horas": 4,
    }

    organization_map = {
        "Muy bajo": 1,
        "Bajo": 2,
        "Medio": 3,
        "Alto": 4,
        "Muy alto": 5,
    }

    # SECTION 4 & 5 — ADOPTION & FRICTION MAPPINGS
    frequency_map = {"Nunca": 1, "Rara vez": 1, "Semanal": 2, "Mensual": 3, "Diario": 4}

    prev_implementation_map = {"Sí": 1, "No": 0}

    difficulty_map = {
        "Muy fácil": 1,
        "Fácil": 2,
        "Moderado": 3,
        "Difícil": 4,
        "Muy difícil": 5,
    }

    utility_map = {"Nada": 1, "Poco": 2, "Medio": 3, "Mucho": 4}

    willingness_map = {
        "Nada dispuestos": 1,
        "Poco dispuestos": 2,
        "Medio": 3,
        "Dispuestos": 4,
        "Muy dispuestos": 5,
    }

    pilot_map = {"Sí": 1, "No": 0, "Tal vez": 0.5}

    # Aplicación segura de mapeos columna por columna preservando tipos enteros
    if "available_devices" in df_transformed.columns:
        df_transformed["available_devices"] = (
            df_transformed["available_devices"].map(devices_map).fillna(0).astype(int)
        )

    if "internet_constraint" in df_transformed.columns:
        df_transformed["internet_constraint"] = (
            df_transformed["internet_constraint"]
            .map(internet_map)
            .fillna(3)
            .astype(int)
        )

    if "resource_constraint_norm" in df_transformed.columns:
        df_transformed["resource_constraint_norm"] = (
            df_transformed["resource_constraint_norm"]
            .map(resource_map)
            .fillna(3)
            .astype(int)
        )

    if "information_recording_method" in df_transformed.columns:
        df_transformed["information_recording_method"] = (
            df_transformed["information_recording_method"]
            .map(registration_map)
            .fillna(1)
            .astype(int)
        )

    if "admin_time_load_norm" in df_transformed.columns:
        df_transformed["admin_time_load_norm"] = (
            df_transformed["admin_time_load_norm"]
            .map(admin_time_map)
            .fillna(2)
            .astype(int)
        )

    if "admin_disorganization" in df_transformed.columns:
        df_transformed["admin_disorganization"] = (
            df_transformed["admin_disorganization"]
            .map(organization_map)
            .fillna(3)
            .astype(int)
        )

    if "digital_usage_constraint" in df_transformed.columns:
        df_transformed["digital_usage_constraint"] = (
            df_transformed["digital_usage_constraint"]
            .map(frequency_map)
            .fillna(2)
            .astype(int)
        )

    if "previous_digital_implementation" in df_transformed.columns:
        df_transformed["previous_digital_implementation"] = (
            df_transformed["previous_digital_implementation"]
            .map(prev_implementation_map)
            .fillna(0)
            .astype(int)
        )

    # Escalas estandarizadas de dificultad y barreras (1-5)
    shared_1_5_cols = [
        "implementation_difficulty_norm",
        "time_constraint_norm",
        "staffing_constraint_norm",
        "training_deficit_norm",
        "system_change_resistance_norm",
    ]
    for col in shared_1_5_cols:
        if col in df_transformed.columns:
            df_transformed[col] = (
                df_transformed[col].map(difficulty_map).fillna(3).astype(int)
            )

    if "perceived_digital_utility_norm" in df_transformed.columns:
        df_transformed["perceived_digital_utility_norm"] = (
            df_transformed["perceived_digital_utility_norm"]
            .map(utility_map)
            .fillna(3)
            .astype(int)
        )

    if "willingness_constraint" in df_transformed.columns:
        df_transformed["willingness_constraint"] = (
            df_transformed["willingness_constraint"]
            .map(willingness_map)
            .fillna(3)
            .astype(int)
        )

    if "pilot_openness" in df_transformed.columns:
        df_transformed["pilot_openness"] = (
            df_transformed["pilot_openness"].map(pilot_map).fillna(0.0).astype(float)
        )

    return df_transformed


def apply_methodological_criteria(df):
    """Aplica los criterios de inclusión/exclusión descritos en methodology.pdf."""
    initial_count = len(df)

    # Criterio: Eliminar registros duplicados
    df = df.drop_duplicates()

    # Criterio: Asegurar consistencia geográfica y demográfica mínima
    df = df[df["state"].notna() & (df["state"].str.strip() != "")]
    df = df[df["institution_type"].notna()]

    print(f"[METODOLOGÍA] Registros crudos leídos: {initial_count}")
    print(f"[METODOLOGÍA] Registros válidos finales: {len(df)}")
    return df


def main():
    # CAMBIO: Ruta de entrada cruda (Local, oculta por .gitignore)
    RAW_FILE = os.path.join(
        "data", "raw", "Uso de herramientas digitales en casas hogar.csv"
    )
    # CAMBIO OPTIMIZADO: Carpeta en minúsculas y sin bloquear que irá a GitHub
    PROCESSED_FILE = os.path.join("data", "data_processed", "clean_data.csv")

    try:
        print("[PIPELINE] Ejecutando módulo 'cleaning.py'...")

        # 1. Cargar la información
        df_raw = load_raw_data(RAW_FILE)

        # 2. Renombrar y filtrar columnas bajo el diccionario analítico
        mapping = get_column_mapping()
        df_mapped = df_raw.rename(columns=mapping)

        # Respaldar las variables cualitativas de texto que servirán para lógicas derivadas complejas
        tools_backup = df_mapped["current_digital_tools"].copy()
        children_backup = df_mapped["children_served"].copy()
        staff_backup = df_mapped["staff_size"].copy()

        # Proyectar solo las columnas analíticas permitidas (elimina variables de identidad automáticamente)
        valid_cols = [col for col in mapping.values() if col in df_mapped.columns]
        df_projected = df_mapped[valid_cols].copy()

        # 3. Aplicar filtros de calidad de filas
        df_filtered = apply_methodological_criteria(df_projected)

        # 4. Transformar variables core a sus Raw Values numéricos
        df_clean = transform_responses_to_raw_values(df_filtered)

        # Reinyectar variables cualitativas críticas usando .loc de manera segura
        df_clean = df_clean.assign(
            current_digital_tools=tools_backup.loc[df_clean.index],
            children_served=children_backup.loc[df_clean.index],
            staff_size=staff_backup.loc[df_clean.index],
        )

        # 5. PROTOCOLO DE PRIVACIDAD: Shuffling estricto sin correlación temporal
        df_anonymized = df_clean.sample(frac=1, random_state=42).reset_index(
            drop=True
        )
        print(
            "[PRIVACIDAD] Data shuffling aplicado. Orden temporal roto exitosamente."
        )

        # 6. Exportar base de datos limpia para scoring
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        df_anonymized.to_csv(PROCESSED_FILE, index=False)
        print(f"[ÉXITO] Base de datos limpia exportada en: {PROCESSED_FILE}")

    except Exception as e:
        print(f"[ERROR] No se pudo completar la limpieza: {str(e)}")


if __name__ == "__main__":
    main()
