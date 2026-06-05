import os
import re
import pandas as pd

RAW_FILE = os.path.join(
    "data", "raw", "Uso de herramientas digitales en casas hogar.csv"
)

OUTPUT_DIR = os.path.join("pilot", "reports")
CONTACT_LIST_FILE = os.path.join("pilot", "pilot_contact_list.csv")


CONTACT_COLUMN = (
    "Si desean participar en una prueba piloto o recibir un análisis comparativo "
    "de su institución, pueden dejar un correo o medio de contacto (opcional)"
)


COLUMN_MAPPING = {
    "Estado": "state",
    "¿Qué tipo de institución es?": "institution_type",
    "¿Cuántos niños/niñas atiende la institución?": "children_served",
    "Número aproximado de empleados": "staff_size",
    "¿Cuántas computadoras tienen disponibles?": "available_devices",
    "¿Qué tan estable es su acceso a internet?": "internet_stability",
    "¿Qué herramientas digitales utilizan actualmente?": "current_digital_tools",
    "¿Cómo registran la información de los niños?": "registration_system_type",
    "¿Cuánto tiempo diario se dedica a tareas administrativas?": "admin_time_load",
    "¿Qué tan organizados consideran sus procesos administrativos?": "administrative_organization",
    "¿Con qué frecuencia usan herramientas digitales?": "digital_usage_frequency",
    "¿Han intentado implementar nuevas herramientas digitales?": "previous_digital_implementation",
    "¿Qué tan difícil fue implementarlas?": "implementation_difficulty",
    "¿Qué tanto tienen las siguientes dificultades?[Falta de tiempo]": "time_constraint",
    "¿Qué tanto tienen las siguientes dificultades?[Falta de personal]": "staffing_constraint",
    "¿Qué tanto tienen las siguientes dificultades?[Falta de capacitación]": "training_deficit",
    "¿Qué tanto tienen las siguientes dificultades?[Falta de recursos]": "resource_constraint",
    "¿Qué tan difícil sería cambiar su sistema actual por uno nuevo?": "system_change_resistance",
    "¿Qué tanto ayudaría una herramienta digital bien diseñada?": "perceived_digital_utility",
    "¿Qué tan dispuestos estarían a probar una nueva herramienta?": "tool_adoption_willingness",
    "¿Estarían abiertos a probar una herramienta digital gratuita en su institución?": "pilot_openness",
}


MAPPINGS = {
    "available_devices": {"0": 0, "Ninguna": 0, "1-2": 1, "3-5": 2, "6+": 3},
    "internet_stability": {
        "No hay acceso": 1,
        "Muy inestable": 2,
        "Inestable": 3,
        "Estable": 4,
        "Muy estable": 5,
    },
    "registration_system_type": {"Papel": 1, "Excel": 2, "Mixto": 3, "Software": 4},
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
    "digital_usage_frequency": {"Nunca": 1, "Rara vez": 2, "Semanal": 3, "Diario": 4},
    "previous_digital_implementation": {"No": 0, "Sí": 1},
    "implementation_difficulty": {
        "Muy fácil": 1,
        "Fácil": 2,
        "Moderado": 3,
        "Difícil": 4,
        "Muy difícil": 5,
    },
    "time_constraint": {"Nada": 1, "Poco": 2, "Medio": 3, "Alto": 4, "Muy alto": 5},
    "staffing_constraint": {"Nada": 1, "Poco": 2, "Medio": 3, "Alto": 4, "Muy alto": 5},
    "training_deficit": {"Nada": 1, "Poco": 2, "Medio": 3, "Alto": 4, "Muy alto": 5},
    "resource_constraint": {"Nada": 1, "Poco": 2, "Medio": 3, "Alto": 4, "Muy alto": 5},
    "system_change_resistance": {
        "Muy fácil": 1,
        "Fácil": 2,
        "Neutral": 3,
        "Difícil": 4,
        "Muy difícil": 5,
    },
    "perceived_digital_utility": {"Nada": 1, "Poco": 2, "Algo": 3, "Mucho": 4},
    "tool_adoption_willingness": {
        "Nada dispuestos": 1,
        "Poco dispuestos": 2,
        "Neutrales": 3,
        "Dispuestos": 4,
        "Muy dispuestos": 5,
    },
    "pilot_openness": {"No": 0, "Tal vez": 0.5, "Sí": 1},
}


def normalize_direct(series, min_value, max_value):
    series = pd.to_numeric(series, errors="coerce")
    return (series - min_value) / (max_value - min_value)


def normalize_reverse(series, min_value, max_value):
    series = pd.to_numeric(series, errors="coerce")
    return (max_value - series) / (max_value - min_value)


def clean_text(df):
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


def encode_variables(df):
    df = df.copy()
    for col, mapping in MAPPINGS.items():
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].map(mapping)
    return df


def create_tool_columns(df):
    df = df.copy()
    tools = df["current_digital_tools"].fillna("")

    df["uses_excel"] = tools.str.contains("Excel", case=False, regex=False).astype(int)
    df["uses_whatsapp"] = tools.str.contains(
        "WhatsApp", case=False, regex=False
    ).astype(int)
    df["uses_google_workspace"] = tools.str.contains(
        "Google Drive / Docs", case=False, regex=False
    ).astype(int)
    df["uses_specialized_software"] = tools.str.contains(
        "Software especializado", case=False, regex=False
    ).astype(int)
    df["uses_other_tool"] = tools.str.contains("Otra", case=False, regex=False).astype(
        int
    )
    df["uses_no_tool"] = tools.str.contains("Ninguna", case=False, regex=False).astype(
        int
    )

    df["digital_tool_variety"] = (
        df["uses_excel"]
        + df["uses_whatsapp"]
        + df["uses_google_workspace"]
        + df["uses_specialized_software"]
        + df["uses_other_tool"]
    )

    return df


def compute_scores(df):
    df = df.copy()

    df["device_constraint"] = normalize_reverse(df["available_devices_encoded"], 0, 3)
    df["internet_stability_constraint"] = normalize_reverse(
        df["internet_stability_encoded"], 1, 5
    )
    df["digital_tool_variety_constraint"] = normalize_reverse(
        df["digital_tool_variety"], 0, 5
    )

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

    df["digital_usage_constraint_score"] = normalize_reverse(
        df["digital_usage_frequency_encoded"], 1, 4
    )
    df["implementation_difficulty_constraint"] = normalize_direct(
        df["implementation_difficulty_encoded"], 1, 5
    )
    df["previous_implementation_constraint"] = 1 - pd.to_numeric(
        df["previous_digital_implementation_encoded"], errors="coerce"
    )

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

    df["AFS_theoretical"] = (
        0.25 * df["ICI"] + 0.30 * df["OCI"] + 0.20 * df["OLI"] + 0.25 * df["HCARI"]
    )

    return df


def classify_score(value):
    if value < 0.34:
        return "Low"
    elif value < 0.67:
        return "Medium"
    return "High"


def comparative_label(value, mean):
    if value < mean - 0.05:
        return "Below sample average"
    elif value > mean + 0.05:
        return "Above sample average"
    return "Near sample average"


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text[:50].strip("_")


def top_friction_drivers(row):
    candidates = {
        "Device availability": row["device_constraint"],
        "Internet stability": row["internet_stability_constraint"],
        "Digital tool variety": row["digital_tool_variety_constraint"],
        "Record-keeping system": row["recording_system_constraint"],
        "Administrative time load": row["admin_time_load_constraint"],
        "Administrative organization": row["administrative_disorganization_constraint"],
        "Digital usage frequency": row["digital_usage_constraint_score"],
        "Implementation difficulty": row["implementation_difficulty_constraint"],
        "Previous implementation experience": row["previous_implementation_constraint"],
        "Time availability": row["time_constraint_score"],
        "Staffing capacity": row["staffing_constraint_score"],
        "Training deficit": row["training_deficit_score"],
        "Resource availability": row["resource_constraint_score"],
        "Willingness to adopt tools": row["willingness_constraint_score"],
        "Pilot openness": row["pilot_openness_constraint"],
    }

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return ranked[:3]


def generate_recommendations(drivers):
    recommendations = []

    for driver, _ in drivers:
        if driver == "Device availability":
            recommendations.append(
                "Prioritize minimum access to shared digital devices for administrative and record-keeping tasks."
            )
        elif driver == "Record-keeping system":
            recommendations.append(
                "Begin migrating key records from paper-only processes to a simple digital or hybrid registry."
            )
        elif driver == "Digital usage frequency":
            recommendations.append(
                "Introduce a small, recurring digital workflow instead of attempting a full system change at once."
            )
        elif driver == "Previous implementation experience":
            recommendations.append(
                "Start with a low-risk pilot to build internal experience with digital implementation."
            )
        elif driver == "Training deficit":
            recommendations.append(
                "Use short, task-specific training sessions focused on one tool or workflow at a time."
            )
        elif driver == "Resource availability":
            recommendations.append(
                "Identify low-cost or free digital tools before considering more complex paid systems."
            )
        else:
            recommendations.append(
                f"Review the institutional process related to {driver.lower()} and identify one small improvement action."
            )

    return recommendations[:3]


def generate_report(row, sample_means):
    drivers = top_friction_drivers(row)
    recommendations = generate_recommendations(drivers)

    contact_or_comment = row["pilot_contact"]

    report = f"""# FRICTA Institutional Diagnostic Report

**Institution / Contact Reference:** {contact_or_comment}

**Institution ID:** {row["pilot_id"]}

**Framework:** FRICTA – Framework for Institutional Digital Adoption Friction Assessment

---

## Executive Summary

Thank you for participating in the FRICTA study on digital adoption in Mexican childcare institutions.

Based on your survey responses, we generated an institutional friction profile designed to identify factors that may affect the adoption and effective use of digital tools within your organization.

This report is diagnostic and educational. It is not an audit, certification, ranking, or institutional evaluation.

---

## Overall Friction Profile

| Dimension | Score | Interpretation |
|---|---:|---|
| Infrastructure Constraints Index (ICI) | {row["ICI"]:.3f} | {classify_score(row["ICI"])} |
| Organizational Constraints Index (OCI) | {row["OCI"]:.3f} | {classify_score(row["OCI"])} |
| Operational Limitations Index (OLI) | {row["OLI"]:.3f} | {classify_score(row["OLI"])} |
| Human Capacity and Readiness Index (HCARI) | {row["HCARI"]:.3f} | {classify_score(row["HCARI"])} |
| Overall FRICTA Score | {row["AFS_theoretical"]:.3f} | {classify_score(row["AFS_theoretical"])} |

---

## Comparative Position Within the Sample

Compared with the 81 institutions included in the FRICTA study:

- Infrastructure profile: **{comparative_label(row["ICI"], sample_means["ICI"])}**
- Organizational profile: **{comparative_label(row["OCI"], sample_means["OCI"])}**
- Operational profile: **{comparative_label(row["OLI"], sample_means["OLI"])}**
- Human-capacity profile: **{comparative_label(row["HCARI"], sample_means["HCARI"])}**
- Overall friction profile: **{comparative_label(row["AFS_theoretical"], sample_means["AFS_theoretical"])}**

---

## Main Friction Drivers

The factors contributing most strongly to the diagnostic profile appear to be:

1. **{drivers[0][0]}** – score: {drivers[0][1]:.3f}
2. **{drivers[1][0]}** – score: {drivers[1][1]:.3f}
3. **{drivers[2][0]}** – score: {drivers[2][1]:.3f}

---

## Suggested Improvement Priorities

Based on the observed profile, the following actions may be useful:

1. {recommendations[0]}
2. {recommendations[1]}
3. {recommendations[2]}

---

## Pilot Feedback Question

To continue with the pilot phase, please reply to the email with a number from 1 to 5:

**To what extent does this diagnostic reflect the current reality of your institution?**

1 = Does not reflect our reality at all  
2 = Reflects our reality slightly  
3 = Reflects our reality moderately  
4 = Reflects our reality well  
5 = Reflects our reality very accurately  

Optional comments are welcome.

---

Thank you for supporting the FRICTA research project.
"""

    return report


def main():
    print("[PIPELINE] Generating pilot diagnostic reports...")

    raw = pd.read_csv(RAW_FILE)
    raw = clean_text(raw)

    if CONTACT_COLUMN not in raw.columns:
        raise ValueError(f"No se encontró la columna de contacto: {CONTACT_COLUMN}")

    raw = raw[raw[CONTACT_COLUMN].notna()].copy()

    print(f"[INFO] Institutions with contact/comment: {len(raw)}")

    raw["pilot_contact"] = raw[CONTACT_COLUMN]

    mapped = raw.rename(columns=COLUMN_MAPPING)

    keep_cols = ["pilot_contact"] + [
        col for col in COLUMN_MAPPING.values() if col in mapped.columns
    ]

    df = mapped[keep_cols].copy()
    df = clean_text(df)
    df = encode_variables(df)
    df = create_tool_columns(df)
    df = compute_scores(df)

    df = df.reset_index(drop=True)
    df.insert(0, "pilot_id", [f"PILOT_{i+1:03d}" for i in range(len(df))])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_means = df[["ICI", "OCI", "OLI", "HCARI", "AFS_theoretical"]].mean()

    contact_rows = []

    for _, row in df.iterrows():
        filename = f"{row['pilot_id']}_{safe_filename(row['pilot_contact'])}.md"
        path = os.path.join(OUTPUT_DIR, filename)

        report = generate_report(row, sample_means)

        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

        contact_rows.append(
            {
                "pilot_id": row["pilot_id"],
                "pilot_contact": row["pilot_contact"],
                "report_path": path,
                "AFS_theoretical": row["AFS_theoretical"],
                "ICI": row["ICI"],
                "OCI": row["OCI"],
                "OLI": row["OLI"],
                "HCARI": row["HCARI"],
            }
        )

    pd.DataFrame(contact_rows).to_csv(CONTACT_LIST_FILE, index=False)

    print("[SUCCESS] Pilot reports generated.")
    print(f"[OUTPUT] Reports folder: {OUTPUT_DIR}")
    print(f"[OUTPUT] Contact list: {CONTACT_LIST_FILE}")


if __name__ == "__main__":
    main()
