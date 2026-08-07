cat > src/icdm_feature_audit.py <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fricta_scored.csv"
)

AUDIT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "audit"
DESIGN_DIR = PROJECT_ROOT / "icdm" / "design"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
DESIGN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE ROLES PROPOSED BEFORE SEEING CLUSTERING RESULTS
# ============================================================

FEATURE_SPEC = {
    # Primary institutional indicators
    "device_constraint": {
        "group": "infrastructure",
        "planned_role": "PRIMARY",
        "reason": "Direct measure of device-access friction.",
    },
    "internet_stability_constraint": {
        "group": "infrastructure",
        "planned_role": "PRIMARY",
        "reason": "Direct measure of connectivity instability.",
    },
    "digital_tool_variety_constraint": {
        "group": "infrastructure",
        "planned_role": "PRIMARY",
        "reason": "Measures limited institutional tool availability.",
    },
    "recording_system_constraint": {
        "group": "organizational",
        "planned_role": "PRIMARY",
        "reason": "Represents the maturity of administrative recording systems.",
    },
    "admin_time_load_constraint": {
        "group": "organizational_operational",
        "planned_role": "PRIMARY",
        "reason": "Captures administrative workload associated with institutional processes.",
    },
    "administrative_disorganization_constraint": {
        "group": "organizational",
        "planned_role": "PRIMARY",
        "reason": "Measures organizational-process disorder.",
    },
    "system_change_resistance_constraint": {
        "group": "organizational",
        "planned_role": "PRIMARY",
        "reason": "Represents institutional resistance to changing systems.",
    },
    "digital_usage_constraint_score": {
        "group": "digital_capacity",
        "planned_role": "PRIMARY",
        "reason": "Measures limited routine use of digital tools.",
    },
    "time_constraint_score": {
        "group": "operational_capacity",
        "planned_role": "PRIMARY",
        "reason": "Captures time scarcity affecting adoption.",
    },
    "staffing_constraint_score": {
        "group": "operational_capacity",
        "planned_role": "PRIMARY",
        "reason": "Captures personnel limitations.",
    },
    "training_deficit_score": {
        "group": "human_capacity",
        "planned_role": "PRIMARY",
        "reason": "Measures training limitations.",
    },
    "resource_constraint_score": {
        "group": "resource_capacity",
        "planned_role": "PRIMARY",
        "reason": "Captures general resource scarcity.",
    },
    "willingness_constraint_score": {
        "group": "adoption_readiness",
        "planned_role": "PRIMARY",
        "reason": "Measures readiness or willingness to adopt tools.",
    },

    # Auxiliary indicators
    "implementation_difficulty_constraint": {
        "group": "implementation_history",
        "planned_role": "VALIDATION_ONLY",
        "reason": (
            "May depend on prior implementation experience and is better "
            "reserved for external descriptive comparison."
        ),
    },
    "previous_implementation_constraint": {
        "group": "implementation_history",
        "planned_role": "SENSITIVITY_ONLY",
        "reason": (
            "Binary historical indicator rather than a continuous current "
            "friction condition."
        ),
    },
    "perceived_utility_constraint": {
        "group": "attitudinal",
        "planned_role": "SENSITIVITY_ONLY",
        "reason": (
            "Perceived utility may be attitudinal and conceptually distinct "
            "from institutional constraints."
        ),
    },
    "pilot_openness_constraint": {
        "group": "research_participation",
        "planned_role": "SENSITIVITY_ONLY",
        "reason": (
            "May reflect willingness to participate in the pilot rather than "
            "digital-adoption friction itself."
        ),
    },
}


HIGH_CORRELATION_THRESHOLD = 0.80
LOW_VARIANCE_THRESHOLD = 0.01
MAX_MISSING_RATE_PRIMARY = 0.20


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset canónico:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if len(df) == 0:
        raise ValueError("El dataset está vacío.")

    return df


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [
        feature
        for feature in FEATURE_SPEC
        if feature not in df.columns
    ]

    if missing:
        raise KeyError(
            "Faltan features candidatas en fricta_scored.csv:\n- "
            + "\n- ".join(missing)
        )


def audit_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature, specification in FEATURE_SPEC.items():
        series = pd.to_numeric(df[feature], errors="coerce")

        missing_count = int(series.isna().sum())
        missing_rate = float(series.isna().mean())
        variance = float(series.var(ddof=1))
        unique_values = int(series.nunique(dropna=True))
        outside_range = int(
            ((series < 0) | (series > 1)).fillna(False).sum()
        )

        flags = []

        if missing_rate > MAX_MISSING_RATE_PRIMARY:
            flags.append("HIGH_MISSINGNESS")

        if pd.isna(variance) or variance < LOW_VARIANCE_THRESHOLD:
            flags.append("LOW_VARIANCE")

        if unique_values <= 1:
            flags.append("CONSTANT")

        if outside_range > 0:
            flags.append("OUTSIDE_0_1")

        if specification["planned_role"] != "PRIMARY":
            flags.append("CONCEPTUAL_REVIEW")

        rows.append(
            {
                "feature": feature,
                "conceptual_group": specification["group"],
                "planned_role": specification["planned_role"],
                "reason": specification["reason"],
                "n_total": len(series),
                "n_observed": int(series.notna().sum()),
                "n_missing": missing_count,
                "missing_rate": round(missing_rate, 4),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std(ddof=1)), 4),
                "variance": round(variance, 4),
                "minimum": round(float(series.min()), 4),
                "maximum": round(float(series.max()), 4),
                "unique_values": unique_values,
                "outside_0_1_count": outside_range,
                "audit_status": "REVIEW" if flags else "PASS",
                "flags": ";".join(flags),
                "final_role": "",
                "final_decision_reason": "",
            }
        )

    return pd.DataFrame(rows)


def calculate_correlations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = list(FEATURE_SPEC)

    numeric = df[features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation_matrix = numeric.corr(method="pearson")

    pairs = []

    for index, feature_a in enumerate(features):
        for feature_b in features[index + 1:]:
            correlation = correlation_matrix.loc[
                feature_a,
                feature_b,
            ]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= HIGH_CORRELATION_THRESHOLD:
                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": round(float(correlation), 4),
                        "absolute_correlation": round(
                            abs(float(correlation)),
                            4,
                        ),
                        "same_conceptual_group": (
                            FEATURE_SPEC[feature_a]["group"]
                            == FEATURE_SPEC[feature_b]["group"]
                        ),
                        "decision": "REVIEW_REDUNDANCY",
                    }
                )

    high_correlations = pd.DataFrame(
        pairs,
        columns=[
            "feature_a",
            "feature_b",
            "correlation",
            "absolute_correlation",
            "same_conceptual_group",
            "decision",
        ],
    )

    return correlation_matrix, high_correlations


def find_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    features = list(FEATURE_SPEC)
    duplicates = []

    for index, feature_a in enumerate(features):
        series_a = pd.to_numeric(
            df[feature_a],
            errors="coerce",
        ).to_numpy(dtype=float)

        for feature_b in features[index + 1:]:
            series_b = pd.to_numeric(
                df[feature_b],
                errors="coerce",
            ).to_numpy(dtype=float)

            if np.allclose(
                series_a,
                series_b,
                equal_nan=True,
            ):
                duplicates.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "status": "EXACT_DUPLICATE",
                    }
                )

    return pd.DataFrame(
        duplicates,
        columns=[
            "feature_a",
            "feature_b",
            "status",
        ],
    )


def main() -> None:
    df = load_data()
    validate_feature_columns(df)

    audit = audit_features(df)
    correlation_matrix, high_correlations = (
        calculate_correlations(df)
    )
    exact_duplicates = find_exact_duplicates(df)

    audit_path = AUDIT_DIR / "icdm_feature_audit.csv"
    correlation_path = (
        AUDIT_DIR
        / "icdm_feature_correlation_matrix.csv"
    )
    high_correlation_path = (
        AUDIT_DIR
        / "icdm_high_correlations.csv"
    )
    duplicate_path = (
        AUDIT_DIR
        / "icdm_exact_duplicate_features.csv"
    )
    manifest_path = (
        DESIGN_DIR
        / "feature_manifest_draft.csv"
    )

    audit.to_csv(audit_path, index=False)
    audit.to_csv(manifest_path, index=False)
    correlation_matrix.to_csv(correlation_path)
    high_correlations.to_csv(
        high_correlation_path,
        index=False,
    )
    exact_duplicates.to_csv(
        duplicate_path,
        index=False,
    )

    print("\n=== DATASET SUMMARY ===\n")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Candidate features: {len(FEATURE_SPEC)}")

    print("\n=== FEATURE AUDIT ===\n")

    display_columns = [
        "feature",
        "planned_role",
        "n_missing",
        "missing_rate",
        "variance",
        "unique_values",
        "audit_status",
        "flags",
    ]

    print(
        audit[display_columns]
        .to_string(index=False)
    )

    print(
        "\n=== HIGH CORRELATIONS "
        f"|r| >= {HIGH_CORRELATION_THRESHOLD} ===\n"
    )

    if high_correlations.empty:
        print("No high-correlation pairs detected.")
    else:
        print(
            high_correlations
            .to_string(index=False)
        )

    print("\n=== EXACT DUPLICATES ===\n")

    if exact_duplicates.empty:
        print("No exact duplicate features detected.")
    else:
        print(
            exact_duplicates
            .to_string(index=False)
        )

    print("\n=== GENERATED FILES ===\n")
    print(audit_path)
    print(correlation_path)
    print(high_correlation_path)
    print(duplicate_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
PY


