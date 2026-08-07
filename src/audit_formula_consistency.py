from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORED_PATH = PROJECT_ROOT / "data" / "processed" / "fricta_scored.csv"

ARCHETYPES_PATH = PROJECT_ROOT / "data" / "processed" / "friction_archetypes.csv"

OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def mean_available(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Reproduce pandas row-wise mean while ignoring missing values."""
    return df[columns].mean(axis=1)


def max_absolute_error(
    observed: pd.Series,
    expected: pd.Series,
) -> float:
    difference = (
        pd.to_numeric(observed, errors="coerce")
        - pd.to_numeric(expected, errors="coerce")
    ).abs()

    return float(difference.max())


def classify_current_code(row: pd.Series) -> str:
    scores = {
        "ICI": row["ICI"],
        "OCI": row["OCI"],
        "OLI": row["OLI"],
        "HCARI": row["HCARI"],
    }

    high_branches = sum(score >= 0.60 for score in scores.values() if pd.notna(score))

    if high_branches >= 3:
        return "Multi-Constraint"

    dominant = max(scores, key=scores.get)

    mapping = {
        "ICI": "Infrastructure-Limited",
        "OCI": "Organizationally-Limited",
        "OLI": "Operationally-Limited",
        "HCARI": "Human-Capacity-Limited",
    }

    return mapping[dominant]


def classify_paper_rule(row: pd.Series) -> str:
    scores = {
        "ICI": row["ICI"],
        "OCI": row["OCI"],
        "OLI": row["OLI"],
        "HCARI": row["HCARI"],
    }

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    highest_name, highest_score = ordered[0]
    _, second_score = ordered[1]

    if abs(highest_score - second_score) <= 0.10:
        return "Multi-Constraint"

    mapping = {
        "ICI": "Infrastructure-Limited",
        "OCI": "Organizationally-Limited",
        "OLI": "Operationally-Limited",
        "HCARI": "Human-Capacity-Limited",
    }

    return mapping[highest_name]


def main() -> None:
    if not SCORED_PATH.exists():
        raise FileNotFoundError(f"No se encontró:\n{SCORED_PATH}")

    df = pd.read_csv(SCORED_PATH)

    # ========================================================
    # VERSION A: formulas currently implemented in scoring.py
    # ========================================================

    df["ICI_current_code"] = mean_available(
        df,
        [
            "device_constraint",
            "internet_stability_constraint",
            "digital_tool_variety_constraint",
        ],
    )

    df["OCI_current_code"] = mean_available(
        df,
        [
            "recording_system_constraint",
            "admin_time_load_constraint",
            "administrative_disorganization_constraint",
            "system_change_resistance_constraint",
        ],
    )

    df["OLI_current_code"] = mean_available(
        df,
        [
            "digital_usage_constraint_score",
            "implementation_difficulty_constraint",
            "previous_implementation_constraint",
        ],
    )

    df["HCARI_current_code"] = mean_available(
        df,
        [
            "time_constraint_score",
            "staffing_constraint_score",
            "training_deficit_score",
            "resource_constraint_score",
            "willingness_constraint_score",
            "perceived_utility_constraint",
            "pilot_openness_constraint",
        ],
    )

    df["AFS_current_code"] = (
        0.25 * df["ICI_current_code"]
        + 0.30 * df["OCI_current_code"]
        + 0.20 * df["OLI_current_code"]
        + 0.25 * df["HCARI_current_code"]
    )

    # ========================================================
    # VERSION B: formulas described in the submitted paper
    # ========================================================

    df["ICI_paper"] = mean_available(
        df,
        [
            "device_constraint",
            "internet_stability_constraint",
            "digital_tool_variety_constraint",
            "resource_constraint_score",
        ],
    )

    df["OCI_paper"] = mean_available(
        df,
        [
            "administrative_disorganization_constraint",
            "implementation_difficulty_constraint",
            "system_change_resistance_constraint",
        ],
    )

    df["OLI_paper"] = mean_available(
        df,
        [
            "admin_time_load_constraint",
            "time_constraint_score",
            "staffing_constraint_score",
        ],
    )

    df["HCARI_paper"] = mean_available(
        df,
        [
            "training_deficit_score",
            "digital_usage_constraint_score",
            "willingness_constraint_score",
        ],
    )

    df["AFS_paper"] = (
        0.30 * df["ICI_paper"]
        + 0.30 * df["OCI_paper"]
        + 0.25 * df["OLI_paper"]
        + 0.15 * df["HCARI_paper"]
    )

    comparisons = []

    for branch in ["ICI", "OCI", "OLI", "HCARI"]:
        comparisons.append(
            {
                "score": branch,
                "error_vs_current_code": max_absolute_error(
                    df[branch],
                    df[f"{branch}_current_code"],
                ),
                "error_vs_paper_formula": max_absolute_error(
                    df[branch],
                    df[f"{branch}_paper"],
                ),
            }
        )

    comparisons.append(
        {
            "score": "AFS_theoretical",
            "error_vs_current_code": max_absolute_error(
                df["AFS_theoretical"],
                df["AFS_current_code"],
            ),
            "error_vs_paper_formula": max_absolute_error(
                df["AFS_theoretical"],
                df["AFS_paper"],
            ),
        }
    )

    comparison_df = pd.DataFrame(comparisons)

    print("\n=== FORMULA CONSISTENCY AUDIT ===\n")
    print(comparison_df.to_string(index=False))

    print(
        "\nInterpretación:"
        "\n- Error cercano a 0 significa que esa versión produjo el CSV."
        "\n- El conjunto de errores más pequeño identifica la arquitectura real."
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "formula_consistency_audit.csv",
        index=False,
    )

    # ========================================================
    # Archetype-rule comparison
    # ========================================================

    df["archetype_current_code"] = df.apply(
        classify_current_code,
        axis=1,
    )

    df["archetype_paper_rule"] = df.apply(
        classify_paper_rule,
        axis=1,
    )

    if ARCHETYPES_PATH.exists():
        saved = pd.read_csv(ARCHETYPES_PATH)

        if "friction_archetype" in saved.columns:
            saved_labels = saved[["institution_id", "friction_archetype"]].rename(
                columns={"friction_archetype": "archetype_saved"}
            )

            df = df.merge(
                saved_labels,
                on="institution_id",
                how="left",
            )

            current_match = (
                df["archetype_saved"] == df["archetype_current_code"]
            ).mean()

            paper_match = (df["archetype_saved"] == df["archetype_paper_rule"]).mean()

            print("\n=== ARCHETYPE RULE AUDIT ===\n")
            print("Agreement with current Python rule: " f"{current_match:.3f}")
            print("Agreement with paper ≤ 0.10 rule: " f"{paper_match:.3f}")

            print("\nSaved archetype distribution:")
            print(df["archetype_saved"].value_counts(dropna=False).to_string())

            print("\nCurrent-code distribution:")
            print(df["archetype_current_code"].value_counts(dropna=False).to_string())

            print("\nPaper-rule distribution:")
            print(df["archetype_paper_rule"].value_counts(dropna=False).to_string())

    output_columns = [
        "institution_id",
        "ICI",
        "ICI_current_code",
        "ICI_paper",
        "OCI",
        "OCI_current_code",
        "OCI_paper",
        "OLI",
        "OLI_current_code",
        "OLI_paper",
        "HCARI",
        "HCARI_current_code",
        "HCARI_paper",
        "AFS_theoretical",
        "AFS_current_code",
        "AFS_paper",
        "archetype_current_code",
        "archetype_paper_rule",
    ]

    if "archetype_saved" in df.columns:
        output_columns.append("archetype_saved")

    df[output_columns].to_csv(
        OUTPUT_DIR / "formula_and_archetype_comparison.csv",
        index=False,
    )

    print("\nArchivos generados:")
    print(OUTPUT_DIR / "formula_consistency_audit.csv")
    print(OUTPUT_DIR / "formula_and_archetype_comparison.csv")


if __name__ == "__main__":
    main()
