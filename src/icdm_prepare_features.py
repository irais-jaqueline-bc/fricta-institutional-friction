from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
MANIFEST_PATH = PROJECT_ROOT / "icdm" / "design" / "feature_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "features"
ARCHETYPES_PATH = PROJECT_ROOT / "data" / "processed" / "friction_archetypes.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró la configuración: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No se encontró el feature manifest: {MANIFEST_PATH}")
    return pd.read_csv(MANIFEST_PATH)


def validate_feature_roles(config: dict, manifest: pd.DataFrame) -> None:
    role_map = {
        "PRIMARY": config["primary_features"],
        "VALIDATION_ONLY": config["validation_only_features"],
        "SENSITIVITY_ONLY": config["sensitivity_only_features"],
        "METADATA_ONLY": config["metadata_only_features"],
        "EXCLUDED": config["excluded_features"],
    }

    for role, configured in role_map.items():
        manifested = manifest.loc[manifest["final_role"] == role, "feature"].tolist()
        if set(configured) != set(manifested):
            raise ValueError(
                f"Inconsistencia config/manifest para {role}.\n"
                f"Config: {configured}\nManifest: {manifested}"
            )


def validate_dataset(df: pd.DataFrame, config: dict) -> None:
    id_column = config["id_column"]
    required_features = (
        config["primary_features"]
        + config["validation_only_features"]
        + config["sensitivity_only_features"]
        + config["metadata_only_features"]
    )
    required_columns = [id_column] + required_features
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise KeyError("Faltan columnas:\n- " + "\n- ".join(missing))
    if df[id_column].isna().any():
        raise ValueError("Hay IDs institucionales faltantes.")
    if df[id_column].duplicated().any():
        raise ValueError("Hay IDs institucionales duplicados.")

    model_features = set(
        config["primary_features"] + config["sensitivity_only_features"]
    )
    overlap = sorted(model_features.intersection(config["never_model_columns"]))
    if overlap:
        raise ValueError(f"Columnas prohibidas dentro del modelado: {overlap}")


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].apply(pd.to_numeric, errors="coerce")


def main() -> None:
    config = load_config()
    manifest = load_manifest()
    validate_feature_roles(config, manifest)

    input_path = PROJECT_ROOT / config["input_dataset"]
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {input_path}")

    df = pd.read_csv(input_path)
    validate_dataset(df, config)

    id_column = config["id_column"]
    primary_features = config["primary_features"]
    sensitivity_features = primary_features + config["sensitivity_only_features"]
    validation_features = config["validation_only_features"]

    primary = numeric_frame(df, primary_features)
    if primary.isna().any().any():
        missing = primary.isna().sum()
        raise ValueError(
            "La matriz primaria contiene faltantes:\n"
            + missing[missing > 0].to_string()
        )

    out_of_range = {
        column: int(((primary[column] < 0) | (primary[column] > 1)).sum())
        for column in primary.columns
    }
    out_of_range = {k: v for k, v in out_of_range.items() if v > 0}
    if out_of_range:
        raise ValueError(f"Valores fuera de [0,1]: {out_of_range}")

    primary_raw = pd.concat(
        [df[[id_column]].reset_index(drop=True), primary.reset_index(drop=True)],
        axis=1,
    )

    scaler = StandardScaler()
    standardized = scaler.fit_transform(primary)
    primary_standardized = pd.DataFrame(standardized, columns=primary_features)
    primary_standardized.insert(0, id_column, df[id_column].to_numpy())

    scaler_parameters = pd.DataFrame(
        {
            "feature": primary_features,
            "mean_full_dataset": scaler.mean_,
            "scale_full_dataset": scaler.scale_,
            "variance_full_dataset": scaler.var_,
            "note": (
                "Full-dataset descriptive scaler only; refit preprocessing "
                "inside every resampling run."
            ),
        }
    )

    sensitivity = numeric_frame(df, sensitivity_features)
    if sensitivity.isna().any().any():
        missing = sensitivity.isna().sum()
        raise ValueError(
            "La matriz de sensibilidad contiene faltantes:\n"
            + missing[missing > 0].to_string()
        )
    sensitivity_output = pd.concat(
        [df[[id_column]].reset_index(drop=True), sensitivity.reset_index(drop=True)],
        axis=1,
    )

    validation_all = df[[id_column] + validation_features].copy()
    for feature in validation_features:
        validation_all[feature] = pd.to_numeric(
            validation_all[feature], errors="coerce"
        )
    validation_complete = validation_all.dropna().copy()

    candidate_metadata = [
        id_column,
        "state",
        "institution_type",
        "children_served",
        "staff_size",
    ] + config["metadata_only_features"]
    metadata_columns = [column for column in candidate_metadata if column in df.columns]
    metadata = df[metadata_columns].copy()

    if ARCHETYPES_PATH.exists():
        archetypes = pd.read_csv(ARCHETYPES_PATH)
        if (
            id_column in archetypes.columns
            and "friction_archetype" in archetypes.columns
        ):
            metadata = metadata.merge(
                archetypes[[id_column, "friction_archetype"]],
                on=id_column,
                how="left",
                validate="one_to_one",
            )

    primary_raw.to_csv(OUTPUT_DIR / "X_primary_raw.csv", index=False)
    primary_standardized.to_csv(
        OUTPUT_DIR / "X_primary_standardized_full.csv", index=False
    )
    scaler_parameters.to_csv(
        OUTPUT_DIR / "full_dataset_scaler_parameters.csv", index=False
    )
    sensitivity_output.to_csv(
        OUTPUT_DIR / "X_sensitivity_augmented_raw.csv", index=False
    )
    validation_all.to_csv(OUTPUT_DIR / "validation_variables_all_rows.csv", index=False)
    validation_complete.to_csv(
        OUTPUT_DIR / "validation_variables_complete_case.csv", index=False
    )
    metadata.to_csv(OUTPUT_DIR / "institution_metadata.csv", index=False)

    standardized_features = primary_standardized[primary_features]
    report = {
        "status": "PREPROCESSING_COMPLETE",
        "input": {
            "path": str(input_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(input_path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "unique_institution_ids": int(df[id_column].nunique()),
        },
        "design_files": {
            "experiment_config_sha256": sha256_file(CONFIG_PATH),
            "feature_manifest_sha256": sha256_file(MANIFEST_PATH),
        },
        "primary_matrix": {
            "rows": int(len(primary_raw)),
            "features": int(len(primary_features)),
            "feature_order": primary_features,
            "missing_values": int(primary.isna().sum().sum()),
            "raw_range_min": float(primary.min().min()),
            "raw_range_max": float(primary.max().max()),
            "standardized_max_absolute_mean": float(
                standardized_features.mean().abs().max()
            ),
            "standardized_min_std_ddof0": float(
                standardized_features.std(ddof=0).min()
            ),
            "standardized_max_std_ddof0": float(
                standardized_features.std(ddof=0).max()
            ),
        },
        "sensitivity_matrix": {
            "rows": int(len(sensitivity_output)),
            "features": int(len(sensitivity_features)),
        },
        "validation": {
            "features": validation_features,
            "all_rows": int(len(validation_all)),
            "complete_case_rows": int(len(validation_complete)),
            "missing_rows": int(len(validation_all) - len(validation_complete)),
            "imputation_used": False,
        },
        "metadata": {
            "rows": int(len(metadata)),
            "columns": metadata.columns.tolist(),
        },
        "important_note": (
            "X_primary_standardized_full.csv is only for the full-data descriptive "
            "solution. StandardScaler and PCA must be refitted inside every "
            "subsampling/stability iteration."
        ),
    }

    report_path = OUTPUT_DIR / "preprocessing_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== PREPROCESSING SUMMARY ===\n")
    print(f"Institutions: {report['input']['rows']}")
    print(f"Primary features: {report['primary_matrix']['features']}")
    print(f"Primary missing values: {report['primary_matrix']['missing_values']}")
    print(
        "Primary raw range: "
        f"[{report['primary_matrix']['raw_range_min']}, "
        f"{report['primary_matrix']['raw_range_max']}]"
    )
    print(
        "Validation complete cases: "
        f"{report['validation']['complete_case_rows']} / "
        f"{report['validation']['all_rows']}"
    )
    print(
        "Standardized max |mean|: "
        f"{report['primary_matrix']['standardized_max_absolute_mean']:.3e}"
    )
    print(
        "Standardized std range (ddof=0): "
        f"[{report['primary_matrix']['standardized_min_std_ddof0']:.6f}, "
        f"{report['primary_matrix']['standardized_max_std_ddof0']:.6f}]"
    )

    print("\n=== GENERATED FILES ===\n")
    for name in [
        "X_primary_raw.csv",
        "X_primary_standardized_full.csv",
        "X_sensitivity_augmented_raw.csv",
        "validation_variables_all_rows.csv",
        "validation_variables_complete_case.csv",
        "institution_metadata.csv",
        "full_dataset_scaler_parameters.csv",
        "preprocessing_report.json",
    ]:
        print(OUTPUT_DIR / name)

    print(
        "\nGATE STATUS: FEATURE PREPARATION COMPLETE. "
        "Next step is PCA, after reviewing this report."
    )


if __name__ == "__main__":
    main()
