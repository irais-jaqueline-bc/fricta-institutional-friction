from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cipher" / "outputs" / "audit"

TARGET_COUNTS = {
    "NGO": 27,
    "PUBLIC": 22,
    "PRIVATE": 17,
    "MIXED": 14,
    "UNKNOWN": 1,
}

COLUMN_KEYWORDS = [
    "governance",
    "institution_type",
    "type_of_institution",
    "tipo_institucion",
    "tipo_de_institucion",
    "tipo",
    "ownership",
    "management",
    "sector",
    "regimen",
    "régimen",
    "naturaleza",
]

ID_KEYWORDS = [
    "institution_id",
    "id_institucion",
    "id_institution",
    "synthetic_id",
    "institution",
]

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
}


def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = (
        text.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_category(value) -> str | None:
    text = norm_text(value)

    if not text:
        return None

    if any(
        token in text
        for token in [
            "no estoy seguro",
            "no estoy segura",
            "unsure",
            "unknown",
            "no se",
            "no sé",
        ]
    ):
        return "UNKNOWN"

    if any(
        token in text
        for token in [
            "mixta",
            "mixed",
        ]
    ):
        return "MIXED"

    if any(
        token in text
        for token in [
            "asociacion civil",
            "asociación civil",
            "ong",
            "nonprofit",
            "non-profit",
            "civil society",
            "organizacion civil",
            "organización civil",
            "a.c.",
            "ac ",
        ]
    ):
        return "NGO"

    if any(
        token in text
        for token in [
            "publica",
            "publico",
            "public",
            "gobierno",
            "government",
        ]
    ):
        return "PUBLIC"

    if any(
        token in text
        for token in [
            "privada",
            "privado",
            "private",
        ]
    ):
        return "PRIVATE"

    return None


def read_table(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None

    return None


def candidate_id_columns(df: pd.DataFrame) -> list[str]:
    result = []

    for column in df.columns:
        normalized = norm_text(column).replace(" ", "_")
        if normalized in ID_KEYWORDS:
            result.append(column)
        elif normalized.endswith("_id") and "instit" in normalized:
            result.append(column)

    return result


def column_name_score(column: str) -> int:
    normalized = norm_text(column).replace(" ", "_")
    score = 0

    for keyword in COLUMN_KEYWORDS:
        if keyword in normalized:
            score += 3

    if "instit" in normalized:
        score += 1

    return score


def inspect_column(path: Path, df: pd.DataFrame, column: str) -> dict:
    mapped = df[column].map(canonical_category)
    mapped_non_null = mapped.dropna()

    counts = mapped_non_null.value_counts().to_dict()

    target_distance = sum(
        abs(int(counts.get(category, 0)) - target)
        for category, target in TARGET_COUNTS.items()
    )

    exact_target = (
        len(df) == 81
        and len(mapped_non_null) == 81
        and all(
            int(counts.get(category, 0)) == target
            for category, target in TARGET_COUNTS.items()
        )
    )

    category_coverage = float(len(mapped_non_null) / len(df)) if len(df) else 0.0

    unique_values = [
        str(value) for value in df[column].dropna().astype(str).unique().tolist()[:20]
    ]

    return {
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "column": str(column),
        "column_name_score": column_name_score(str(column)),
        "mapped_count": int(len(mapped_non_null)),
        "category_coverage": category_coverage,
        "target_distance": int(target_distance),
        "exact_target_match": bool(exact_target),
        "ngo": int(counts.get("NGO", 0)),
        "public": int(counts.get("PUBLIC", 0)),
        "private": int(counts.get("PRIVATE", 0)),
        "mixed": int(counts.get("MIXED", 0)),
        "unknown": int(counts.get("UNKNOWN", 0)),
        "id_candidates": candidate_id_columns(df),
        "unique_values_preview": unique_values,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    candidates = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue

        if path.suffix.lower() not in {
            ".csv",
            ".tsv",
            ".txt",
            ".xlsx",
            ".xls",
            ".parquet",
        }:
            continue

        # Avoid rescanning CIPHER result matrices, which are not source metadata.
        if "cipher/outputs" in str(path.relative_to(ROOT)).replace("\\", "/"):
            continue

        df = read_table(path)
        if df is None or df.empty:
            continue

        for column in df.columns:
            name_score = column_name_score(str(column))

            # Inspect obvious governance columns, and any low-cardinality string column
            # in an 81-row table because the raw survey may use an unexpected Spanish label.
            series = df[column]
            low_cardinality = series.nunique(dropna=True) <= 12

            if name_score > 0 or (len(df) == 81 and low_cardinality):
                result = inspect_column(path, df, column)

                if result["mapped_count"] > 0 or result["column_name_score"] > 0:
                    candidates.append(result)

    candidates.sort(
        key=lambda row: (
            not row["exact_target_match"],
            row["target_distance"],
            -row["category_coverage"],
            -row["column_name_score"],
        )
    )

    json_path = OUT / "stage3_governance_locator.json"
    json_path.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 3 — GOVERNANCE LOCATOR ===\n")

    if not candidates:
        print("No plausible governance columns were found.")
        print("\nGATE STATUS: GOVERNANCE_SOURCE_NOT_FOUND")
        return

    top = candidates[:15]

    for idx, row in enumerate(top, start=1):
        print(f"[{idx}] {row['file']}")
        print(f"    column: {row['column']}")
        print(f"    rows: {row['rows']}")
        print(
            "    mapped counts:",
            {
                "NGO": row["ngo"],
                "PUBLIC": row["public"],
                "PRIVATE": row["private"],
                "MIXED": row["mixed"],
                "UNKNOWN": row["unknown"],
            },
        )
        print(f"    mapped_count: {row['mapped_count']}")
        print(f"    target_distance: {row['target_distance']}")
        print(f"    exact_target_match: {row['exact_target_match']}")
        print(f"    id_candidates: {row['id_candidates']}")
        print(f"    values: {row['unique_values_preview']}")
        print()

    exact = [row for row in candidates if row["exact_target_match"]]

    if exact:
        print("EXACT TARGET MATCHES:", len(exact))
        for row in exact:
            print(
                f"- {row['file']} :: {row['column']} "
                f"(ID candidates: {row['id_candidates']})"
            )
        print("\nGATE STATUS: GOVERNANCE_SOURCE_FOUND")
    else:
        print(
            "No exact 27/22/17/14/1 match was found. "
            "Review the highest-ranked candidate rather than guessing."
        )
        print("\nGATE STATUS: GOVERNANCE_SOURCE_REVIEW_REQUIRED")


if __name__ == "__main__":
    main()
