from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "audit"
    / "stage8c0_real_pipeline_semantics_report.json"
)

SEARCH_DIRS = [
    ROOT / "src",
    ROOT / "cipher" / "design",
    ROOT / "cipher" / "outputs",
    ROOT / "icdm" / "design",
    ROOT / "icdm" / "outputs",
]

KEY_TERMS = [
    "severity_nearly_reconstructs",
    "matched pairs",
    "matched_pairs",
    "governance",
    "cramer",
    "permutation",
    "balanced_accuracy",
    "model selection",
    "model_selection",
    "selected_model",
    "minimum_cluster_size",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "stability",
    "reference_profile_probability",
    "family_consistency",
    "normalized_entropy",
    "membership_margin",
    "core",
    "halo",
    "boundary",
]

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".csv",
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )


def compact_line(line: str, limit: int = 500) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) > limit:
        return line[:limit] + " ..."
    return line


def inspect_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(load_text(path))
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    interesting = {}

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_path = f"{prefix}.{key}" if prefix else str(key)
                lowered = key_path.lower()

                if any(term in lowered for term in KEY_TERMS):
                    if isinstance(child, (str, int, float, bool)) or child is None:
                        interesting[key_path] = child
                    elif isinstance(child, list) and len(child) <= 20:
                        interesting[key_path] = child

                walk(child, key_path)

        elif isinstance(value, list):
            for index, child in enumerate(value[:50]):
                walk(child, f"{prefix}[{index}]")

    walk(obj)

    if not interesting:
        return None

    return {
        "path": relative(path),
        "interesting_keys": interesting,
    }


def inspect_text(path: Path) -> dict[str, Any] | None:
    text = load_text(path)
    lines = text.splitlines()

    matches = []

    for line_no, line in enumerate(lines, start=1):
        lowered = line.lower()

        matched_terms = [term for term in KEY_TERMS if term in lowered]

        if matched_terms:
            matches.append(
                {
                    "line": line_no,
                    "terms": matched_terms,
                    "text": compact_line(line),
                }
            )

    if not matches:
        return None

    return {
        "path": relative(path),
        "matches": matches[:200],
    }


def score_record(record: dict[str, Any]) -> int:
    path = record["path"].lower()
    score = 0

    for token, weight in [
        ("stage3", 10),
        ("03", 4),
        ("severity", 8),
        ("governance", 8),
        ("selection", 7),
        ("cluster", 5),
        ("uncertainty", 7),
        ("certainty", 5),
        ("ensemble", 3),
        ("design", 2),
        ("audit", 2),
    ]:
        if token in path:
            score += weight

    if "interesting_keys" in record:
        score += min(
            20,
            len(record["interesting_keys"]),
        )

    if "matches" in record:
        score += min(
            20,
            len(record["matches"]),
        )

    return score


def main() -> None:
    json_records = []
    text_records = []

    seen = set()

    for search_dir in SEARCH_DIRS:
        if not search_dir.exists():
            continue

        for path in search_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            if path.suffix.lower() == ".json":
                record = inspect_json(path)
                if record is not None:
                    record["score"] = score_record(record)
                    json_records.append(record)

            record = inspect_text(path)
            if record is not None:
                record["score"] = score_record(record)
                text_records.append(record)

    json_records.sort(
        key=lambda row: (
            -row["score"],
            row["path"],
        )
    )
    text_records.sort(
        key=lambda row: (
            -row["score"],
            row["path"],
        )
    )

    report = {
        "purpose": (
            "Read-only extraction of the already-used real-pipeline semantics "
            "needed to freeze Stage 8 synthetic evaluator rules before any "
            "synthetic model-performance result is observed."
        ),
        "json_candidates": json_records[:20],
        "text_candidates": text_records[:20],
        "gate_status": "STAGE_8C0_REAL_PIPELINE_SEMANTICS_EXTRACTED",
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8C0 — REAL PIPELINE SEMANTICS EXTRACTION ===\n")
    print("This stage is read-only. It does not fit a synthetic model.")

    print("\n=== TOP JSON / REPORT CANDIDATES ===\n")

    for idx, record in enumerate(
        json_records[:12],
        start=1,
    ):
        print(f"[{idx:02d}] score={record['score']:02d} {record['path']}")

        items = list(record["interesting_keys"].items())[:20]

        for key, value in items:
            print(f"     {key} = {value}")

    print("\n=== TOP SOURCE / TEXT CANDIDATES ===\n")

    for idx, record in enumerate(
        text_records[:12],
        start=1,
    ):
        print(f"[{idx:02d}] score={record['score']:02d} {record['path']}")

        for match in record["matches"][:15]:
            print(f"     L{match['line']}: {match['text']}")

    print("\n=== WHAT WE NEED BEFORE STAGE 8C ===\n")
    print("  1) exact real-pipeline model-selection rule / gates;")
    print("  2) exact severity-null flag rule;")
    print("  3) exact governance-null flag rule;")
    print("  4) exact primary uncertainty quantity used for synthetic AUROC;")
    print(
        "  5) exact conditions under which the synthetic pipeline is allowed "
        "to make a 'stable configurational profile' claim."
    )

    print("\nGATE STATUS: STAGE_8C0_REAL_PIPELINE_SEMANTICS_EXTRACTED")
    print("Do not run synthetic model-performance experiments yet.")


if __name__ == "__main__":
    main()
