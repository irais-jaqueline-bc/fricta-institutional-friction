from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STAGE6_FREEZE = ROOT / "cipher" / "design" / "stage6_ensemble_robustness_freeze.json"

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "audit"
MANIFEST_CSV = OUTPUT_DIR / "stage6b0_ensemble_artifact_manifest.csv"
REPORT_JSON = OUTPUT_DIR / "stage6b0_ensemble_artifact_report.json"

KEYWORDS = (
    "ensemble",
    "member",
    "stage1",
    "stage_1",
    "stage4",
    "stage_4",
    "counterfactual",
    "inductive",
    "ward",
    "kmeans",
)

SEARCH_ROOTS = (
    ROOT / "cipher" / "outputs",
    ROOT / "cipher" / "design",
    ROOT / "icdm" / "outputs",
    ROOT / "icdm" / "design",
)

SUPPORTED_TEXT = {".csv", ".json", ".jsonl", ".parquet"}
SUPPORTED_BINARY = {".joblib", ".pkl", ".pickle", ".npz", ".npy"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def relevant(path: Path) -> bool:
    lower = str(path.relative_to(ROOT)).lower()
    return any(keyword in lower for keyword in KEYWORDS)


def inspect_csv(path: Path) -> dict[str, Any]:
    try:
        sample = pd.read_csv(path, nrows=5)
        try:
            row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        except Exception:
            row_count = None
        return {
            "readable": True,
            "row_count": row_count,
            "columns": list(sample.columns),
        }
    except Exception as exc:
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}


def inspect_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return {
                "readable": True,
                "json_type": "dict",
                "top_level_keys": list(obj.keys())[:100],
            }
        if isinstance(obj, list):
            return {
                "readable": True,
                "json_type": "list",
                "list_length": len(obj),
            }
        return {
            "readable": True,
            "json_type": type(obj).__name__,
        }
    except Exception as exc:
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}


def inspect_parquet(path: Path) -> dict[str, Any]:
    try:
        frame = pd.read_parquet(path)
        return {
            "readable": True,
            "row_count": len(frame),
            "columns": list(frame.columns),
        }
    except Exception as exc:
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}


def inspect_path(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    record: dict[str, Any] = {
        "relative_path": str(path.relative_to(ROOT)),
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
    }

    if suffix == ".csv":
        record.update(inspect_csv(path))
    elif suffix == ".json":
        record.update(inspect_json(path))
    elif suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as handle:
                first = handle.readline()
            json.loads(first) if first else None
            record.update({"readable": True, "json_type": "jsonl"})
        except Exception as exc:
            record.update({"readable": False, "error": f"{type(exc).__name__}: {exc}"})
    elif suffix == ".parquet":
        record.update(inspect_parquet(path))
    elif suffix in SUPPORTED_BINARY:
        record.update({"readable": False, "binary_artifact": True})
    else:
        record.update({"readable": False, "unsupported_suffix": True})

    return record


def score_candidate(record: dict[str, Any]) -> int:
    path = record["relative_path"].lower()
    score = 0

    for token, weight in [
        ("stage4", 10),
        ("counterfactual", 8),
        ("ensemble", 8),
        ("member", 6),
        ("stage1", 5),
        ("inductive", 4),
        ("ward", 2),
        ("kmeans", 2),
    ]:
        if token in path:
            score += weight

    columns = [str(c).lower() for c in record.get("columns", [])]
    keys = [str(k).lower() for k in record.get("top_level_keys", [])]

    for token, weight in [
        ("member_id", 10),
        ("family", 7),
        ("algorithm", 6),
        ("representation", 6),
        ("feature", 4),
        ("sample", 4),
        ("centroid", 4),
        ("scaler", 4),
        ("pca", 4),
        ("fidelity", 3),
        ("eligible", 3),
    ]:
        if any(token in item for item in columns + keys):
            score += weight

    return score


def main() -> None:
    freeze = load_json(STAGE6_FREEZE)

    if freeze.get("gate_status") != "PASS_STAGE_6A_DESIGN_FREEZE":
        raise ValueError("Stage 6A design freeze has not passed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []

    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue

        for path in search_root.rglob("*"):
            if (
                path.is_file()
                and relevant(path)
                and path.suffix.lower() in (SUPPORTED_TEXT | SUPPORTED_BINARY)
            ):
                files.append(path)

    files = sorted(set(files))
    records = []

    for path in files:
        record = inspect_path(path)
        record["candidate_score"] = score_candidate(record)
        records.append(record)

    records.sort(key=lambda item: (-item["candidate_score"], item["relative_path"]))

    manifest_rows = []
    for record in records:
        manifest_rows.append(
            {
                "candidate_score": record["candidate_score"],
                "relative_path": record["relative_path"],
                "suffix": record["suffix"],
                "size_bytes": record["size_bytes"],
                "readable": record.get("readable"),
                "row_count": record.get("row_count"),
                "columns_or_keys": json.dumps(
                    record.get("columns", record.get("top_level_keys", [])),
                    ensure_ascii=False,
                ),
                "binary_artifact": record.get("binary_artifact", False),
                "error": record.get("error", ""),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False)

    REPORT_JSON.write_text(
        json.dumps(
            {
                "files_scanned": len(files),
                "top_candidates": records[:20],
                "gate_status": "STAGE_6B0_ARTIFACT_LOCATOR_COMPLETE",
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 6B0 — ENSEMBLE ARTIFACT LOCATOR ===\n")
    print("Candidate files found:", len(files))
    print("\n=== TOP 20 ARTIFACT CANDIDATES ===\n")

    for idx, record in enumerate(records[:20], start=1):
        print(
            f"[{idx:02d}] score={record['candidate_score']:02d} "
            f"{record['relative_path']}"
        )

        if "row_count" in record:
            print(
                f"     rows={record.get('row_count')} "
                f"columns={record.get('columns')}"
            )
        elif "top_level_keys" in record:
            print(f"     keys={record.get('top_level_keys')}")
        elif record.get("binary_artifact"):
            print("     binary artifact")
        elif record.get("error"):
            print(f"     read error={record['error']}")

    print("\n=== TARGETS FOR STAGE 6B ===\n")
    print("We need artifacts that can reconstruct or audit:")
    print("  1) all 1000 Stage-1 ensemble member definitions;")
    print("  2) the 984 Stage-4 counterfactual-eligible member IDs;")
    print("  3) each member's sampled rows and 11-feature subset;")
    print("  4) representation/algorithm family and fitted prediction geometry;")
    print("  5) stored Ward inductive fidelity / eligibility.")

    print("\nGATE STATUS: STAGE_6B0_ARTIFACT_LOCATOR_COMPLETE")
    print("Do not evaluate counterfactual candidates yet.")


if __name__ == "__main__":
    main()
