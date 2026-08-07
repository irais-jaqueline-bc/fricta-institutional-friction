from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8c0b_targeted_semantics_report.json"
)

SEARCH_ROOTS = [
    ROOT / "src",
    ROOT / "cipher" / "design",
    ROOT / "icdm" / "design",
]

TARGETS = {
    "MODEL_SELECTION": [
        "selected_model",
        "model_selection",
        "selection rule",
        "candidate_score",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari_median",
        "jaccard",
        "minimum_cluster_size",
    ],
    "SEVERITY_NULL": [
        "severity_nearly_reconstructs",
        "severity",
        "matched_pairs",
        "matched pairs",
        "balanced_accuracy",
        "roc_auc",
        "ari",
        "nmi",
    ],
    "GOVERNANCE_NULL": [
        "governance_nearly_reconstructs_profiles",
        "strong_governance_association",
        "cramers_v",
        "cramer",
        "permutation_p",
        "balanced_accuracy",
    ],
    "UNCERTAINTY": [
        "normalized_entropy",
        "reference_profile_probability",
        "family_consistency",
        "membership_margin",
        "consensus_gap",
        "certainty_class",
    ],
    "STABLE_PROFILE_CLAIM": [
        "stable profile",
        "stable_profile",
        "configurational",
        "minimum_cluster_size",
        "stability",
        "silhouette",
        "false_configurational_profile_claim",
    ],
}

SUFFIXES = {".py", ".json", ".md", ".txt"}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def find_candidate_files() -> list[Path]:
    files = []
    for base in SEARCH_ROOTS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUFFIXES:
                files.append(path)
    return sorted(set(files))


def score_file(path: Path, text: str, terms: list[str]) -> int:
    lower = text.lower()
    score = sum(lower.count(term.lower()) for term in terms)
    name = path.name.lower()

    for token, weight in [
        ("severity", 30),
        ("governance", 30),
        ("selection", 20),
        ("cluster", 10),
        ("certainty", 20),
        ("uncertainty", 20),
        ("stage3", 15),
        ("stage2", 10),
        ("icdm", 5),
    ]:
        if token in name or token in str(path.parent).lower():
            score += weight

    return score


def line_matches(text: str, terms: list[str]) -> list[int]:
    indices = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        lower = line.lower()
        if any(term.lower() in lower for term in terms):
            indices.append(idx)
    return indices


def context_blocks(
    path: Path,
    text: str,
    terms: list[str],
    before: int = 12,
    after: int = 22,
    max_blocks: int = 12,
):
    lines = text.splitlines()
    hits = line_matches(text, terms)

    blocks = []
    occupied = []

    for idx in hits:
        start = max(0, idx - before)
        end = min(len(lines), idx + after + 1)

        if any(not (end < s or start > e) for s, e in occupied):
            continue

        occupied.append((start, end))
        block_lines = []
        for j in range(start, end):
            block_lines.append(f"{j+1:04d}: {lines[j]}")

        blocks.append(
            {
                "path": rel(path),
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(block_lines),
            }
        )

        if len(blocks) >= max_blocks:
            break

    return blocks


def main() -> None:
    files = find_candidate_files()

    report = {
        "purpose": (
            "Targeted read-only extraction of exact pre-existing evaluator semantics "
            "needed before Stage 8C can be frozen."
        ),
        "targets": {},
        "gate_status": "STAGE_8C0B_TARGETED_SEMANTICS_EXTRACTED",
    }

    print("\n=== CIPHER STAGE 8C0B — TARGETED EVALUATOR SEMANTICS ===\n")
    print("Read-only. No synthetic model is fitted.\n")

    for target_name, terms in TARGETS.items():
        scored = []

        for path in files:
            text = read_text(path)
            score = score_file(path, text, terms)

            if score > 0 and any(term.lower() in text.lower() for term in terms):
                scored.append((score, path, text))

        scored.sort(
            key=lambda item: (
                -item[0],
                rel(item[1]),
            )
        )

        chosen = scored[:6]
        target_records = []

        print(f"=== {target_name} ===\n")

        if not chosen:
            print("No matching source/config file found.\n")
            report["targets"][target_name] = []
            continue

        for rank, (score, path, text) in enumerate(chosen, start=1):
            print(f"[{rank}] score={score} {rel(path)}")

            blocks = context_blocks(
                path,
                text,
                terms,
                before=12,
                after=22,
                max_blocks=5,
            )

            target_records.append(
                {
                    "score": score,
                    "path": rel(path),
                    "blocks": blocks,
                }
            )

            for block in blocks:
                print(
                    f"\n--- {block['path']} "
                    f"L{block['start_line']}-L{block['end_line']} ---"
                )
                print(block["text"])

            print()

        report["targets"][target_name] = target_records

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== DECISION RULE ===\n")
    print(
        "Stage 8C evaluator freeze is allowed only after the exact operational "
        "rules for model selection, severity flag, governance flag, and primary "
        "uncertainty score are visible in this output."
    )
    print(
        "If no pre-existing stable-profile claim gate exists, Stage 8C may define "
        "one prospectively from the already frozen Stage-8 criteria, but only after "
        "documenting that absence."
    )

    print("\nGATE STATUS: STAGE_8C0B_TARGETED_SEMANTICS_EXTRACTED")
    print("Do not run synthetic model-performance experiments yet.")


if __name__ == "__main__":
    main()
