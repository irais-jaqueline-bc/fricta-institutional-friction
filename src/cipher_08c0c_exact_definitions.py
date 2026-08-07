from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8c0c_exact_flag_definitions.json"
)

TARGETS = {
    "SEVERITY": [
        "severity_nearly_reconstructs",
        "matched_pairs",
        "matched_pair",
    ],
    "GOVERNANCE": [
        "strong_governance_association",
        "governance_nearly_reconstructs_profiles",
    ],
    "UNCERTAINTY": [
        "normalized_entropy",
        "reference_profile_probability",
        "family_consistency",
        "membership_margin",
        "consensus_gap",
    ],
}

SEARCH_GLOBS = [
    "src/cipher_02*.py",
    "src/cipher_03*.py",
    "src/cipher*certainty*.py",
    "src/cipher*severity*.py",
    "src/cipher*governance*.py",
]

REPORT_CANDIDATES = [
    "cipher/outputs/null_models/severity_null_report.json",
    "cipher/outputs/null_models/governance_null_report.json",
    "cipher/outputs/certainty/certainty_report.json",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def candidate_files() -> list[Path]:
    found = set()

    for pattern in SEARCH_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                found.add(path)

    return sorted(found)


def exact_context(text: str, term: str, radius: int = 8) -> list[dict]:
    lines = text.splitlines()
    results = []

    for idx, line in enumerate(lines):
        if term.lower() not in line.lower():
            continue

        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)

        results.append(
            {
                "term": term,
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(f"{j+1:04d}: {lines[j]}" for j in range(start, end)),
            }
        )

    return results


def enclosing_ast_blocks(path: Path, terms: list[str]) -> list[dict]:
    source = read(path)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    results = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Assign,
                ast.AnnAssign,
                ast.If,
            ),
        ):
            continue

        segment = ast.get_source_segment(source, node)
        if not segment:
            continue

        matched = [term for term in terms if term.lower() in segment.lower()]

        if not matched:
            continue

        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)

        if start is None or end is None:
            continue

        # Keep blocks small. Large functions are not dumped wholesale.
        if end - start + 1 > 45:
            continue

        results.append(
            {
                "node_type": type(node).__name__,
                "name": getattr(node, "name", ""),
                "terms": matched,
                "start_line": start,
                "end_line": end,
                "text": "\n".join(
                    f"{j+1:04d}: {lines[j]}" for j in range(start - 1, end)
                ),
            }
        )

    # De-duplicate identical line spans.
    unique = {}
    for row in results:
        key = (
            row["start_line"],
            row["end_line"],
            tuple(row["terms"]),
        )
        unique[key] = row

    return list(unique.values())


def print_report_json(path: Path) -> None:
    if not path.exists():
        print(f"{rel(path)}: MISSING")
        return

    obj = json.loads(read(path))

    print(f"{rel(path)}:")

    if "interpretive_flags" in obj:
        print(
            "  interpretive_flags =",
            json.dumps(
                obj["interpretive_flags"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    if "thresholds" in obj:
        print(
            "  thresholds =",
            json.dumps(
                obj["thresholds"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    for key in [
        "matched_severity_pairs",
        "matched_pair_count",
        "cramers_v_bias_corrected",
        "permutation_p",
        "balanced_accuracy",
        "reference_profile_probability",
        "family_consistency",
    ]:
        if key in obj:
            print(
                f"  {key} =",
                json.dumps(
                    obj[key],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )


def main() -> None:
    files = candidate_files()

    report = {
        "purpose": (
            "Minimal read-only extraction of exact flag/uncertainty definitions "
            "without dumping unrelated repository content."
        ),
        "files_scanned": [rel(path) for path in files],
        "targets": {},
        "gate_status": "STAGE_8C0C_EXACT_DEFINITIONS_EXTRACTED",
    }

    print("\n=== CIPHER STAGE 8C0C — EXACT FLAG / UNCERTAINTY DEFINITIONS ===\n")
    print("This is read-only. No synthetic model is fitted.")
    print(
        "Files scanned:",
        [rel(path) for path in files],
    )

    for target_name, terms in TARGETS.items():
        print(f"\n=== {target_name} ===\n")

        rows = []

        for path in files:
            source = read(path)

            contexts = []
            for term in terms:
                contexts.extend(
                    exact_context(
                        source,
                        term,
                        radius=8,
                    )
                )

            ast_blocks = enclosing_ast_blocks(
                path,
                terms,
            )

            if not contexts and not ast_blocks:
                continue

            rows.append(
                {
                    "path": rel(path),
                    "contexts": contexts,
                    "ast_blocks": ast_blocks,
                }
            )

            print(f"FILE: {rel(path)}")

            printed_spans = set()

            # Prefer compact AST blocks because they often show the complete
            # boolean assignment or function definition.
            for block in ast_blocks:
                span = (
                    block["start_line"],
                    block["end_line"],
                )

                if span in printed_spans:
                    continue

                printed_spans.add(span)

                print(
                    f"\n--- AST {block['node_type']} "
                    f"L{block['start_line']}-L{block['end_line']} ---"
                )
                print(block["text"])

            for context in contexts:
                span = (
                    context["start_line"],
                    context["end_line"],
                )

                # Avoid reprinting a context fully contained in an AST block.
                if any(a <= span[0] and span[1] <= b for a, b in printed_spans):
                    continue

                print(
                    f"\n--- CONTEXT term={context['term']} "
                    f"L{context['start_line']}-L{context['end_line']} ---"
                )
                print(context["text"])

            print()

        report["targets"][target_name] = rows

    print("\n=== EXISTING REPORT FLAGS / THRESHOLDS ===\n")

    for relative_path in REPORT_CANDIDATES:
        print_report_json(ROOT / relative_path)
        print()

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

    print("=== REQUIRED REVIEW BEFORE STAGE 8C ===")
    print(
        "We need the literal boolean conditions for the severity and governance "
        "flags, plus the formulas for the continuous uncertainty quantities."
    )
    print(
        "If the code contains no pre-existing 'primary uncertainty score', "
        "Stage 8C will explicitly choose one prospectively before synthetic results."
    )

    print("\nGATE STATUS: STAGE_8C0C_EXACT_DEFINITIONS_EXTRACTED")
    print("Do not run synthetic model-performance experiments yet.")


if __name__ == "__main__":
    main()
