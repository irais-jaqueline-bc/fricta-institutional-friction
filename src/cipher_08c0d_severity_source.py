from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "cipher_03_null_models.py"


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    lines = TARGET.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    terms = [
        "interpretive",
        "severity",
        "matched",
        "nearly",
        "balanced_accuracy",
        "roc_auc",
        "ari",
        "nmi",
        "report",
        "OUTPUT_DIR",
    ]

    hit_indices = sorted(
        {
            i
            for i, line in enumerate(lines)
            if any(term.lower() in line.lower() for term in terms)
        }
    )

    # Merge nearby hits into compact windows.
    windows = []
    for idx in hit_indices:
        start = max(0, idx - 10)
        end = min(len(lines) - 1, idx + 18)

        if windows and start <= windows[-1][1] + 1:
            windows[-1] = (
                windows[-1][0],
                max(
                    windows[-1][1],
                    end,
                ),
            )
        else:
            windows.append((start, end))

    print("\n=== CIPHER STAGE 8C0D — EXACT SEVERITY-NULL SOURCE AUDIT ===\n")
    print("Target:", TARGET.relative_to(ROOT))
    print("Read-only. No synthetic model is fitted.\n")

    for number, (start, end) in enumerate(windows, start=1):
        print(f"=== BLOCK {number} — L{start+1}-L{end+1} ===\n")
        for j in range(start, end + 1):
            print(f"{j+1:04d}: {lines[j]}")
        print()

    print("=== REVIEW TARGET ===\n")
    print(
        "Find the literal boolean condition used to decide whether aggregate "
        "severity nearly reconstructs the discovered profiles, plus any separate "
        "matched-severity-pair flag."
    )

    print("\nGATE STATUS: STAGE_8C0D_SEVERITY_SOURCE_EXTRACTED")
    print("Do not run synthetic model-performance experiments yet.")


if __name__ == "__main__":
    main()
