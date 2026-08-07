from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FREEZE_V1 = ROOT / "cipher" / "design" / "counterfactual_method_freeze.json"
FREEZE_V2 = ROOT / "cipher" / "design" / "counterfactual_method_freeze_v2.json"
DIAG = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "smoke"
    / "smoke_search_diagnostics.csv"
)
REPORT = ROOT / "cipher" / "outputs" / "counterfactuals" / "smoke" / "smoke_report.json"
AUDIT = ROOT / "cipher" / "outputs" / "audit"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    if FREEZE_V2.exists():
        raise FileExistsError(f"Versioned amendment already exists: {FREEZE_V2}")

    freeze = load_json(FREEZE_V1)
    report = load_json(REPORT)
    d = pd.read_csv(DIAG)

    solvable = d[d["exact_solution_exists"].astype(bool)].copy()
    unsolvable = d[~d["exact_solution_exists"].astype(bool)].copy()
    existence = d["existence_matches_exact"].astype(bool)
    exact_match = solvable["beam_matches_exact_best"].astype(bool)
    mismatch = solvable[~exact_match].copy()
    gaps = pd.to_numeric(solvable["optimality_gap"], errors="coerce").dropna()
    max_gap = float(gaps.max()) if len(gaps) else 0.0

    checks = {
        "stage5a_freeze_passed": freeze.get("gate_status") == "PASS_STAGE_5A",
        "smoke_failed_as_recorded": report.get("gate_status") == "FAIL_STAGE_5B_SMOKE",
        "six_cases": len(d) == 6,
        "all_six_exact_audited": bool(d["exact_audited"].astype(bool).all()),
        "existence_agreement_6_of_6": bool(existence.all()),
        "at_least_one_solvable_case": len(solvable) > 0,
        "at_least_one_beam_exact_cost_mismatch": len(mismatch) > 0,
        "finite_exact_search_space_confirmed": bool(
            (
                pd.to_numeric(d["exact_evaluated_candidates"], errors="coerce")
                == 147291
            ).all()
        ),
    }

    if not all(checks.values()):
        for k, v in checks.items():
            print(f"[{'PASS' if v else 'FAIL'}] {k}")
        print("\nGATE STATUS: FAIL_STAGE_5B_METHOD_AMENDMENT")
        raise SystemExit(1)

    freeze_v2 = json.loads(json.dumps(freeze))
    freeze_v2["status"] = (
        "COUNTERFACTUAL_METHOD_V2_FROZEN_AFTER_SMOKE_VALIDATION_FAILURE"
    )
    freeze_v2["parent_freeze"] = {
        "path": str(FREEZE_V1.relative_to(ROOT)),
        "sha256": sha256_file(FREEZE_V1),
    }
    freeze_v2["method_amendment"] = {
        "stage": "5B",
        "reason": (
            "Beam width 500 agreed with exact exhaustive search on solution existence "
            "for all six smoke cases but missed the exact minimum-cost solution in at "
            "least one solvable case. Rather than relax the optimality audit post hoc, "
            "the official single-model counterfactual optimizer is changed to exact "
            "exhaustive enumeration of the already-frozen finite search space."
        ),
        "official_optimizer": "EXACT_EXHAUSTIVE_ENUMERATION",
        "beam_role": "efficiency baseline / ablation only",
        "unchanged_constraints": [
            "13 frozen features",
            "observed normalized feature levels only",
            "maximum 4 changed features",
            "IQR-weighted L1 cost",
            "L0 penalty",
            "Euclidean 13D plausibility",
            "5-NN plausibility rule",
            "target-profile 95th-percentile plausibility thresholds",
            "selected reference geometry",
            "diagnostic non-causal interpretation",
        ],
        "smoke_validation": {
            "cases": len(d),
            "solvable_cases": len(solvable),
            "unsolvable_cases": len(unsolvable),
            "existence_matches": int(existence.sum()),
            "exact_cost_matches_among_solvable": int(exact_match.sum()),
            "exact_cost_mismatches_among_solvable": int((~exact_match).sum()),
            "maximum_absolute_cost_gap": max_gap,
            "mismatch_institutions": mismatch["institution_id"].astype(str).tolist(),
            "exact_candidates_per_case": 147291,
        },
        "anti_p_hacking_note": (
            "No feasibility, sparsity, plausibility, cost, or validity threshold is relaxed."
        ),
    }
    freeze_v2["search_space"][
        "official_search_optimizer"
    ] = "EXACT_EXHAUSTIVE_ENUMERATION"
    freeze_v2["search_space"][
        "beam_role_after_amendment"
    ] = "efficiency baseline / ablation only"
    freeze_v2["amendment_source_hashes"] = {
        "smoke_report": sha256_file(REPORT),
        "smoke_diagnostics": sha256_file(DIAG),
    }
    freeze_v2["gate_status"] = "PASS_STAGE_5B_METHOD_AMENDMENT"

    FREEZE_V2.write_text(
        json.dumps(freeze_v2, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    audit = {
        "checks": checks,
        "solvable_cases": len(solvable),
        "unsolvable_cases": len(unsolvable),
        "exact_cost_matches": int(exact_match.sum()),
        "exact_cost_mismatches": int((~exact_match).sum()),
        "maximum_absolute_cost_gap": max_gap,
        "mismatch_institutions": mismatch["institution_id"].astype(str).tolist(),
        "freeze_v2_sha256": sha256_file(FREEZE_V2),
        "gate_status": "PASS_STAGE_5B_METHOD_AMENDMENT",
    }
    (AUDIT / "stage5b_method_amendment_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== CIPHER STAGE 5B — METHOD AMENDMENT ===\n")
    print("Exact-audited smoke cases:", len(d))
    print("Solvable cases:", len(solvable))
    print("Unsolvable cases:", len(unsolvable))
    print("Beam/exact existence agreement:", f"{int(existence.sum())}/{len(d)}")
    print(
        "Exact minimum-cost matches among solvable:",
        f"{int(exact_match.sum())}/{len(solvable)}",
    )
    print("Maximum observed absolute cost gap:", f"{max_gap:.12f}")
    print("Mismatch institutions:", mismatch["institution_id"].astype(str).tolist())
    print("\nMethod decision:")
    print("  Official Stage 5 optimizer -> EXACT_EXHAUSTIVE_ENUMERATION")
    print("  Beam width 500 -> efficiency baseline / ablation only")
    print("  No scientific threshold was relaxed.")
    print("\n=== AMENDMENT CHECKS ===\n")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    print("\nGATE STATUS: PASS_STAGE_5B_METHOD_AMENDMENT")
    print("Versioned method freeze:", FREEZE_V2.relative_to(ROOT))


if __name__ == "__main__":
    main()
