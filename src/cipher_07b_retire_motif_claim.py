from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STAGE7_POLICY_PATH = ROOT / "cipher" / "design" / "stage7_motif_policy_v1.json"
STAGE7_REPORT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "motifs"
    / "readiness"
    / "stage7a_readiness_report.json"
)
RAW_ITEMSETS_PATH = (
    ROOT / "cipher" / "outputs" / "motifs" / "readiness" / "raw_frequent_itemsets.csv"
)
ITEM_FREQ_PATH = (
    ROOT / "cipher" / "outputs" / "motifs" / "readiness" / "signed_item_frequency.csv"
)

THESIS_V2_PATH = ROOT / "cipher" / "design" / "cipher_thesis_revision_v2.json"

FREEZE_PATH = ROOT / "cipher" / "design" / "cipher_thesis_revision_v3.json"
AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage7b_motif_retirement_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Revision already exists: {FREEZE_PATH}")

    policy = load_json(STAGE7_POLICY_PATH)
    report = load_json(STAGE7_REPORT_PATH)
    thesis_v2 = load_json(THESIS_V2_PATH)

    # Stage 7A legitimately writes an empty CSV when no frequent itemsets exist.
    # pandas raises EmptyDataError on a zero-column/zero-row file, so treat that
    # exact artifact state as an empty table rather than as a pipeline failure.
    try:
        raw_itemsets = pd.read_csv(RAW_ITEMSETS_PATH)
    except pd.errors.EmptyDataError:
        raw_itemsets = pd.DataFrame()

    item_freq = pd.read_csv(ITEM_FREQ_PATH)

    checks = {
        "stage7a_stop_rule_triggered": (
            report.get("gate_status") == "STOP_STAGE_7_NO_MOTIF_READINESS"
        ),
        "policy_matches_stage7a_stop": (
            policy.get("gate_status") == "STOP_STAGE_7_NO_MOTIF_READINESS"
        ),
        "zero_raw_frequent_itemsets": (
            int(report.get("raw_frequent_itemsets", -1)) == 0 and len(raw_itemsets) == 0
        ),
        "empty_itemset_artifact_is_consistent_with_stop_rule": (
            len(raw_itemsets) == 0 and RAW_ITEMSETS_PATH.exists()
        ),
        "zero_raw_closed_itemsets": (int(report.get("raw_closed_itemsets", -1)) == 0),
        "effective_support_count_was_four": (
            int(report.get("effective_min_support_count", -1)) == 4
        ),
        "thesis_v2_had_passed": (
            thesis_v2.get("gate_status") == "PASS_STAGE_5E_THESIS_REVISION"
        ),
    }

    if not all(checks.values()):
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("\nGATE STATUS: FAIL_STAGE_7B_MOTIF_RETIREMENT")
        raise SystemExit(1)

    top_singletons = (
        item_freq.sort_values(
            ["support_count", "signed_item"],
            ascending=[False, True],
        )
        .head(10)
        .to_dict(orient="records")
    )

    revision = {
        "version": "CIPHER_THESIS_V3",
        "status": "FROZEN_AFTER_EMPIRICAL_MOTIF_STOP_RULE",
        "method_name": (
            "CIPHER: Counterfactual Institutional Profiling under "
            "Heterogeneous Ensemble Robustness"
        ),
        "parent_revision": {
            "path": str(THESIS_V2_PATH.relative_to(ROOT)),
            "sha256": sha256_file(THESIS_V2_PATH),
        },
        "empirical_motif_claim": {
            "status": "RETIRED",
            "reason": (
                "Among the 10 primary ensemble-robust Profile-1-to-Profile-2 "
                "transactions, no signed itemset of size 2-4 reached the frozen "
                "minimum support of four institutions. Therefore bootstrap and "
                "randomized-margin motif inference is not run, and the empirical "
                "motif claim is removed without threshold relaxation."
            ),
            "frozen_minimum_support_count": 4,
            "raw_frequent_itemsets": 0,
            "raw_closed_itemsets": 0,
        },
        "single_feature_recurrence_policy": {
            "status": "DESCRIPTIVE_ONLY",
            "allowed": (
                "Individual signed changes may be reported descriptively as "
                "recurring items, but they are not called motifs and are not "
                "treated as inferential discoveries."
            ),
            "top_singletons": top_singletons,
        },
        "revised_core_empirical_story": [
            (
                "A stable two-profile institutional partition is recovered under "
                "representation and algorithm perturbations."
            ),
            (
                "Membership uncertainty is concentrated on the organizational-"
                "capacity side, while the infrastructure-bottleneck profile forms "
                "a perturbation-stable core."
            ),
            (
                "The partition is not reducible to aggregate friction severity or "
                "governance type."
            ),
            (
                "Exact sparse plausible profile transitions are available for only "
                "19/81 institutions, rejecting broad counterfactual reachability."
            ),
            (
                "Reachability is strongly associated with membership uncertainty and "
                "is asymmetric by reference profile."
            ),
            (
                "Among the 19 single-model reachable institutions, 10/19 retain "
                "target-profile support >=0.90 across the 984-member heterogeneous "
                "ensemble; all 10 are Profile 1 -> Profile 2."
            ),
            (
                "Those 10 robust transitions show positive ensemble support gain and "
                "substantial source-to-target member flips, so high final support is "
                "not merely inherited from baseline ambiguity."
            ),
            (
                "No recurrent multi-feature signed motif meets the frozen empirical "
                "support criterion, so motif discovery is not part of the retained "
                "empirical claim."
            ),
        ],
        "revised_methodological_question": (
            "When stable institutional profiles are discovered under heterogeneous "
            "perturbation, can membership uncertainty and ensemble-validated "
            "counterfactual reachability characterize which profile transitions are "
            "structurally accessible without forcing implausible explanations?"
        ),
        "claims_removed": [
            "broad counterfactual availability across institutions",
            "empirical recurrent multi-feature counterfactual motif discovery",
            "empirical profile-specific motif discrimination",
        ],
        "claims_retained_subject_to_synthetic_validation": [
            "heterogeneous-ensemble profile stability",
            "uncertainty-aware counterfactual reachability",
            "directional asymmetry of ensemble-robust reachability",
            (
                "counterfactual support gain as a safeguard against confusing "
                "baseline ambiguity with true prediction change"
            ),
        ],
        "stage8_policy": {
            "status": "REQUIRED_NEXT",
            "purpose": (
                "Synthetic validation must now test the retained reachability and "
                "uncertainty claims, including whether the method detects stable "
                "cores, ambiguous boundaries, directional accessibility, and avoids "
                "forcing counterfactual or motif structure under severity-only and "
                "no-cluster nulls."
            ),
            "motif_metrics": (
                "Motif false-discovery behavior remains useful as a negative-control "
                "diagnostic in synthetic experiments, but empirical motif recovery is "
                "no longer a headline objective."
            ),
        },
        "interpretive_constraints": [
            (
                "Counterfactual transitions are diagnostic model transitions, not "
                "causal interventions or policy recommendations."
            ),
            (
                "The Stage-5D uncertainty-reachability association is exploratory "
                "post-failure and structurally internal to the feature geometry."
            ),
            (
                "The real sample does not identify profile-specific motif effects "
                "because all robust transitions are Profile 1 -> Profile 2."
            ),
            ("The Mexican N=81 case is not nationally representative."),
        ],
        "source_hashes": {
            "stage7_policy": sha256_file(STAGE7_POLICY_PATH),
            "stage7_report": sha256_file(STAGE7_REPORT_PATH),
            "raw_itemsets": sha256_file(RAW_ITEMSETS_PATH),
            "item_frequency": sha256_file(ITEM_FREQ_PATH),
        },
        "gate_status": "PASS_STAGE_7B_MOTIF_RETIREMENT",
    }

    FREEZE_PATH.write_text(
        json.dumps(
            revision,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "checks": checks,
                "revision_path": str(FREEZE_PATH.relative_to(ROOT)),
                "revision_sha256": sha256_file(FREEZE_PATH),
                "gate_status": "PASS_STAGE_7B_MOTIF_RETIREMENT",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 7B — EMPIRICAL MOTIF CLAIM RETIREMENT ===\n")

    print("Empirical multi-feature motif claim:")
    print("  RETIRED BY FROZEN STAGE-7 SUPPORT RULE")
    print("  raw frequent itemsets = 0")
    print("  raw closed itemsets = 0")
    print("  minimum support = 4/10")

    print("\nSingle signed items:")
    print("  May remain descriptive only; they are not called motifs.")

    print("\nRetained CIPHER core:")
    print("  stable profiles")
    print("  membership uncertainty")
    print("  exact sparse plausible reachability")
    print("  ensemble-robust reachability")
    print("  counterfactual support gain / source-to-target flips")

    print("\nMethod name:")
    print(
        "  CIPHER: Counterfactual Institutional Profiling under "
        "Heterogeneous Ensemble Robustness"
    )

    print("\nNext required stage:")
    print("  Stage 8 — synthetic validation of the retained reachability thesis")

    print("\n=== FREEZE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_7B_MOTIF_RETIREMENT")
    print("Frozen revision:", FREEZE_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
