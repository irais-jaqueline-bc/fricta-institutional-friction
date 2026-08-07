from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STAGE6D_REPORT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "ensemble_lift_audit"
    / "stage6d_lift_audit_report.json"
)
SELECTED_LIFT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "ensemble_lift_audit"
    / "institution_selected_candidate_lift.csv"
)
CANDIDATE_SUMMARY_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "ensemble_robustness"
    / "candidate_ensemble_support.csv"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "motifs" / "readiness"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"
POLICY_PATH = ROOT / "cipher" / "design" / "stage7_motif_policy_v1.json"

PRIMARY_TAU = 0.90
MIN_SUPPORT_FRACTION = 0.25
MIN_SUPPORT_ABSOLUTE = 4
MIN_ITEMSET_SIZE = 2
MAX_ITEMSET_SIZE = 4
BOOTSTRAPS_FROZEN = 1000
BOOTSTRAP_RECURRENCE_FROZEN = 0.70
RANDOMIZATIONS_FROZEN = 10_000
BH_Q_FROZEN = 0.05


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_list(value: str) -> list[str]:
    obj = json.loads(value)
    if not isinstance(obj, list):
        raise ValueError("Expected JSON list.")
    return [str(item) for item in obj]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def support_ids(
    itemset: frozenset[str],
    transactions: dict[str, frozenset[str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            institution_id
            for institution_id, transaction in transactions.items()
            if itemset.issubset(transaction)
        )
    )


def enumerate_frequent_itemsets(
    transactions: dict[str, frozenset[str]],
    min_support_count: int,
) -> list[dict[str, Any]]:
    support_map: dict[frozenset[str], tuple[str, ...]] = {}

    for institution_id, transaction in transactions.items():
        ordered = sorted(transaction)

        for size in range(
            MIN_ITEMSET_SIZE,
            min(MAX_ITEMSET_SIZE, len(ordered)) + 1,
        ):
            for combo in itertools.combinations(ordered, size):
                key = frozenset(combo)
                if key not in support_map:
                    support_map[key] = support_ids(
                        key,
                        transactions,
                    )

    frequent = []

    for itemset, institutions in support_map.items():
        count = len(institutions)

        if count >= min_support_count:
            frequent.append(
                {
                    "itemset": itemset,
                    "support_count": count,
                    "support_fraction": count / len(transactions),
                    "support_institution_ids": institutions,
                }
            )

    frequent.sort(
        key=lambda row: (
            -row["support_count"],
            len(row["itemset"]),
            tuple(sorted(row["itemset"])),
        )
    )

    return frequent


def mark_closed(
    frequent: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for row in frequent:
        itemset = row["itemset"]
        support_count = row["support_count"]

        has_equal_support_superset = False

        for other in frequent:
            other_itemset = other["itemset"]

            if (
                len(other_itemset) > len(itemset)
                and itemset.issubset(other_itemset)
                and other["support_count"] == support_count
            ):
                has_equal_support_superset = True
                break

        out = dict(row)
        out["closed"] = not has_equal_support_superset
        rows.append(out)

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    stage6d = load_json(STAGE6D_REPORT_PATH)

    if stage6d.get("gate_status") != "PASS_STAGE_6D_LIFT_AUDIT":
        raise ValueError("Stage 6D lift audit has not passed.")

    selected = pd.read_csv(SELECTED_LIFT_PATH)
    candidates = pd.read_csv(CANDIDATE_SUMMARY_PATH)

    selected["institution_id"] = selected["institution_id"].astype(str)
    candidates["institution_id"] = candidates["institution_id"].astype(str)
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    robust = selected[selected["robustly_reachable_tau_0.90"].map(parse_bool)].copy()

    robust = robust.merge(
        candidates[
            [
                "candidate_id",
                "institution_id",
                "reference_profile",
                "target_profile",
                "certainty_class",
                "rank",
                "total_cost",
                "ensemble_support",
                "minimum_family_support",
                "signed_items_json",
                "changed_features_json",
            ]
        ],
        on=[
            "candidate_id",
            "institution_id",
        ],
        how="inner",
        suffixes=("_lift", "_candidate"),
        validate="one_to_one",
    )

    if len(robust) != 10:
        raise ValueError(
            f"Expected 10 primary-robust institutions; found {len(robust)}."
        )

    # Hotfix audit: the merge contains overlapping Stage-6D and Stage-6C
    # metadata, so pandas suffixes those columns. Confirm both sources agree
    # before using the candidate-side frozen metadata.
    metadata_pairs = [
        ("reference_profile_lift", "reference_profile_candidate"),
        ("target_profile_lift", "target_profile_candidate"),
        ("certainty_class_lift", "certainty_class_candidate"),
        ("rank_lift", "rank_candidate"),
        ("total_cost_lift", "total_cost_candidate"),
        ("ensemble_support_lift", "ensemble_support_candidate"),
    ]

    for left_col, right_col in metadata_pairs:
        if left_col not in robust.columns or right_col not in robust.columns:
            raise KeyError(
                f"Expected suffixed merge columns {left_col!r} and {right_col!r}; "
                f"available columns are: {list(robust.columns)}"
            )

    numeric_pairs = [
        ("reference_profile_lift", "reference_profile_candidate"),
        ("target_profile_lift", "target_profile_candidate"),
        ("rank_lift", "rank_candidate"),
        ("total_cost_lift", "total_cost_candidate"),
        ("ensemble_support_lift", "ensemble_support_candidate"),
    ]

    for left_col, right_col in numeric_pairs:
        left = pd.to_numeric(robust[left_col], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(robust[right_col], errors="raise").to_numpy(dtype=float)
        if not ((abs(left - right) <= 1e-12).all()):
            raise ValueError(
                f"Stage-6D/Stage-6C metadata mismatch between "
                f"{left_col} and {right_col}."
            )

    if not (
        robust["certainty_class_lift"].astype(str).to_numpy()
        == robust["certainty_class_candidate"].astype(str).to_numpy()
    ).all():
        raise ValueError("Stage-6D/Stage-6C certainty_class metadata mismatch.")

    transactions: dict[str, frozenset[str]] = {}

    transaction_rows = []

    for _, row in robust.iterrows():
        institution_id = str(row["institution_id"])
        signed_items = parse_json_list(str(row["signed_items_json"]))

        transaction = frozenset(signed_items)

        if len(transaction) != len(signed_items):
            raise ValueError(
                f"{institution_id}: duplicate signed item within selected candidate."
            )

        transactions[institution_id] = transaction

        transaction_rows.append(
            {
                "institution_id": institution_id,
                "candidate_id": str(row["candidate_id"]),
                "reference_profile": int(row["reference_profile_candidate"]),
                "target_profile": int(row["target_profile_candidate"]),
                "certainty_class": str(row["certainty_class_candidate"]),
                "candidate_rank": int(row["rank_candidate"]),
                "total_cost": float(row["total_cost_candidate"]),
                "ensemble_support": float(row["ensemble_support_candidate"]),
                "minimum_family_support": float(row["minimum_family_support"]),
                "transaction_size": len(transaction),
                "signed_items_json": json.dumps(
                    sorted(transaction),
                    ensure_ascii=False,
                ),
            }
        )

    transactions_df = pd.DataFrame(transaction_rows)

    source_profiles = sorted(transactions_df["reference_profile"].unique().tolist())
    target_profiles = sorted(transactions_df["target_profile"].unique().tolist())

    effective_min_support = max(
        MIN_SUPPORT_ABSOLUTE,
        math.ceil(MIN_SUPPORT_FRACTION * len(transactions)),
    )

    frequent = enumerate_frequent_itemsets(
        transactions,
        effective_min_support,
    )
    marked = mark_closed(frequent)
    closed = [row for row in marked if row["closed"]]

    item_frequency: dict[str, int] = {}

    for transaction in transactions.values():
        for item in transaction:
            item_frequency[item] = item_frequency.get(item, 0) + 1

    item_frequency_df = pd.DataFrame(
        [
            {
                "signed_item": item,
                "support_count": count,
                "support_fraction": count / len(transactions),
            }
            for item, count in sorted(
                item_frequency.items(),
                key=lambda pair: (
                    -pair[1],
                    pair[0],
                ),
            )
        ]
    )

    frequent_df = pd.DataFrame(
        [
            {
                "itemset_size": len(row["itemset"]),
                "itemset_json": json.dumps(
                    sorted(row["itemset"]),
                    ensure_ascii=False,
                ),
                "support_count": row["support_count"],
                "support_fraction": row["support_fraction"],
                "support_institution_ids_json": json.dumps(
                    row["support_institution_ids"],
                    ensure_ascii=False,
                ),
                "closed": row["closed"],
            }
            for row in marked
        ]
    )

    transactions_df.to_csv(
        OUTPUT_DIR / "robust_transactions.csv",
        index=False,
    )
    item_frequency_df.to_csv(
        OUTPUT_DIR / "signed_item_frequency.csv",
        index=False,
    )
    frequent_df.to_csv(
        OUTPUT_DIR / "raw_frequent_itemsets.csv",
        index=False,
    )

    all_same_direction = len(source_profiles) == 1 and len(target_profiles) == 1

    checks = {
        "stage6d_passed": (stage6d.get("gate_status") == "PASS_STAGE_6D_LIFT_AUDIT"),
        "exactly_10_primary_robust_institutions": (len(transactions_df) == 10),
        "exactly_one_transaction_per_institution": (
            transactions_df["institution_id"].nunique() == 10
        ),
        "equal_institution_weighting_preserved": True,
        "all_transaction_sizes_between_2_and_4": bool(
            transactions_df["transaction_size"]
            .between(
                MIN_ITEMSET_SIZE,
                MAX_ITEMSET_SIZE,
            )
            .all()
        ),
        "all_robust_transactions_same_direction": (all_same_direction),
        "effective_min_support_is_4": (effective_min_support == 4),
        "at_least_one_raw_closed_itemset_meets_support": (len(closed) > 0),
    }

    policy = {
        "version": "STAGE7_MOTIF_POLICY_V1",
        "status": ("FROZEN_BEFORE_BOOTSTRAP_AND_RANDOMIZATION"),
        "empirical_scope": {
            "institutions": 10,
            "transaction_unit": (
                "one Stage-6C best primary-robust candidate per institution"
            ),
            "equal_institution_weighting": True,
            "source_profiles_present": source_profiles,
            "target_profiles_present": target_profiles,
            "scope_mode": (
                "UNIDIRECTIONAL_ROBUST_TRANSITION_MOTIFS"
                if all_same_direction
                else "BIDIRECTIONAL_ROBUST_TRANSITION_MOTIFS"
            ),
        },
        "frozen_thresholds": {
            "signed_items": True,
            "itemset_size_min": MIN_ITEMSET_SIZE,
            "itemset_size_max": MAX_ITEMSET_SIZE,
            "minimum_support_fraction": MIN_SUPPORT_FRACTION,
            "minimum_support_absolute": MIN_SUPPORT_ABSOLUTE,
            "effective_minimum_support_count_real_case": (effective_min_support),
            "bootstrap_iterations": BOOTSTRAPS_FROZEN,
            "bootstrap_recurrence_threshold": (BOOTSTRAP_RECURRENCE_FROZEN),
            "randomized_collections": RANDOMIZATIONS_FROZEN,
            "bh_q_threshold": BH_Q_FROZEN,
        },
        "specificity_constraint": {
            "real_case_profile_specificity_identifiable": (not all_same_direction),
            "reason_if_not_identifiable": (
                "All primary-robust empirical transactions share the same "
                "reference-to-target direction, so real-case profile specificity "
                "cannot be estimated separately from transition direction."
                if all_same_direction
                else ""
            ),
            "allowed_real_case_claim": (
                "recurrent signed motifs among robust Profile-1-to-Profile-2 "
                "diagnostic transitions"
                if all_same_direction
                else "recurrent signed motifs among robust diagnostic transitions"
            ),
            "forbidden_claim": (
                "motifs distinguish robustness from source profile in the real sample"
                if all_same_direction
                else ""
            ),
            "profile_specificity_recovery": (
                "evaluate in synthetic validation scenarios rather than "
                "pretending the real sample identifies it"
                if all_same_direction
                else "evaluate empirically and in synthetic validation"
            ),
        },
        "stop_rule": (
            "If no itemset survives both bootstrap recurrence >=0.70 and "
            "randomized-margin testing with BH q<0.05, remove the empirical "
            "motif claim without relaxing thresholds."
        ),
        "raw_readiness": {
            "raw_frequent_itemsets": len(frequent),
            "raw_closed_itemsets": len(closed),
        },
        "gate_status": (
            "PASS_STAGE_7A_MOTIF_READINESS"
            if all(checks.values())
            else "STOP_STAGE_7_NO_MOTIF_READINESS"
        ),
    }

    POLICY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if POLICY_PATH.exists():
        raise FileExistsError(f"Stage 7 policy already exists: {POLICY_PATH}")

    POLICY_PATH.write_text(
        json.dumps(
            policy,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = {
        "transactions": len(transactions_df),
        "source_profiles": source_profiles,
        "target_profiles": target_profiles,
        "effective_min_support_count": (effective_min_support),
        "unique_signed_items": len(item_frequency_df),
        "raw_frequent_itemsets": len(frequent),
        "raw_closed_itemsets": len(closed),
        "checks": checks,
        "gate_status": policy["gate_status"],
    }

    (OUTPUT_DIR / "stage7a_readiness_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (AUDIT_DIR / "stage7a_motif_readiness_audit.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 7A — MOTIF READINESS + TRANSACTION FREEZE ===\n")

    print(
        "Robust institutions:",
        len(transactions_df),
    )
    print(
        "Transition direction(s):",
        source_profiles,
        "->",
        target_profiles,
    )
    print(
        "Effective minimum motif support:",
        f"{effective_min_support}/{len(transactions_df)}",
        f"({effective_min_support / len(transactions_df):.2%})",
    )
    print(
        "Unique signed items:",
        len(item_frequency_df),
    )
    print(
        "Raw frequent itemsets:",
        len(frequent),
    )
    print(
        "Raw closed itemsets:",
        len(closed),
    )

    print("\n=== ROBUST TRANSACTIONS ===\n")
    print(
        transactions_df[
            [
                "institution_id",
                "candidate_id",
                "certainty_class",
                "transaction_size",
                "signed_items_json",
            ]
        ].to_string(index=False)
    )

    print("\n=== SIGNED ITEM FREQUENCY ===\n")
    print(item_frequency_df.to_string(index=False))

    print("\n=== RAW CLOSED ITEMSETS MEETING SUPPORT ===\n")

    if closed:
        closed_df = frequent_df[frequent_df["closed"].astype(bool)].copy()

        print(
            closed_df[
                [
                    "itemset_size",
                    "itemset_json",
                    "support_count",
                    "support_fraction",
                    "support_institution_ids_json",
                ]
            ].to_string(index=False)
        )
    else:
        print("None.")

    print("\n=== INTERPRETATION CONSTRAINT ===\n")

    if all_same_direction:
        print("All 10 robust empirical transactions are unidirectional.")
        print("Stage 7 may test recurrent motifs within that transition direction,")
        print(
            "but may NOT claim that motifs distinguish robustness from source profile."
        )
        print("Profile-specificity recovery must be tested in synthetic validation.")

    print("\n=== READINESS CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {policy['gate_status']}")

    if policy["gate_status"] == "PASS_STAGE_7A_MOTIF_READINESS":
        print(
            "Raw recurrence is sufficient to attempt the frozen bootstrap + "
            "randomized-margin motif validation. No motif is scientific yet."
        )
    else:
        print(
            "Do not run inferential motif validation. Remove the empirical motif claim."
        )


if __name__ == "__main__":
    main()
