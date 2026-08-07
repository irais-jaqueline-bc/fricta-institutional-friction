from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, chi2_contingency

ROOT = Path(__file__).resolve().parents[1]

DIAGNOSTICS_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "official_exact"
    / "institution_counterfactual_diagnostics.csv"
)
CERTAINTY_PATH = ROOT / "cipher" / "outputs" / "certainty" / "institution_certainty.csv"
STAGE5C_REPORT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "official_exact"
    / "stage5c_report.json"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "counterfactuals" / "reachability_audit"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"

PERMUTATIONS = 100_000
SEED = 20260806


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cramers_v(table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    r, k = table.shape
    denom = min(r - 1, k - 1)
    if n == 0 or denom <= 0:
        return np.nan
    return float(np.sqrt((chi2 / n) / denom))


def permutation_p_for_contingency(
    row_groups: np.ndarray,
    outcome_labels: np.ndarray,
    observed_stat: float,
    permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    count = 0

    unique_rows = np.unique(row_groups)
    unique_outcomes = np.unique(outcome_labels)

    for _ in range(permutations):
        shuffled = rng.permutation(outcome_labels)
        table = (
            pd.crosstab(
                pd.Series(row_groups, name="row"),
                pd.Series(shuffled, name="outcome"),
            )
            .reindex(
                index=unique_rows,
                columns=unique_outcomes,
                fill_value=0,
            )
            .to_numpy()
        )

        stat = cramers_v(table)

        if stat >= observed_stat - 1e-15:
            count += 1

    return float((count + 1) / (permutations + 1))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    greater = 0
    less = 0

    for a in x:
        greater += int(np.sum(a > y))
        less += int(np.sum(a < y))

    return float((greater - less) / (len(x) * len(y)))


def format_fisher(table: np.ndarray) -> dict:
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    return {
        "table": table.tolist(),
        "odds_ratio": (float(odds_ratio) if np.isfinite(odds_ratio) else "inf"),
        "p_value": float(p_value),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    report5c = load_json(STAGE5C_REPORT_PATH)

    if report5c.get("gate_status") != "FAIL_STAGE_5C":
        raise ValueError(
            "This audit is specifically for the failed Stage 5C coverage gate."
        )

    diagnostics = pd.read_csv(DIAGNOSTICS_PATH)
    certainty = pd.read_csv(CERTAINTY_PATH)

    diagnostics["institution_id"] = diagnostics["institution_id"].astype(str)
    certainty["institution_id"] = certainty["institution_id"].astype(str)

    data = diagnostics.merge(
        certainty[
            [
                "institution_id",
                "reference_profile_probability",
                "family_consistency",
                "normalized_entropy",
                "membership_margin",
            ]
        ],
        on="institution_id",
        how="inner",
        suffixes=("", "_certainty"),
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 aligned institutions; found {len(data)}.")

    data["solvable"] = data["failure_mode"] == "SOLVABLE"

    # -----------------------------
    # Descriptive contingency tables
    # -----------------------------
    profile_reachability = pd.crosstab(
        data["reference_profile"],
        data["solvable"],
    ).reindex(
        index=[1, 2],
        columns=[False, True],
        fill_value=0,
    )

    certainty_reachability = pd.crosstab(
        data["certainty_class"],
        data["solvable"],
    ).reindex(
        index=["CORE", "HALO", "BOUNDARY"],
        columns=[False, True],
        fill_value=0,
    )

    profile_failure_modes = pd.crosstab(
        data["reference_profile"],
        data["failure_mode"],
    ).reindex(
        index=[1, 2],
        columns=[
            "SOLVABLE",
            "TRANSITIONS_FAIL_PLAUSIBILITY",
            "NO_TRANSITION_WITHIN_4_FEATURES",
        ],
        fill_value=0,
    )

    certainty_failure_modes = pd.crosstab(
        data["certainty_class"],
        data["failure_mode"],
    ).reindex(
        index=["CORE", "HALO", "BOUNDARY"],
        columns=[
            "SOLVABLE",
            "TRANSITIONS_FAIL_PLAUSIBILITY",
            "NO_TRANSITION_WITHIN_4_FEATURES",
        ],
        fill_value=0,
    )

    profile_reachability.to_csv(OUTPUT_DIR / "reachability_by_profile.csv")
    certainty_reachability.to_csv(OUTPUT_DIR / "reachability_by_certainty.csv")
    profile_failure_modes.to_csv(OUTPUT_DIR / "failure_modes_by_profile.csv")
    certainty_failure_modes.to_csv(OUTPUT_DIR / "failure_modes_by_certainty.csv")

    # -----------------------------
    # Exact 2x2 tests
    # -----------------------------
    # Profile 2 vs profile 1 for reachability.
    profile2_vs_profile1 = np.array(
        [
            [
                int(((data["reference_profile"] == 2) & data["solvable"]).sum()),
                int(((data["reference_profile"] == 2) & ~data["solvable"]).sum()),
            ],
            [
                int(((data["reference_profile"] == 1) & data["solvable"]).sum()),
                int(((data["reference_profile"] == 1) & ~data["solvable"]).sum()),
            ],
        ]
    )

    # Boundary vs non-boundary.
    boundary = data["certainty_class"] == "BOUNDARY"

    boundary_vs_nonboundary = np.array(
        [
            [
                int((boundary & data["solvable"]).sum()),
                int((boundary & ~data["solvable"]).sum()),
            ],
            [
                int((~boundary & data["solvable"]).sum()),
                int((~boundary & ~data["solvable"]).sum()),
            ],
        ]
    )

    profile_fisher = format_fisher(profile2_vs_profile1)
    boundary_fisher = format_fisher(boundary_vs_nonboundary)

    # -----------------------------
    # Multi-category exploratory tests
    # -----------------------------
    cert_table = certainty_reachability.to_numpy()
    cert_v = cramers_v(cert_table)
    cert_perm_p = permutation_p_for_contingency(
        data["certainty_class"].to_numpy(),
        data["solvable"].astype(str).to_numpy(),
        observed_stat=cert_v,
        permutations=PERMUTATIONS,
        seed=SEED,
    )

    failure_profile_table = profile_failure_modes.to_numpy()
    failure_profile_v = cramers_v(failure_profile_table)
    failure_profile_perm_p = permutation_p_for_contingency(
        data["reference_profile"].to_numpy(),
        data["failure_mode"].to_numpy(),
        observed_stat=failure_profile_v,
        permutations=PERMUTATIONS,
        seed=SEED + 1,
    )

    # -----------------------------
    # Within-profile-1 uncertainty comparison
    # -----------------------------
    # This is explicitly exploratory after the failed coverage gate.
    p1 = data[data["reference_profile"] == 1].copy()

    p1_solved = p1[p1["solvable"]]
    p1_unsolved = p1[~p1["solvable"]]

    uncertainty_tests = {}

    for column in [
        "reference_profile_probability",
        "family_consistency",
        "normalized_entropy",
        "membership_margin",
    ]:
        solved_values = p1_solved[column].to_numpy(dtype=float)
        unsolved_values = p1_unsolved[column].to_numpy(dtype=float)

        u, p_value = mannwhitneyu(
            solved_values,
            unsolved_values,
            alternative="two-sided",
        )

        uncertainty_tests[column] = {
            "solved_n": int(len(solved_values)),
            "unsolved_n": int(len(unsolved_values)),
            "solved_median": float(np.median(solved_values)),
            "unsolved_median": float(np.median(unsolved_values)),
            "mann_whitney_u": float(u),
            "p_value": float(p_value),
            "cliffs_delta_solved_minus_unsolved": cliffs_delta(
                solved_values,
                unsolved_values,
            ),
        }

    # -----------------------------
    # Compact descriptive summary
    # -----------------------------
    rows = []

    for profile, group in data.groupby(
        "reference_profile",
        sort=True,
    ):
        rows.append(
            {
                "dimension": "REFERENCE_PROFILE",
                "group": str(profile),
                "n": len(group),
                "solvable": int(group["solvable"].sum()),
                "coverage": float(group["solvable"].mean()),
                "plausibility_failures": int(
                    (group["failure_mode"] == "TRANSITIONS_FAIL_PLAUSIBILITY").sum()
                ),
                "no_transition_failures": int(
                    (group["failure_mode"] == "NO_TRANSITION_WITHIN_4_FEATURES").sum()
                ),
            }
        )

    for certainty_class, group in data.groupby(
        "certainty_class",
        sort=True,
    ):
        rows.append(
            {
                "dimension": "CERTAINTY_CLASS",
                "group": str(certainty_class),
                "n": len(group),
                "solvable": int(group["solvable"].sum()),
                "coverage": float(group["solvable"].mean()),
                "plausibility_failures": int(
                    (group["failure_mode"] == "TRANSITIONS_FAIL_PLAUSIBILITY").sum()
                ),
                "no_transition_failures": int(
                    (group["failure_mode"] == "NO_TRANSITION_WITHIN_4_FEATURES").sum()
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT_DIR / "reachability_summary.csv",
        index=False,
    )

    report = {
        "status": ("EXPLORATORY_POST_FAILURE_AUDIT"),
        "stage5c_original_coverage_gate": {
            "threshold": 0.70,
            "observed": float(report5c["coverage"]),
            "passed": False,
        },
        "profile2_vs_profile1_reachability_fisher": (profile_fisher),
        "boundary_vs_nonboundary_reachability_fisher": (boundary_fisher),
        "certainty_class_vs_reachability": {
            "cramers_v": cert_v,
            "permutation_p": cert_perm_p,
            "permutations": PERMUTATIONS,
        },
        "profile_vs_failure_mode": {
            "cramers_v": failure_profile_v,
            "permutation_p": failure_profile_perm_p,
            "permutations": PERMUTATIONS,
        },
        "profile1_uncertainty_comparisons": (uncertainty_tests),
        "interpretation_boundary": (
            "These analyses characterize structured counterfactual reachability "
            "after the preregistered Stage 5C coverage gate failed. They are "
            "exploratory and must not be presented as if they were the original "
            "confirmatory counterfactual claim."
        ),
        "gate_status": ("STAGE_5D_REACHABILITY_AUDIT_COMPLETE"),
    }

    (OUTPUT_DIR / "reachability_audit_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 5D — COUNTERFACTUAL REACHABILITY AUDIT ===\n")

    print(
        "NOTE: Stage 5C's >=70% coverage claim remains FAILED. "
        "This is an exploratory post-failure characterization.\n"
    )

    print("=== REACHABILITY SUMMARY ===\n")
    print(summary.to_string(index=False))

    print("\n=== PROFILE 2 vs PROFILE 1 — REACHABILITY ===\n")
    print(
        "2x2 table [[P2 solved, P2 unsolved], [P1 solved, P1 unsolved]]:",
        profile_fisher["table"],
    )
    print(
        "Fisher odds ratio:",
        profile_fisher["odds_ratio"],
    )
    print(
        "Fisher p:",
        f"{profile_fisher['p_value']:.8f}",
    )

    print("\n=== BOUNDARY vs NON-BOUNDARY — REACHABILITY ===\n")
    print(
        "2x2 table [[Boundary solved, Boundary unsolved], [Other solved, Other unsolved]]:",
        boundary_fisher["table"],
    )
    print(
        "Fisher odds ratio:",
        boundary_fisher["odds_ratio"],
    )
    print(
        "Fisher p:",
        f"{boundary_fisher['p_value']:.8f}",
    )

    print("\n=== CERTAINTY CLASS vs REACHABILITY ===\n")
    print(
        "Cramer's V:",
        f"{cert_v:.4f}",
    )
    print(
        "Permutation p:",
        f"{cert_perm_p:.8f}",
    )

    print("\n=== PROFILE vs FAILURE MODE ===\n")
    print(
        "Cramer's V:",
        f"{failure_profile_v:.4f}",
    )
    print(
        "Permutation p:",
        f"{failure_profile_perm_p:.8f}",
    )

    print("\n=== WITHIN PROFILE 1: SOLVABLE vs UNSOLVABLE UNCERTAINTY ===\n")

    for column, result in uncertainty_tests.items():
        print(
            f"{column}: "
            f"solved median={result['solved_median']:.4f}, "
            f"unsolved median={result['unsolved_median']:.4f}, "
            f"Cliff delta={result['cliffs_delta_solved_minus_unsolved']:.4f}, "
            f"p={result['p_value']:.8f}"
        )

    print("\nGATE STATUS: STAGE_5D_REACHABILITY_AUDIT_COMPLETE")
    print(
        "Do not resume the original Stage 6 path until the CIPHER thesis is explicitly revised."
    )


if __name__ == "__main__":
    main()
