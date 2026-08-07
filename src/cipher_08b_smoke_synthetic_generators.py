from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cipher_synthetic_generators import (
    FEATURE_NAMES,
    IMPLEMENTATION_SPEC,
    generate_scenario,
    plausibility_audit_s3,
)

ROOT = Path(__file__).resolve().parents[1]

DESIGN_PATH = ROOT / "cipher" / "design" / "stage8_synthetic_validation_freeze_v1.json"
IMPLEMENTATION_FREEZE_PATH = (
    ROOT / "cipher" / "design" / "stage8_generator_implementation_freeze_v1.json"
)
OUTPUT_DIR = ROOT / "cipher" / "outputs" / "synthetic" / "smoke"
AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8b_generator_smoke_audit.json"
)

SCENARIOS = [
    "S1_CONFIG_TWO_PROFILE",
    "S2_CORE_BOUNDARY",
    "S3_DIRECTIONAL_REACHABILITY",
    "S4_SEVERITY_CONTINUUM",
    "S5_GOVERNANCE_CONFOUNDED",
    "S6_NO_CLUSTER_NULL",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def df_hash(frame: pd.DataFrame) -> str:
    return sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def pairwise_centroid_distances(
    frame: pd.DataFrame,
    labels: pd.Series,
) -> list[float]:
    centroids = {}
    for label in sorted(labels.astype(str).unique()):
        centroids[label] = (
            frame[labels.astype(str) == label][FEATURE_NAMES]
            .mean(axis=0)
            .to_numpy(dtype=float)
        )

    distances = []
    keys = sorted(centroids)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            distances.append(
                float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]]))
            )
    return distances


def main() -> None:
    design = load_json(DESIGN_PATH)

    if design.get("gate_status") != "PASS_STAGE_8A_SYNTHETIC_DESIGN_FREEZE":
        raise ValueError("Stage 8A synthetic design freeze has not passed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMPLEMENTATION_FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)

    implementation_payload = {
        "parent_design_path": str(DESIGN_PATH.relative_to(ROOT)),
        "parent_design_sha256": sha256_bytes(DESIGN_PATH.read_bytes()),
        "implementation_spec": IMPLEMENTATION_SPEC,
        "gate_status": "FROZEN_STAGE8_GENERATOR_IMPLEMENTATION_V1",
    }

    if IMPLEMENTATION_FREEZE_PATH.exists():
        existing = load_json(IMPLEMENTATION_FREEZE_PATH)
        if canonical_json(existing) != canonical_json(implementation_payload):
            raise ValueError(
                "Existing Stage-8 generator implementation freeze differs from "
                "the current generator code. Do not overwrite it."
            )
    else:
        IMPLEMENTATION_FREEZE_PATH.write_text(
            json.dumps(
                implementation_payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print("\n=== CIPHER STAGE 8B — SYNTHETIC GENERATOR SMOKE/AUDIT ===\n")
    print(
        "Implementation freeze:",
        IMPLEMENTATION_FREEZE_PATH.relative_to(ROOT),
    )
    print("No clustering/model-selection performance is evaluated in this stage.")

    bundles = {}
    determinism_checks = {}
    common_checks = {}
    summary_rows = []

    for scenario in SCENARIOS:
        bundle = generate_scenario(
            scenario_id=scenario,
            replicate=0,
            master_seed=20260807,
        )
        repeat = generate_scenario(
            scenario_id=scenario,
            replicate=0,
            master_seed=20260807,
        )

        bundles[scenario] = bundle

        data_same = df_hash(bundle.data) == df_hash(repeat.data)
        truth_same = df_hash(bundle.truth) == df_hash(repeat.truth)
        metadata_same = canonical_json(bundle.metadata) == canonical_json(
            repeat.metadata
        )

        if bundle.oracle_candidates is None:
            oracle_same = repeat.oracle_candidates is None
        else:
            oracle_same = repeat.oracle_candidates is not None and df_hash(
                bundle.oracle_candidates
            ) == df_hash(repeat.oracle_candidates)

        determinism_checks[scenario] = bool(
            data_same and truth_same and metadata_same and oracle_same
        )

        X = bundle.data[FEATURE_NAMES].to_numpy(dtype=float)

        common_checks[scenario] = bool(
            bundle.data.shape == (80, 14)
            and bundle.truth.shape[0] == 80
            and bundle.data["institution_id"].nunique() == 80
            and np.isfinite(X).all()
            and ((X >= 0.0) & (X <= 1.0)).all()
        )

        scenario_dir = OUTPUT_DIR / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        bundle.data.to_csv(
            scenario_dir / "smoke_data.csv",
            index=False,
        )
        bundle.truth.to_csv(
            scenario_dir / "smoke_truth.csv",
            index=False,
        )
        (scenario_dir / "smoke_metadata.json").write_text(
            json.dumps(
                bundle.metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if bundle.oracle_candidates is not None:
            bundle.oracle_candidates.to_csv(
                scenario_dir / "smoke_oracle_candidates.csv",
                index=False,
            )

        summary_rows.append(
            {
                "scenario": scenario,
                "n": len(bundle.data),
                "feature_min": float(X.min()),
                "feature_max": float(X.max()),
                "feature_mean": float(X.mean()),
                "numeric_label_swap": bundle.metadata.get("numeric_label_swap"),
                "deterministic_repeat": determinism_checks[scenario],
                "common_integrity": common_checks[scenario],
            }
        )

    # -----------------------------
    # Scenario-specific audits
    # -----------------------------
    s1 = bundles["S1_CONFIG_TWO_PROFILE"]
    s1_counts = (
        s1.truth["true_profile"].astype(int).value_counts().sort_index().to_dict()
    )
    proto_a = np.array(s1.metadata["prototype_A"], dtype=float)
    proto_b = np.array(s1.metadata["prototype_B"], dtype=float)

    s1_checks = {
        "balanced_40_40": sorted(s1_counts.values()) == [40, 40],
        "prototype_mean_severity_equal": abs(float(proto_a.mean() - proto_b.mean()))
        <= 1e-12,
        "prototype_euclidean_separation_gt_1": (
            float(np.linalg.norm(proto_a - proto_b)) > 1.0
        ),
    }

    s2 = bundles["S2_CORE_BOUNDARY"]
    s2_counts = (
        s2.truth["true_profile"].astype(int).value_counts().sort_index().to_dict()
    )
    s2_boundary_n = int(s2.truth["true_boundary"].astype(bool).sum())

    merged_s2 = s2.data.merge(
        s2.truth[["institution_id", "true_boundary"]],
        on="institution_id",
        validate="one_to_one",
    )
    midpoint = 0.5 * (
        np.array(s2.metadata["prototype_A"], dtype=float)
        + np.array(s2.metadata["prototype_B"], dtype=float)
    )

    distance_to_midpoint = np.linalg.norm(
        merged_s2[FEATURE_NAMES].to_numpy(dtype=float) - midpoint[None, :],
        axis=1,
    )

    boundary_mask = merged_s2["true_boundary"].astype(bool).to_numpy()

    s2_checks = {
        "balanced_40_40": sorted(s2_counts.values()) == [40, 40],
        "exactly_24_true_boundary": s2_boundary_n == 24,
        "boundary_median_closer_to_midpoint_than_core": (
            float(np.median(distance_to_midpoint[boundary_mask]))
            < float(np.median(distance_to_midpoint[~boundary_mask]))
        ),
    }

    s3 = bundles["S3_DIRECTIONAL_REACHABILITY"]
    s3_counts = (
        s3.truth["true_profile"].astype(int).value_counts().sort_index().to_dict()
    )
    s3_oracle_positive = int(s3.truth["oracle_reachable"].astype(bool).sum())
    s3_gate_n = len(s3.metadata["gate_feature_indices"])
    s3_reverse_lock_n = len(s3.metadata["reverse_lock_feature_indices"])
    s3_plaus = plausibility_audit_s3(s3)

    s3_checks = {
        "balanced_40_40": sorted(s3_counts.values()) == [40, 40],
        "exactly_16_oracle_reachable": s3_oracle_positive == 16,
        "all_oracle_positive_are_bridges": bool(
            (
                s3.truth.loc[
                    s3.truth["oracle_reachable"].astype(bool),
                    "is_accessible_bridge",
                ].astype(bool)
            ).all()
        ),
        "exactly_3_gate_features": s3_gate_n == 3,
        "exactly_6_reverse_lock_features": s3_reverse_lock_n == 6,
        "oracle_candidate_rows_16": (
            s3.oracle_candidates is not None and len(s3.oracle_candidates) == 16
        ),
        "forward_oracle_plausibility_rate_meets_frozen_smoke_min": (
            s3_plaus["forward_oracle_plausibility_rate"]
            >= IMPLEMENTATION_SPEC["S3_DIRECTIONAL_REACHABILITY"][
                "smoke_forward_plausibility_rate_min"
            ]
        ),
        "mirrored_reverse_three_gate_plausibility_below_frozen_smoke_max": (
            s3_plaus["mirrored_reverse_three_gate_plausibility_rate"]
            <= IMPLEMENTATION_SPEC["S3_DIRECTIONAL_REACHABILITY"][
                "smoke_mirrored_reverse_plausibility_rate_max"
            ]
        ),
    }

    s4 = bundles["S4_SEVERITY_CONTINUUM"]
    s4_merged = s4.data.merge(
        s4.truth[["institution_id", "latent_severity"]],
        on="institution_id",
        validate="one_to_one",
    )

    spearman_values = [
        float(
            s4_merged[[feature, "latent_severity"]].corr(method="spearman").iloc[0, 1]
        )
        for feature in FEATURE_NAMES
    ]

    s4_median_spearman = float(np.median(spearman_values))

    s4_checks = {
        "no_true_profiles": bool(s4.truth["true_profile"].isna().all()),
        "all_feature_severity_spearman_positive": bool(
            np.all(np.array(spearman_values) > 0)
        ),
        "median_spearman_meets_frozen_smoke_min": (
            s4_median_spearman
            >= IMPLEMENTATION_SPEC["S4_SEVERITY_CONTINUUM"][
                "smoke_median_spearman_with_z_min"
            ]
        ),
    }

    s5 = bundles["S5_GOVERNANCE_CONFOUNDED"]
    s5_merged = s5.data.merge(
        s5.truth[["institution_id", "governance_type"]],
        on="institution_id",
        validate="one_to_one",
    )

    s5_counts = s5_merged["governance_type"].astype(str).value_counts().to_dict()

    expected_s5_counts = IMPLEMENTATION_SPEC["S5_GOVERNANCE_CONFOUNDED"]["exact_counts"]

    s5_distances = pairwise_centroid_distances(
        s5_merged,
        s5_merged["governance_type"],
    )

    s5_checks = {
        "no_true_profiles": bool(s5.truth["true_profile"].isna().all()),
        "governance_counts_exact": (s5_counts == expected_s5_counts),
        "minimum_governance_centroid_distance_meets_smoke_min": (
            min(s5_distances)
            >= IMPLEMENTATION_SPEC["S5_GOVERNANCE_CONFOUNDED"][
                "smoke_min_pairwise_governance_centroid_distance_min"
            ]
        ),
    }

    s6 = bundles["S6_NO_CLUSTER_NULL"]
    s6_X = s6.data[FEATURE_NAMES]
    s6_corr = s6_X.corr(method="spearman").to_numpy(dtype=float)
    offdiag = s6_corr[
        np.triu_indices(
            13,
            k=1,
        )
    ]

    s6_checks = {
        "no_true_profiles": bool(s6.truth["true_profile"].isna().all()),
        "feature_means_are_interior": bool(
            s6_X.mean()
            .between(
                0.30,
                0.70,
            )
            .all()
        ),
        "empirical_mean_positive_dependence": (float(np.nanmean(offdiag)) > 0.0),
    }

    all_checks = {
        "stage8a_design_passed": (
            design.get("gate_status") == "PASS_STAGE_8A_SYNTHETIC_DESIGN_FREEZE"
        ),
        "implementation_freeze_written_and_matches_code": True,
        "all_scenarios_deterministic": all(determinism_checks.values()),
        "all_scenarios_common_integrity": all(common_checks.values()),
        **{f"S1_{key}": value for key, value in s1_checks.items()},
        **{f"S2_{key}": value for key, value in s2_checks.items()},
        **{f"S3_{key}": value for key, value in s3_checks.items()},
        **{f"S4_{key}": value for key, value in s4_checks.items()},
        **{f"S5_{key}": value for key, value in s5_checks.items()},
        **{f"S6_{key}": value for key, value in s6_checks.items()},
    }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "smoke_summary.csv",
        index=False,
    )

    report = {
        "implementation_freeze_path": str(IMPLEMENTATION_FREEZE_PATH.relative_to(ROOT)),
        "implementation_freeze_sha256": sha256_bytes(
            IMPLEMENTATION_FREEZE_PATH.read_bytes()
        ),
        "scenario_summary": summary.to_dict(orient="records"),
        "S3_plausibility_audit": s3_plaus,
        "S4_median_spearman_with_latent_severity": (s4_median_spearman),
        "S5_min_pairwise_governance_centroid_distance": float(min(s5_distances)),
        "S6_mean_offdiagonal_spearman": float(np.nanmean(offdiag)),
        "checks": all_checks,
        "gate_status": (
            "PASS_STAGE_8B_GENERATOR_SMOKE_AUDIT"
            if all(all_checks.values())
            else "FAIL_STAGE_8B_GENERATOR_SMOKE_AUDIT"
        ),
    }

    (OUTPUT_DIR / "stage8b_generator_smoke_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    AUDIT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== SCENARIO SMOKE SUMMARY ===\n")
    print(summary.to_string(index=False))

    print("\n=== S2 BOUNDARY GEOMETRY ===\n")
    print(
        "True boundary observations:",
        s2_boundary_n,
    )
    print(
        "Median distance to midpoint — boundary:",
        f"{np.median(distance_to_midpoint[boundary_mask]):.6f}",
    )
    print(
        "Median distance to midpoint — core:",
        f"{np.median(distance_to_midpoint[~boundary_mask]):.6f}",
    )

    print("\n=== S3 DIRECTIONAL REACHABILITY PLANT ===\n")
    print(
        "Accessible latent source:",
        s3.metadata["accessible_source_latent"],
    )
    print(
        "Target latent profile:",
        s3.metadata["target_latent"],
    )
    print(
        "Gate features:",
        s3.metadata["gate_feature_names"],
    )
    print(
        "Reverse-lock features:",
        s3.metadata["reverse_lock_feature_names"],
    )
    print(
        "Oracle reachable:",
        f"{s3_oracle_positive}/80",
    )
    print(
        "Forward 3-feature oracle plausibility rate:",
        f"{s3_plaus['forward_oracle_plausibility_rate']:.4f}",
    )
    print(
        "Mirrored reverse 3-gate plausibility rate:",
        f"{s3_plaus['mirrored_reverse_three_gate_plausibility_rate']:.4f}",
    )

    print("\n=== S4 SEVERITY CONTINUUM ===\n")
    print(
        "Median feature-vs-z Spearman:",
        f"{s4_median_spearman:.4f}",
    )
    print(
        "Feature Spearman range:",
        f"{min(spearman_values):.4f}",
        "to",
        f"{max(spearman_values):.4f}",
    )

    print("\n=== S5 GOVERNANCE CONFOUNDING ===\n")
    print(
        "Counts:",
        s5_counts,
    )
    print(
        "Minimum pairwise governance-centroid distance:",
        f"{min(s5_distances):.4f}",
    )

    print("\n=== S6 NO-CLUSTER NULL ===\n")
    print(
        "Mean off-diagonal feature Spearman:",
        f"{np.nanmean(offdiag):.4f}",
    )

    print("\n=== SMOKE/AUDIT CHECKS ===\n")
    for name, passed in all_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {report['gate_status']}")

    if report["gate_status"] == "PASS_STAGE_8B_GENERATOR_SMOKE_AUDIT":
        print(
            "Generator truth is implemented and audited. "
            "Do not run the 600 official replicates yet; review before Stage 8C."
        )
    else:
        print(
            "Do not run official synthetic replicates. "
            "Review the failed generator-truth check first."
        )


if __name__ == "__main__":
    main()
