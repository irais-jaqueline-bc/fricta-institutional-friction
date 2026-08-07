from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "cipher" / "outputs" / "ensemble" / "official"
AUDIT = ROOT / "cipher" / "outputs" / "audit"

EXPECTED = {
    "R0_WARD": 250,
    "R1_PCA85_WARD": 250,
    "R0_KMEANS": 250,
    "R1_PCA85_KMEANS": 250,
}


def req(name):
    p = OFFICIAL / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def opt(name):
    p = OFFICIAL / name
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def mat(df):
    return df.drop(columns=[df.columns[0]]).to_numpy(dtype=float)


def q(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    probs = [0.025, 0.25, 0.5, 0.75, 0.975]
    vals = s.quantile(probs)
    return {str(p): float(vals.loc[p]) for p in probs}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)

    report = json.loads((OFFICIAL / "ensemble_report.json").read_text(encoding="utf-8"))
    manifest = req("member_manifest.csv")
    metrics = req("member_metrics.csv")
    pred_all = req("member_predictions_all.csv")
    pred_oob = req("member_predictions_oob.csv")
    family_consensus = req("family_consensus.csv")
    coassign = req("coassignment_matrix.csv")
    coassign_oob = req("coassignment_oob_matrix.csv")
    coassign_oob_counts = req("coassignment_oob_counts.csv")
    rejected = opt("rejected_attempts.csv")

    checks = []
    checks.append(("report_mode_official", report.get("mode") == "official"))
    checks.append(("report_gate_pass", report.get("gate_status") == "PASS_STAGE_1"))
    checks.append(("accepted_members_1000", report.get("accepted_members") == 1000))
    checks.append(("manifest_rows_1000", len(manifest) == 1000))
    checks.append(("metrics_rows_1000", len(metrics) == 1000))
    checks.append(("member_ids_unique", manifest["member_id"].is_unique))
    checks.append(
        ("family_counts_exact", manifest["family"].value_counts().to_dict() == EXPECTED)
    )
    checks.append(("sample_size_65", bool((manifest["sample_size"] == 65).all())))
    checks.append(("oob_size_16", bool((manifest["oob_size"] == 16).all())))
    checks.append(("feature_count_11", bool((manifest["feature_count"] == 11).all())))

    pca = manifest["representation"] == "PCA85"
    raw = manifest["representation"] == "RAW_STANDARDIZED"
    checks.append(
        (
            "pca_components_present",
            bool(manifest.loc[pca, "pca_components"].notna().all()),
        )
    )
    checks.append(
        (
            "pca_variance_at_least_085",
            bool((manifest.loc[pca, "pca_explained_variance"] >= 0.85 - 1e-12).all()),
        )
    )
    checks.append(
        ("raw_pca_fields_empty", bool(manifest.loc[raw, "pca_components"].isna().all()))
    )

    finite_cols = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "reference_ari_on_sample",
        "ward_or_native_extension_fidelity_on_sample",
        "minimum_sampled_cluster_size",
    ]
    checks.append(
        (
            "all_metrics_finite",
            bool(np.isfinite(metrics[finite_cols].to_numpy(dtype=float)).all()),
        )
    )
    checks.append(
        (
            "minimum_cluster_size_at_least_5",
            bool((metrics["minimum_sampled_cluster_size"] >= 5).all()),
        )
    )

    members = manifest["member_id"].tolist()
    checks.append(("prediction_rows_81", len(pred_all) == 81))
    checks.append(("oob_prediction_rows_81", len(pred_oob) == 81))
    checks.append(
        (
            "prediction_columns_match",
            set(pred_all.columns) == {"institution_id", *members},
        )
    )
    checks.append(
        (
            "oob_prediction_columns_match",
            set(pred_oob.columns) == {"institution_id", *members},
        )
    )

    vals = {
        int(v) for v in pd.unique(pred_all[members].to_numpy().ravel()) if pd.notna(v)
    }
    checks.append(("all_predictions_only_profiles_1_2", vals == {1, 2}))

    oob_by_member = pred_oob[members].notna().sum(axis=0)
    oob_by_inst = pred_oob[members].notna().sum(axis=1)
    checks.append(("exactly_16_oob_per_member", bool((oob_by_member == 16).all())))
    checks.append(
        ("minimum_oob_per_institution_at_least_150", int(oob_by_inst.min()) >= 150)
    )
    checks.append(("family_consensus_rows_324", len(family_consensus) == 324))

    A = mat(coassign)
    B = mat(coassign_oob)
    C = mat(coassign_oob_counts)
    upper = np.triu_indices(81, 1)

    checks.append(("coassignment_shape_81x81", A.shape == (81, 81)))
    checks.append(("coassignment_symmetric", bool(np.allclose(A, A.T, equal_nan=True))))
    checks.append(("coassignment_diagonal_1", bool(np.allclose(np.diag(A), 1.0))))
    checks.append(
        ("coassignment_range_0_1", bool(np.nanmin(A) >= 0 and np.nanmax(A) <= 1))
    )
    checks.append(("oob_coassignment_shape_81x81", B.shape == (81, 81)))
    checks.append(
        ("oob_coassignment_symmetric", bool(np.allclose(B, B.T, equal_nan=True)))
    )
    checks.append(("oob_count_matrix_symmetric", bool(np.allclose(C, C.T))))
    min_pair = int(np.nanmin(C[upper]))
    checks.append(("minimum_oob_pair_count_at_least_10", min_pair >= 10))
    checks.append(
        (
            "acceptance_rate_at_least_090",
            float(report["overall_acceptance_rate"]) >= 0.90,
        )
    )
    checks.append(
        (
            "every_family_has_both_profiles",
            bool(report["every_family_has_both_profiles"]),
        )
    )

    ward = metrics[metrics["family"].str.contains("WARD")].copy()
    ward_fidelity = ward["ward_or_native_extension_fidelity_on_sample"]
    ward_below_095 = int((ward_fidelity < 0.95).sum())
    ward_below_090 = int((ward_fidelity < 0.90).sum())

    rows = []
    for family, g in metrics.groupby("family", sort=True):
        rows.append(
            {
                "family": family,
                "members": len(g),
                "silhouette_q025": float(g["silhouette"].quantile(0.025)),
                "silhouette_median": float(g["silhouette"].median()),
                "silhouette_q975": float(g["silhouette"].quantile(0.975)),
                "reference_ari_q025": float(
                    g["reference_ari_on_sample"].quantile(0.025)
                ),
                "reference_ari_median": float(g["reference_ari_on_sample"].median()),
                "reference_ari_q975": float(
                    g["reference_ari_on_sample"].quantile(0.975)
                ),
                "extension_fidelity_min": float(
                    g["ward_or_native_extension_fidelity_on_sample"].min()
                ),
                "minimum_cluster_size_min": int(
                    g["minimum_sampled_cluster_size"].min()
                ),
            }
        )
    fam = pd.DataFrame(rows)

    print("\n=== CIPHER STAGE 1 OFFICIAL AUDIT ===\n")
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print("\n=== FAMILY METRIC DISTRIBUTIONS ===\n")
    print(fam.to_string(index=False))

    print("\n=== GLOBAL DISTRIBUTIONS ===\n")
    print("Reference ARI quantiles:", q(metrics["reference_ari_on_sample"]))
    print("Silhouette quantiles:", q(metrics["silhouette"]))
    print("Ward extension fidelity quantiles:", q(ward_fidelity))
    print("Ward members below 0.95 fidelity:", ward_below_095, "of", len(ward))
    print("Ward members below 0.90 fidelity:", ward_below_090, "of", len(ward))
    print(
        "Minimum sampled cluster size:",
        int(metrics["minimum_sampled_cluster_size"].min()),
    )

    print("\n=== OOB COVERAGE ===\n")
    print(
        f"Per institution: min={int(oob_by_inst.min())}, median={float(oob_by_inst.median()):.1f}, max={int(oob_by_inst.max())}"
    )
    print(
        f"Pairwise OOB counts: min={min_pair}, median={float(np.median(C[upper])):.1f}, max={int(np.max(C[upper]))}"
    )
    print("Rejected attempts:", len(rejected))

    failed = [name for name, ok in checks if not ok]

    summary = {
        "gate_checks": {name: ok for name, ok in checks},
        "failed_checks": failed,
        "family_summary": rows,
        "global_reference_ari_quantiles": q(metrics["reference_ari_on_sample"]),
        "global_silhouette_quantiles": q(metrics["silhouette"]),
        "ward_extension_fidelity_quantiles": q(ward_fidelity),
        "ward_members_below_095_fidelity": ward_below_095,
        "ward_members_below_090_fidelity": ward_below_090,
        "oob_predictions_per_institution": {
            "minimum": int(oob_by_inst.min()),
            "median": float(oob_by_inst.median()),
            "maximum": int(oob_by_inst.max()),
        },
        "oob_pair_counts": {
            "minimum": min_pair,
            "median": float(np.median(C[upper])),
            "maximum": int(np.max(C[upper])),
        },
        "rejected_attempts": int(len(rejected)),
        "gate_status": (
            "PASS_STAGE_1_OFFICIAL_AUDIT"
            if not failed
            else "FAIL_STAGE_1_OFFICIAL_AUDIT"
        ),
    }
    (AUDIT / "stage1_official_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fam.to_csv(AUDIT / "stage1_family_metric_summary.csv", index=False)

    print(
        "\nNOTE: Ward extension fidelity is informational at Stage 1; Stage 4 will decide which Ward members enter counterfactual prediction."
    )
    print(
        "NOTE: Reference ARI is descriptive only; members were not accepted/rejected based on agreement with the frozen partition."
    )

    if failed:
        print("\nGATE STATUS: FAIL_STAGE_1_OFFICIAL_AUDIT")
        print("Failed checks:", ", ".join(failed))
        raise SystemExit(1)

    print("\nGATE STATUS: PASS_STAGE_1_OFFICIAL_AUDIT")
    print("Stage 1 is complete. Stage 2 may begin after metric review.")


if __name__ == "__main__":
    main()
