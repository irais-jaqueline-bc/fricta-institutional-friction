from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

STAGE6_FREEZE_PATH = (
    ROOT / "cipher" / "design" / "stage6_ensemble_robustness_freeze.json"
)
STAGE6B_REPORT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "ensemble"
    / "stage6_reconstruction_audit_v2"
    / "stage6b_v2_report.json"
)
STAGE6C_REPORT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "ensemble_robustness"
    / "stage6c_report.json"
)

CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
FRICTA_CONFIG_PATH = ROOT / "icdm" / "design" / "experiment_config.json"

PRIMARY_MATRIX_PATH = ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
CF_MANIFEST_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "inductive_validation"
    / "counterfactual_ensemble_manifest.csv"
)

CANDIDATE_SUPPORT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "ensemble_robustness"
    / "candidate_member_target_support_matrix.csv"
)
CANDIDATE_SUMMARY_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "ensemble_robustness"
    / "candidate_ensemble_support.csv"
)
INSTITUTION_SUMMARY_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "ensemble_robustness"
    / "institution_robust_reachability.csv"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "counterfactuals" / "ensemble_lift_audit"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_list(value: str) -> list[Any]:
    obj = json.loads(value)
    if not isinstance(obj, list):
        raise ValueError("Expected JSON list.")
    return obj


def parse_label_mapping(value: str) -> dict[int, int]:
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object for label mapping.")
    return {int(k): int(v) for k, v in obj.items()}


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"true", "1", "yes"}
        )
    ).astype(bool)


def apply_mapping(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def fit_member_predictor(
    row: pd.Series,
    data_by_id: pd.DataFrame,
    kmeans_n_init: int,
    pca_threshold: float,
):
    member_id = str(row["member_id"])
    algorithm = str(row["algorithm"]).upper()
    representation = str(row["representation"]).upper()
    seed = int(row["seed"])

    features = [str(x) for x in parse_json_list(str(row["feature_names_json"]))]
    sampled_ids = [
        str(x) for x in parse_json_list(str(row["sampled_institution_ids_json"]))
    ]

    X_sample = data_by_id.loc[sampled_ids, features].to_numpy(dtype=float)

    scaler = StandardScaler()
    Z_sample_scaled = scaler.fit_transform(X_sample)

    if "PCA" in representation:
        pca = PCA(
            n_components=pca_threshold,
            svd_solver="full",
            random_state=seed,
        )
        Z_sample = pca.fit_transform(Z_sample_scaled)
    else:
        pca = None
        Z_sample = Z_sample_scaled

    mapping = parse_label_mapping(str(row["label_mapping_json"]))

    if algorithm == "KMEANS":
        model = KMeans(
            n_clusters=2,
            n_init=kmeans_n_init,
            random_state=seed,
        )
        model.fit(Z_sample)

        def predict(X_new: pd.DataFrame) -> np.ndarray:
            Z = scaler.transform(X_new[features].to_numpy(dtype=float))
            if pca is not None:
                Z = pca.transform(Z)
            return apply_mapping(model.predict(Z), mapping)

        return predict

    if "WARD" in algorithm:
        model = AgglomerativeClustering(n_clusters=2, linkage="ward")
        raw_sample = model.fit_predict(Z_sample)
        raw_values = sorted(np.unique(raw_sample).tolist())
        centroids = np.vstack(
            [Z_sample[raw_sample == raw_label].mean(axis=0) for raw_label in raw_values]
        )

        def predict(X_new: pd.DataFrame) -> np.ndarray:
            Z = scaler.transform(X_new[features].to_numpy(dtype=float))
            if pca is not None:
                Z = pca.transform(Z)

            distances = np.sqrt(
                ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            )
            nearest = np.argmin(distances, axis=1)
            raw = np.array([raw_values[idx] for idx in nearest], dtype=int)
            return apply_mapping(raw, mapping)

        return predict

    raise ValueError(f"{member_id}: unsupported algorithm {algorithm}")


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    freeze = load_json(STAGE6_FREEZE_PATH)
    stage6b = load_json(STAGE6B_REPORT_PATH)
    stage6c = load_json(STAGE6C_REPORT_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    fricta_config = load_json(FRICTA_CONFIG_PATH)

    if freeze.get("gate_status") != "PASS_STAGE_6A_DESIGN_FREEZE":
        raise ValueError("Stage 6A freeze has not passed.")
    if stage6b.get("gate_status") != "PASS_STAGE_6B_RECONSTRUCTION_AUDIT_V2":
        raise ValueError("Stage 6B v2 has not passed.")
    if stage6c.get("gate_status") != "PASS_STAGE_6C_OFFICIAL_EVALUATION":
        raise ValueError("Stage 6C official evaluation has not passed.")

    id_column = fricta_config["id_column"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    manifest = pd.read_csv(CF_MANIFEST_PATH)
    support_matrix = pd.read_csv(CANDIDATE_SUPPORT_PATH).set_index("candidate_id")
    candidate_summary = pd.read_csv(CANDIDATE_SUMMARY_PATH)
    institution_summary = pd.read_csv(INSTITUTION_SUMMARY_PATH)

    primary[id_column] = primary[id_column].astype(str)
    manifest["member_id"] = manifest["member_id"].astype(str)
    candidate_summary["institution_id"] = candidate_summary["institution_id"].astype(
        str
    )
    institution_summary["institution_id"] = institution_summary[
        "institution_id"
    ].astype(str)

    eligibility = parse_bool_series(manifest["eligible_for_counterfactual_ensemble"])
    eligible_manifest = manifest[eligibility].copy()

    member_ids = support_matrix.columns.astype(str).tolist()
    eligible_ids = eligible_manifest["member_id"].astype(str).tolist()

    frozen_ids = [str(x) for x in freeze["scope"]["institution_ids"]]
    original_matrix = primary.set_index(id_column).loc[frozen_ids].copy()

    prechecks = {
        "support_matrix_is_69_by_984": support_matrix.shape == (69, 984),
        "support_matrix_member_set_matches_eligible_manifest": set(member_ids)
        == set(eligible_ids),
        "exactly_19_frozen_institutions": len(frozen_ids) == 19,
        "candidate_summary_has_69_rows": len(candidate_summary) == 69,
        "institution_summary_has_19_rows": len(institution_summary) == 19,
    }

    print("\n=== CIPHER STAGE 6D — COUNTERFACTUAL ENSEMBLE-LIFT AUDIT ===\n")
    print("Purpose:")
    print(
        "  Contextualize Stage 6C support by asking how much each counterfactual "
        "actually changes ensemble predictions relative to the institution's "
        "original point under the same audited inductive rules."
    )
    print("\nPrechecks:")
    for name, passed in prechecks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(prechecks.values()):
        print("\nGATE STATUS: FAIL_STAGE_6D_PRECHECK")
        raise SystemExit(1)

    ensemble_cfg = cipher_config.get("ensemble", {})
    kmeans_n_init = int(ensemble_cfg.get("kmeans_n_init", 25))
    pca_threshold = float(ensemble_cfg.get("pca_variance_threshold", 0.85))

    data_by_id = primary.set_index(id_column)

    # Reorder manifest to the exact Stage-6C support-matrix member order.
    eligible_manifest = (
        eligible_manifest.set_index("member_id").loc[member_ids].reset_index()
    )

    original_predictions = np.zeros((19, 984), dtype=np.uint8)

    started = time.perf_counter()

    for member_position, (_, row) in enumerate(eligible_manifest.iterrows()):
        predictor = fit_member_predictor(
            row=row,
            data_by_id=data_by_id,
            kmeans_n_init=kmeans_n_init,
            pca_threshold=pca_threshold,
        )
        original_predictions[:, member_position] = predictor(original_matrix)

        if (member_position + 1) % 100 == 0:
            print(
                f"Reconstructed original inductive predictions "
                f"{member_position + 1:03d}/984...",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    original_prediction_df = pd.DataFrame(
        original_predictions,
        index=frozen_ids,
        columns=member_ids,
    )
    original_prediction_df.index.name = "institution_id"
    original_prediction_df.to_csv(
        OUTPUT_DIR / "original_inductive_predictions_19x984.csv"
    )

    original_pos = {
        institution_id: idx for idx, institution_id in enumerate(frozen_ids)
    }

    family_positions = {}
    for family, group in eligible_manifest.groupby("family", sort=True):
        family_positions[str(family)] = group.index.to_numpy(dtype=int)

    rows = []

    support_matrix = support_matrix.loc[
        candidate_summary["candidate_id"].astype(str).tolist(),
        member_ids,
    ]

    for row_idx, cf_row in candidate_summary.reset_index(drop=True).iterrows():
        candidate_id = str(cf_row["candidate_id"])
        institution_id = str(cf_row["institution_id"])
        reference_profile = int(cf_row["reference_profile"])
        target_profile = int(cf_row["target_profile"])

        base_pred = original_predictions[original_pos[institution_id]].astype(int)

        cf_target = (
            support_matrix.loc[candidate_id].to_numpy(dtype=np.uint8).astype(bool)
        )

        base_target = base_pred == target_profile
        base_source = base_pred == reference_profile

        desired_flip = base_source & cf_target
        retained_target = base_target & cf_target
        reverse_loss = base_target & (~cf_target)
        remained_source = base_source & (~cf_target)

        record = {
            "candidate_id": candidate_id,
            "institution_id": institution_id,
            "reference_profile": reference_profile,
            "target_profile": target_profile,
            "certainty_class": str(cf_row["certainty_class"]),
            "rank": int(cf_row["rank"]),
            "total_cost": float(cf_row["total_cost"]),
            "ensemble_support": float(cf_row["ensemble_support"]),
            "baseline_target_support_inductive": float(base_target.mean()),
            "baseline_source_support_inductive": float(base_source.mean()),
            "ensemble_support_gain": float(cf_target.mean() - base_target.mean()),
            "desired_flip_members": int(desired_flip.sum()),
            "retained_target_members": int(retained_target.sum()),
            "reverse_loss_members": int(reverse_loss.sum()),
            "remained_source_members": int(remained_source.sum()),
            "conditional_flip_rate_among_baseline_source": safe_rate(
                int(desired_flip.sum()),
                int(base_source.sum()),
            ),
            "target_retention_rate_among_baseline_target": safe_rate(
                int(retained_target.sum()),
                int(base_target.sum()),
            ),
            "reverse_loss_rate_among_baseline_target": safe_rate(
                int(reverse_loss.sum()),
                int(base_target.sum()),
            ),
            "net_member_gain_count": int(desired_flip.sum() - reverse_loss.sum()),
        }

        for family, positions in family_positions.items():
            family_base_target = base_target[positions]
            family_base_source = base_source[positions]
            family_cf_target = cf_target[positions]
            family_desired_flip = family_base_source & family_cf_target

            record[f"baseline_target_support_{family}"] = float(
                family_base_target.mean()
            )
            record[f"cf_target_support_{family}"] = float(family_cf_target.mean())
            record[f"support_gain_{family}"] = float(
                family_cf_target.mean() - family_base_target.mean()
            )
            record[f"conditional_flip_rate_{family}"] = safe_rate(
                int(family_desired_flip.sum()),
                int(family_base_source.sum()),
            )

        rows.append(record)

    lift = pd.DataFrame(rows)
    lift.to_csv(
        OUTPUT_DIR / "candidate_ensemble_lift.csv",
        index=False,
    )

    primary_tau = float(freeze["candidate_evaluation"]["primary_tau"])
    best_col = f"best_candidate_id_tau_{primary_tau:.2f}"
    robust_col = f"robustly_reachable_tau_{primary_tau:.2f}"

    selected_rows = []

    for _, inst in institution_summary.iterrows():
        institution_id = str(inst["institution_id"])
        robust = bool(inst[robust_col])

        if robust:
            candidate_id = str(inst[best_col])
        else:
            candidate_id = str(inst["best_available_candidate_id"])

        row = lift[lift["candidate_id"] == candidate_id]

        if len(row) != 1:
            raise ValueError(
                f"{institution_id}: could not uniquely resolve selected candidate {candidate_id}"
            )

        selected = row.iloc[0].to_dict()
        selected["robustly_reachable_tau_0.90"] = robust
        selected["selection_basis"] = (
            "PRIMARY_ROBUST_BEST" if robust else "MAX_SUPPORT_NONROBUST"
        )
        selected_rows.append(selected)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(
        OUTPUT_DIR / "institution_selected_candidate_lift.csv",
        index=False,
    )

    robust_selected = selected[
        selected["robustly_reachable_tau_0.90"].astype(bool)
    ].copy()

    nonrobust_selected = selected[
        ~selected["robustly_reachable_tau_0.90"].astype(bool)
    ].copy()

    profile_rows = []
    for profile, group in selected.groupby("reference_profile", sort=True):
        profile_rows.append(
            {
                "reference_profile": int(profile),
                "n_institutions": len(group),
                "robust_n_tau_0.90": int(
                    group["robustly_reachable_tau_0.90"].astype(bool).sum()
                ),
                "median_baseline_target_support": float(
                    group["baseline_target_support_inductive"].median()
                ),
                "median_selected_candidate_support": float(
                    group["ensemble_support"].median()
                ),
                "median_support_gain": float(group["ensemble_support_gain"].median()),
                "median_conditional_flip_rate": float(
                    group["conditional_flip_rate_among_baseline_source"].median()
                ),
            }
        )

    profile_summary = pd.DataFrame(profile_rows)
    profile_summary.to_csv(
        OUTPUT_DIR / "selected_candidate_lift_by_profile.csv",
        index=False,
    )

    robust_gain_positive = (
        bool((robust_selected["ensemble_support_gain"] > 0).all())
        if len(robust_selected)
        else True
    )

    checks = {
        "all_69_candidates_have_lift_metrics": len(lift) == 69,
        "all_19_institutions_have_selected_candidate_context": len(selected) == 19,
        "exactly_10_primary_robust_institutions": len(robust_selected) == 10,
        "exactly_9_primary_nonrobust_institutions": len(nonrobust_selected) == 9,
        "all_supports_and_flip_rates_in_valid_ranges": bool(
            (
                lift["baseline_target_support_inductive"].between(0, 1)
                & lift["ensemble_support"].between(0, 1)
                & lift["conditional_flip_rate_among_baseline_source"]
                .dropna()
                .between(0, 1)
                & lift["target_retention_rate_among_baseline_target"]
                .dropna()
                .between(0, 1)
            ).all()
        ),
        "stage6c_support_recomputed_from_matrix_matches_summary": bool(
            np.allclose(
                lift["ensemble_support"].to_numpy(dtype=float),
                support_matrix.mean(axis=1).to_numpy(dtype=float),
                atol=1e-12,
                rtol=0,
            )
        ),
    }

    report = {
        "status": "EXPLORATORY_POST_STAGE6C_CONTEXT_AUDIT",
        "purpose": (
            "Distinguish high absolute target-profile support from actual "
            "counterfactual prediction change relative to each original institution."
        ),
        "robust_selected_all_have_positive_support_gain": robust_gain_positive,
        "robust_selected_support_gain_quantiles": (
            {
                "min": float(robust_selected["ensemble_support_gain"].min()),
                "median": float(robust_selected["ensemble_support_gain"].median()),
                "max": float(robust_selected["ensemble_support_gain"].max()),
            }
            if len(robust_selected)
            else {}
        ),
        "profile_summary": profile_summary.to_dict(orient="records"),
        "checks": checks,
        "interpretation_boundary": (
            "This audit does not alter Stage 6C's frozen robust-validity definition. "
            "It contextualizes it by measuring ensemble support gain and member-level "
            "source-to-target flips under the same audited inductive rules."
        ),
        "elapsed_seconds": float(elapsed),
        "gate_status": (
            "PASS_STAGE_6D_LIFT_AUDIT"
            if all(checks.values())
            else "FAIL_STAGE_6D_LIFT_AUDIT"
        ),
    }

    (OUTPUT_DIR / "stage6d_lift_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (AUDIT_DIR / "stage6d_lift_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== SELECTED CANDIDATE LIFT — ALL 19 INSTITUTIONS ===\n")
    print(
        selected[
            [
                "institution_id",
                "reference_profile",
                "certainty_class",
                "robustly_reachable_tau_0.90",
                "candidate_id",
                "baseline_target_support_inductive",
                "ensemble_support",
                "ensemble_support_gain",
                "conditional_flip_rate_among_baseline_source",
                "target_retention_rate_among_baseline_target",
            ]
        ]
        .sort_values(
            [
                "robustly_reachable_tau_0.90",
                "ensemble_support_gain",
            ],
            ascending=[False, False],
        )
        .to_string(index=False)
    )

    print("\n=== SELECTED CANDIDATE LIFT BY REFERENCE PROFILE ===\n")
    print(profile_summary.to_string(index=False))

    print("\n=== ROBUST TAU=0.90 CASES ONLY ===\n")
    print(
        robust_selected[
            [
                "institution_id",
                "certainty_class",
                "candidate_id",
                "baseline_target_support_inductive",
                "ensemble_support",
                "ensemble_support_gain",
                "desired_flip_members",
                "conditional_flip_rate_among_baseline_source",
            ]
        ]
        .sort_values("ensemble_support_gain", ascending=False)
        .to_string(index=False)
    )

    print("\n=== TECHNICAL CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(
        "\nAll primary-robust selected candidates have positive support gain:",
        robust_gain_positive,
    )
    print("Elapsed seconds:", f"{elapsed:.2f}")
    print(f"\nGATE STATUS: {report['gate_status']}")
    print(
        "This is a contextual audit, not a new robustness threshold. "
        "Review before Stage 7 motif mining."
    )


if __name__ == "__main__":
    main()
