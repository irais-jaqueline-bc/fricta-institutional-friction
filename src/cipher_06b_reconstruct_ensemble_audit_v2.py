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
CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
FRICTA_CONFIG_PATH = ROOT / "icdm" / "design" / "experiment_config.json"

PRIMARY_MATRIX_PATH = ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
FINAL_LABELS_PATH = (
    ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)

OFFICIAL_MANIFEST_PATH = (
    ROOT / "cipher" / "outputs" / "ensemble" / "official" / "member_manifest.csv"
)
CF_MANIFEST_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "inductive_validation"
    / "counterfactual_ensemble_manifest.csv"
)
PREDICTIONS_ALL_PATH = (
    ROOT / "cipher" / "outputs" / "ensemble" / "official" / "member_predictions_all.csv"
)
PREDICTIONS_OOB_PATH = (
    ROOT / "cipher" / "outputs" / "ensemble" / "official" / "member_predictions_oob.csv"
)
WARD_FIDELITY_PATH = (
    ROOT / "cipher" / "outputs" / "inductive_validation" / "ward_extension_fidelity.csv"
)
EXCLUDED_WARD_PATH = (
    ROOT / "cipher" / "outputs" / "inductive_validation" / "excluded_ward_members.csv"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "ensemble" / "stage6_reconstruction_audit_v2"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"

EXPECTED_ELIGIBLE_FAMILY_COUNTS = {
    "R0_KMEANS": 250,
    "R0_WARD": 244,
    "R1_PCA85_KMEANS": 250,
    "R1_PCA85_WARD": 240,
}


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


def apply_mapping(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.array(
        [mapping[int(label)] for label in labels],
        dtype=int,
    )


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"true", "1", "yes"}
        )
    ).astype(bool)


def fit_member(
    row: pd.Series,
    data_by_id: pd.DataFrame,
    all_ids: list[str],
    kmeans_n_init: int,
    pca_threshold: float,
) -> dict[str, Any]:
    member_id = str(row["member_id"])
    algorithm = str(row["algorithm"]).upper()
    representation = str(row["representation"]).upper()
    seed = int(row["seed"])

    features = [str(x) for x in parse_json_list(str(row["feature_names_json"]))]
    sampled_ids = [
        str(x) for x in parse_json_list(str(row["sampled_institution_ids_json"]))
    ]
    oob_ids = [str(x) for x in parse_json_list(str(row["oob_institution_ids_json"]))]

    X_sample = data_by_id.loc[sampled_ids, features].to_numpy(dtype=float)
    X_all = data_by_id.loc[all_ids, features].to_numpy(dtype=float)

    scaler = StandardScaler()
    Z_sample_scaled = scaler.fit_transform(X_sample)
    Z_all_scaled = scaler.transform(X_all)

    is_pca = "PCA" in representation

    if is_pca:
        pca = PCA(
            n_components=pca_threshold,
            svd_solver="full",
            random_state=seed,
        )
        Z_sample = pca.fit_transform(Z_sample_scaled)
        Z_all = pca.transform(Z_all_scaled)
        pca_components = int(pca.n_components_)
        pca_variance = float(np.sum(pca.explained_variance_ratio_))
    else:
        pca = None
        Z_sample = Z_sample_scaled
        Z_all = Z_all_scaled
        pca_components = 0
        pca_variance = np.nan

    id_to_pos = {institution_id: i for i, institution_id in enumerate(all_ids)}
    sampled_pos = np.array([id_to_pos[i] for i in sampled_ids], dtype=int)
    oob_pos = np.array([id_to_pos[i] for i in oob_ids], dtype=int)

    mapping = parse_label_mapping(str(row["label_mapping_json"]))

    if algorithm == "KMEANS":
        model = KMeans(
            n_clusters=2,
            n_init=kmeans_n_init,
            random_state=seed,
        )
        raw_sample_native = model.fit_predict(Z_sample)
        raw_all_inductive = model.predict(Z_all)

        sample_native = apply_mapping(raw_sample_native, mapping)
        all_inductive = apply_mapping(raw_all_inductive, mapping)

        # For KMeans, native training assignment and inductive prediction are the same rule.
        sample_inductive = all_inductive[sampled_pos]
        extension_fidelity = float(np.mean(sample_inductive == sample_native))

    elif "WARD" in algorithm:
        model = AgglomerativeClustering(
            n_clusters=2,
            linkage="ward",
        )
        raw_sample_native = model.fit_predict(Z_sample)
        sample_native = apply_mapping(raw_sample_native, mapping)

        raw_values = sorted(np.unique(raw_sample_native).tolist())
        raw_centroids = {
            int(raw_label): Z_sample[raw_sample_native == raw_label].mean(axis=0)
            for raw_label in raw_values
        }
        centroid_matrix = np.vstack([raw_centroids[int(v)] for v in raw_values])

        distances_all = np.sqrt(
            ((Z_all[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
        )
        nearest_all = np.argmin(distances_all, axis=1)
        raw_all_inductive = np.array(
            [raw_values[idx] for idx in nearest_all],
            dtype=int,
        )
        all_inductive = apply_mapping(raw_all_inductive, mapping)

        sample_inductive = all_inductive[sampled_pos]
        extension_fidelity = float(np.mean(sample_inductive == sample_native))

    else:
        raise ValueError(f"{member_id}: unsupported algorithm {algorithm}")

    # Reconstruct the Stage-1 stored ALL matrix semantics:
    # native labels for sampled rows + inductive predictions for OOB rows.
    hybrid_all = all_inductive.copy()
    hybrid_all[sampled_pos] = sample_native

    return {
        "member_id": member_id,
        "family": str(row["family"]),
        "algorithm": algorithm,
        "representation": representation,
        "features": features,
        "sampled_ids": sampled_ids,
        "oob_ids": oob_ids,
        "sampled_pos": sampled_pos,
        "oob_pos": oob_pos,
        "sample_native": sample_native,
        "sample_inductive": sample_inductive,
        "all_inductive": all_inductive,
        "hybrid_all": hybrid_all,
        "extension_fidelity": extension_fidelity,
        "pca_components": pca_components,
        "pca_variance": pca_variance,
    }


def numeric_column(frame: pd.DataFrame, member_id: str) -> np.ndarray:
    values = pd.to_numeric(
        frame[member_id],
        errors="coerce",
    ).to_numpy(dtype=float)
    return values


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    stage6_freeze = load_json(STAGE6_FREEZE_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    fricta_config = load_json(FRICTA_CONFIG_PATH)

    if stage6_freeze.get("gate_status") != "PASS_STAGE_6A_DESIGN_FREEZE":
        raise ValueError("Stage 6A design freeze has not passed.")

    id_column = fricta_config["id_column"]

    manifest = pd.read_csv(OFFICIAL_MANIFEST_PATH)
    cf_manifest = pd.read_csv(CF_MANIFEST_PATH)
    predictions_all = pd.read_csv(PREDICTIONS_ALL_PATH)
    predictions_oob = pd.read_csv(PREDICTIONS_OOB_PATH)
    ward_fidelity = pd.read_csv(WARD_FIDELITY_PATH)
    excluded_ward = pd.read_csv(EXCLUDED_WARD_PATH)

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[[id_column, "cluster_id"]]

    primary[id_column] = primary[id_column].astype(str)
    labels[id_column] = labels[id_column].astype(str)
    predictions_all["institution_id"] = predictions_all["institution_id"].astype(str)
    predictions_oob["institution_id"] = predictions_oob["institution_id"].astype(str)
    manifest["member_id"] = manifest["member_id"].astype(str)
    cf_manifest["member_id"] = cf_manifest["member_id"].astype(str)

    data = primary.merge(
        labels,
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 institutions; found {len(data)}.")

    all_ids = data[id_column].astype(str).tolist()
    data_by_id = data.set_index(id_column)

    predictions_all = predictions_all.set_index("institution_id").loc[all_ids]
    predictions_oob = predictions_oob.set_index("institution_id").loc[all_ids]

    eligibility = parse_bool_series(cf_manifest["eligible_for_counterfactual_ensemble"])
    cf_manifest = cf_manifest.copy()
    cf_manifest["_eligible_bool"] = eligibility

    eligible = cf_manifest[cf_manifest["_eligible_bool"]].copy()
    excluded = cf_manifest[~cf_manifest["_eligible_bool"]].copy()

    family_counts = eligible["family"].value_counts().to_dict()

    static_checks = {
        "manifest_1000": len(manifest) == 1000,
        "cf_manifest_1000": len(cf_manifest) == 1000,
        "eligible_984": len(eligible) == 984,
        "excluded_16": len(excluded) == 16,
        "eligible_family_counts_match": all(
            int(family_counts.get(family, 0)) == expected
            for family, expected in EXPECTED_ELIGIBLE_FAMILY_COUNTS.items()
        ),
        "all_matrix_81_rows": len(predictions_all) == 81,
        "oob_matrix_81_rows": len(predictions_oob) == 81,
        "all_matrix_1000_member_columns": len(predictions_all.columns) == 1000,
        "oob_matrix_1000_member_columns": len(predictions_oob.columns) == 1000,
        "excluded_file_16": len(excluded_ward) == 16,
        "ward_fidelity_500": len(ward_fidelity) == 500,
    }

    print("\n=== CIPHER STAGE 6B v2 — WARD SEMANTICS AUDIT ===\n")
    print("Static checks:")
    for name, passed in static_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(static_checks.values()):
        print("\nGATE STATUS: FAIL_STAGE_6B_V2_STATIC")
        raise SystemExit(1)

    ensemble_cfg = cipher_config.get("ensemble", {})
    kmeans_n_init = int(ensemble_cfg.get("kmeans_n_init", 25))
    pca_threshold = float(ensemble_cfg.get("pca_variance_threshold", 0.85))

    ward_fidelity_lookup = (
        ward_fidelity.set_index("member_id")["recomputed_extension_fidelity"]
        .astype(float)
        .to_dict()
    )

    audit_rows = []
    started = time.perf_counter()

    for idx, row in manifest.iterrows():
        fitted = fit_member(
            row=row,
            data_by_id=data_by_id,
            all_ids=all_ids,
            kmeans_n_init=kmeans_n_init,
            pca_threshold=pca_threshold,
        )

        member_id = fitted["member_id"]
        algorithm = fitted["algorithm"]

        stored_all = numeric_column(predictions_all, member_id)
        stored_oob = numeric_column(predictions_oob, member_id)

        # ALL matrix comparison using its actual hybrid semantics.
        if np.isnan(stored_all).any():
            raise ValueError(f"{member_id}: stored ALL matrix unexpectedly has NaN.")

        stored_all_int = stored_all.astype(int)
        hybrid_mismatch_count = int(np.sum(stored_all_int != fitted["hybrid_all"]))

        # For Stage 6, new counterfactual points are out-of-sample. The relevant
        # auditable rule is therefore the inductive rule on each member's OOB rows.
        oob_pos = fitted["oob_pos"]
        stored_oob_values = stored_oob[oob_pos]
        expected_oob_inductive = fitted["all_inductive"][oob_pos]

        if np.isnan(stored_oob_values).any():
            raise ValueError(
                f"{member_id}: OOB matrix contains NaN on declared OOB institutions."
            )

        oob_mismatch_count = int(
            np.sum(stored_oob_values.astype(int) != expected_oob_inductive.astype(int))
        )

        # On sampled rows, the OOB matrix should generally be blank/NaN.
        sampled_oob_cells = stored_oob[fitted["sampled_pos"]]
        sampled_oob_nonmissing = int(np.sum(~np.isnan(sampled_oob_cells)))

        # Diagnose exactly why the previous audit failed:
        # raw inductive predictions differ from the hybrid ALL matrix only where
        # Ward nearest-centroid extension disagrees with Ward native training labels.
        inductive_vs_hybrid_mismatches = int(
            np.sum(fitted["all_inductive"] != fitted["hybrid_all"])
        )

        sample_extension_disagreements = int(
            np.sum(fitted["sample_inductive"] != fitted["sample_native"])
        )

        localization_identity = (
            inductive_vs_hybrid_mismatches == sample_extension_disagreements
        )

        if "WARD" in algorithm:
            stored_fidelity = float(ward_fidelity_lookup[member_id])
        else:
            stored_fidelity = 1.0

        fidelity_diff = abs(stored_fidelity - fitted["extension_fidelity"])

        audit_rows.append(
            {
                "member_id": member_id,
                "family": fitted["family"],
                "algorithm": algorithm,
                "representation": fitted["representation"],
                "eligible": bool(
                    cf_manifest.set_index("member_id").loc[member_id, "_eligible_bool"]
                ),
                "hybrid_all_mismatch_count": hybrid_mismatch_count,
                "oob_inductive_mismatch_count": oob_mismatch_count,
                "sampled_oob_nonmissing_cells": sampled_oob_nonmissing,
                "inductive_vs_hybrid_mismatches": inductive_vs_hybrid_mismatches,
                "sample_extension_disagreements": sample_extension_disagreements,
                "mismatch_localization_identity": localization_identity,
                "stored_extension_fidelity": stored_fidelity,
                "recomputed_extension_fidelity": fitted["extension_fidelity"],
                "fidelity_absolute_difference": fidelity_diff,
            }
        )

        if (idx + 1) % 100 == 0:
            print(f"Audited {idx + 1:04d}/1000 members...", flush=True)

    elapsed = time.perf_counter() - started
    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        OUTPUT_DIR / "member_semantics_audit.csv",
        index=False,
    )

    eligible_audit = audit[audit["eligible"].astype(bool)].copy()

    family_summary = (
        eligible_audit.groupby("family", sort=True)
        .agg(
            members=("member_id", "size"),
            hybrid_all_mismatches=("hybrid_all_mismatch_count", "sum"),
            oob_inductive_mismatches=("oob_inductive_mismatch_count", "sum"),
            sample_extension_disagreements=("sample_extension_disagreements", "sum"),
            sampled_oob_nonmissing_cells=("sampled_oob_nonmissing_cells", "sum"),
            fidelity_mismatches=(
                "fidelity_absolute_difference",
                lambda s: int((s.astype(float) > 1e-12).sum()),
            ),
        )
        .reset_index()
    )

    family_summary.to_csv(
        OUTPUT_DIR / "eligible_family_semantics_summary.csv",
        index=False,
    )

    ward_rows = audit[audit["algorithm"].astype(str).str.contains("WARD")].copy()
    kmeans_rows = audit[audit["algorithm"].astype(str).str.contains("KMEANS")].copy()

    checks = {
        "all_1000_members_audited": len(audit) == 1000,
        "stored_all_matrix_reconstructed_with_hybrid_semantics": bool(
            (audit["hybrid_all_mismatch_count"] == 0).all()
        ),
        "all_declared_oob_predictions_match_inductive_rule": bool(
            (audit["oob_inductive_mismatch_count"] == 0).all()
        ),
        "oob_matrix_blank_on_sampled_rows": bool(
            (audit["sampled_oob_nonmissing_cells"] == 0).all()
        ),
        "ward_previous_mismatches_localize_exactly_to_sample_extension_disagreements": bool(
            ward_rows["mismatch_localization_identity"].astype(bool).all()
        ),
        "kmeans_native_and_inductive_rules_agree": bool(
            (kmeans_rows["sample_extension_disagreements"] == 0).all()
        ),
        "all_extension_fidelities_reproduce": bool(
            (audit["fidelity_absolute_difference"] <= 1e-12).all()
        ),
        "all_984_eligible_members_have_exact_oob_inductive_reconstruction": bool(
            (eligible_audit["oob_inductive_mismatch_count"] == 0).all()
        ),
    }

    report = {
        "members_audited": int(len(audit)),
        "eligible_members": int(len(eligible_audit)),
        "ward_members": int(len(ward_rows)),
        "kmeans_members": int(len(kmeans_rows)),
        "total_hybrid_all_mismatches": int(audit["hybrid_all_mismatch_count"].sum()),
        "total_oob_inductive_mismatches": int(
            audit["oob_inductive_mismatch_count"].sum()
        ),
        "total_sample_extension_disagreements": int(
            audit["sample_extension_disagreements"].sum()
        ),
        "eligible_sample_extension_disagreements": int(
            eligible_audit["sample_extension_disagreements"].sum()
        ),
        "elapsed_seconds": float(elapsed),
        "checks": checks,
        "interpretation": (
            "The Stage-1 member_predictions_all matrix uses native training labels "
            "for sampled Ward institutions and inductive nearest-centroid predictions "
            "for OOB institutions. New counterfactual candidates are out-of-sample, "
            "so Stage 6 must use the audited inductive rule. Exact OOB reproduction "
            "therefore validates the Stage-6 prediction engine."
        ),
        "gate_status": (
            "PASS_STAGE_6B_RECONSTRUCTION_AUDIT_V2"
            if all(checks.values())
            else "FAIL_STAGE_6B_RECONSTRUCTION_AUDIT_V2"
        ),
    }

    (OUTPUT_DIR / "stage6b_v2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (AUDIT_DIR / "stage6b_v2_reconstruction_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== ELIGIBLE FAMILY SEMANTICS SUMMARY ===\n")
    print(family_summary.to_string(index=False))

    print("\n=== INTERPRETATION OF THE PREVIOUS 449 MISMATCHES ===\n")
    print(
        "Total sample native-vs-inductive disagreements across 1000 members:",
        int(audit["sample_extension_disagreements"].sum()),
    )
    print(
        "Eligible-member sample native-vs-inductive disagreements:",
        int(eligible_audit["sample_extension_disagreements"].sum()),
    )
    print(
        "These are expected Ward training-label vs nearest-centroid-extension "
        "differences, not failures of the out-of-sample prediction engine."
    )

    print("\n=== STAGE 6B v2 CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nHybrid ALL mismatch cells:", int(audit["hybrid_all_mismatch_count"].sum()))
    print(
        "OOB inductive mismatch cells:",
        int(audit["oob_inductive_mismatch_count"].sum()),
    )
    print("Elapsed seconds:", f"{elapsed:.2f}")

    print(f"\nGATE STATUS: {report['gate_status']}")

    if report["gate_status"] == "PASS_STAGE_6B_RECONSTRUCTION_AUDIT_V2":
        print(
            "The out-of-sample prediction engine for all 984 eligible members is "
            "exactly reproduced. Stage 6C may begin after review."
        )
    else:
        print(
            "Do not run Stage 6C. Review the remaining semantic/reconstruction mismatch."
        )


if __name__ == "__main__":
    main()
