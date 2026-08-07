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

# -----------------------------
# Frozen design / audit inputs
# -----------------------------
STAGE6_FREEZE_PATH = (
    ROOT / "cipher" / "design" / "stage6_ensemble_robustness_freeze.json"
)
STAGE6B_V2_REPORT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "ensemble"
    / "stage6_reconstruction_audit_v2"
    / "stage6b_v2_report.json"
)
CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
FRICTA_CONFIG_PATH = ROOT / "icdm" / "design" / "experiment_config.json"

# -----------------------------
# Data / ensemble artifacts
# -----------------------------
PRIMARY_MATRIX_PATH = ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
FINAL_LABELS_PATH = (
    ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
CF_MANIFEST_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "inductive_validation"
    / "counterfactual_ensemble_manifest.csv"
)
WARD_FIDELITY_PATH = (
    ROOT / "cipher" / "outputs" / "inductive_validation" / "ward_extension_fidelity.csv"
)

# -----------------------------
# Stage 5C candidates
# -----------------------------
COUNTERFACTUALS_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "official_exact"
    / "official_counterfactuals.csv"
)
STAGE5C_DIAGNOSTICS_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "official_exact"
    / "institution_counterfactual_diagnostics.csv"
)

# -----------------------------
# Outputs
# -----------------------------
OUTPUT_DIR = ROOT / "cipher" / "outputs" / "counterfactuals" / "ensemble_robustness"
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


def parse_json_dict(value: str) -> dict[str, Any]:
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object.")
    return obj


def parse_label_mapping(value: str) -> dict[int, int]:
    obj = parse_json_dict(value)
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
    return np.array(
        [mapping[int(label)] for label in labels],
        dtype=int,
    )


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

    features = [str(item) for item in parse_json_list(str(row["feature_names_json"]))]
    sampled_ids = [
        str(item) for item in parse_json_list(str(row["sampled_institution_ids_json"]))
    ]

    X_sample = data_by_id.loc[
        sampled_ids,
        features,
    ].to_numpy(dtype=float)

    scaler = StandardScaler()
    Z_sample_scaled = scaler.fit_transform(X_sample)

    is_pca = "PCA" in representation

    if is_pca:
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

        def predict(X_new_full: pd.DataFrame) -> np.ndarray:
            X_new = X_new_full[features].to_numpy(dtype=float)
            Z_new = scaler.transform(X_new)

            if pca is not None:
                Z_new = pca.transform(Z_new)

            raw = model.predict(Z_new)
            return apply_mapping(raw, mapping)

        return {
            "member_id": member_id,
            "family": str(row["family"]),
            "algorithm": algorithm,
            "representation": representation,
            "predict": predict,
        }

    if "WARD" in algorithm:
        model = AgglomerativeClustering(
            n_clusters=2,
            linkage="ward",
        )
        raw_sample = model.fit_predict(Z_sample)

        raw_values = sorted(np.unique(raw_sample).tolist())
        raw_centroids = {
            int(raw_label): Z_sample[raw_sample == raw_label].mean(axis=0)
            for raw_label in raw_values
        }
        centroid_matrix = np.vstack([raw_centroids[int(value)] for value in raw_values])

        def predict(X_new_full: pd.DataFrame) -> np.ndarray:
            X_new = X_new_full[features].to_numpy(dtype=float)
            Z_new = scaler.transform(X_new)

            if pca is not None:
                Z_new = pca.transform(Z_new)

            distances = np.sqrt(
                ((Z_new[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
            )

            nearest = np.argmin(
                distances,
                axis=1,
            )

            raw = np.array(
                [raw_values[index] for index in nearest],
                dtype=int,
            )

            return apply_mapping(raw, mapping)

        return {
            "member_id": member_id,
            "family": str(row["family"]),
            "algorithm": algorithm,
            "representation": representation,
            "predict": predict,
        }

    raise ValueError(f"{member_id}: unsupported algorithm {algorithm}")


def build_candidate_matrix(
    counterfactuals: pd.DataFrame,
    primary: pd.DataFrame,
    id_column: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_by_id = primary.set_index(id_column)

    candidate_rows = []
    metadata_rows = []

    for candidate_index, row in counterfactuals.reset_index(drop=True).iterrows():
        institution_id = str(row["institution_id"])

        base = (
            primary_by_id.loc[
                institution_id,
                features,
            ]
            .astype(float)
            .copy()
        )

        changes = parse_json_list(str(row["changes_json"]))

        for change in changes:
            feature = str(change["feature"])

            if feature not in features:
                raise ValueError(
                    f"{institution_id}: unknown counterfactual feature {feature}"
                )

            stored_from = float(change["from"])
            original_value = float(base[feature])

            if abs(stored_from - original_value) > 1e-9:
                raise ValueError(
                    f"{institution_id}: change 'from' value for {feature} "
                    f"does not match frozen primary matrix."
                )

            base[feature] = float(change["to"])

        changed_features = [
            str(item) for item in parse_json_list(str(row["changed_features_json"]))
        ]

        if len(changed_features) != int(row["l0"]):
            raise ValueError(f"{institution_id}: changed_features_json count != l0.")

        if len(changed_features) > 4:
            raise ValueError(
                f"{institution_id}: Stage 5C candidate exceeds four changes."
            )

        candidate_id = f"{institution_id}__CF{int(row['rank']):02d}"

        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                **{feature: float(base[feature]) for feature in features},
            }
        )

        metadata_rows.append(
            {
                "candidate_id": candidate_id,
                "institution_id": institution_id,
                "reference_profile": int(row["reference_profile"]),
                "target_profile": int(row["target_profile"]),
                "certainty_class": str(row["certainty_class"]),
                "rank": int(row["rank"]),
                "is_exact_global_minimum": bool(row["is_exact_global_minimum"]),
                "weighted_l1": float(row["weighted_l1"]),
                "l0": int(row["l0"]),
                "total_cost": float(row["total_cost"]),
                "plausibility_distance": float(row["plausibility_distance"]),
                "target_margin": float(row["target_margin"]),
                "changed_features_json": str(row["changed_features_json"]),
                "signed_items_json": str(row["signed_items_json"]),
                "changes_json": str(row["changes_json"]),
            }
        )

    candidate_matrix = pd.DataFrame(candidate_rows).set_index("candidate_id")

    metadata = pd.DataFrame(metadata_rows)

    return candidate_matrix, metadata


def summarize_group(
    institution_summary: pd.DataFrame,
    group_type: str,
    group_column: str | None,
    taus: list[float],
) -> pd.DataFrame:
    rows = []

    if group_column is None:
        groups = [("ALL", institution_summary)]
    else:
        groups = institution_summary.groupby(
            group_column,
            sort=True,
        )

    for value, group in groups:
        row = {
            "group_type": group_type,
            "group_value": str(value),
            "n_reachable_stage5c": len(group),
        }

        for tau in taus:
            column = f"robustly_reachable_tau_{tau:.2f}"
            robust_n = int(group[column].astype(bool).sum())
            row[f"robust_n_tau_{tau:.2f}"] = robust_n
            row[f"robust_rate_tau_{tau:.2f}"] = robust_n / len(group)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage6_freeze = load_json(STAGE6_FREEZE_PATH)
    stage6b_report = load_json(STAGE6B_V2_REPORT_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    fricta_config = load_json(FRICTA_CONFIG_PATH)

    if stage6_freeze.get("gate_status") != "PASS_STAGE_6A_DESIGN_FREEZE":
        raise ValueError("Stage 6A design freeze has not passed.")

    if stage6b_report.get("gate_status") != "PASS_STAGE_6B_RECONSTRUCTION_AUDIT_V2":
        raise ValueError("Stage 6B v2 reconstruction audit has not passed.")

    id_column = fricta_config["id_column"]
    features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[
        [
            id_column,
            "cluster_id",
        ]
    ]
    manifest = pd.read_csv(CF_MANIFEST_PATH)
    ward_fidelity = pd.read_csv(WARD_FIDELITY_PATH)
    counterfactuals = pd.read_csv(COUNTERFACTUALS_PATH)
    stage5c_diagnostics = pd.read_csv(STAGE5C_DIAGNOSTICS_PATH)

    primary[id_column] = primary[id_column].astype(str)
    labels[id_column] = labels[id_column].astype(str)
    manifest["member_id"] = manifest["member_id"].astype(str)
    counterfactuals["institution_id"] = counterfactuals["institution_id"].astype(str)
    stage5c_diagnostics["institution_id"] = stage5c_diagnostics[
        "institution_id"
    ].astype(str)

    data = primary.merge(
        labels,
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 institutions; found {len(data)}.")

    data_by_id = data.set_index(id_column)

    eligibility = parse_bool_series(manifest["eligible_for_counterfactual_ensemble"])
    eligible_manifest = manifest[eligibility].copy()

    family_counts = eligible_manifest["family"].value_counts().to_dict()

    primary_tau = float(stage6_freeze["candidate_evaluation"]["primary_tau"])
    sensitivity_taus = [
        float(value)
        for value in stage6_freeze["candidate_evaluation"]["sensitivity_taus"]
    ]
    all_taus = sorted(set([primary_tau] + sensitivity_taus))

    reachable_stage5c = stage5c_diagnostics[
        stage5c_diagnostics["failure_mode"] == "SOLVABLE"
    ].copy()

    frozen_scope_ids = sorted(
        [str(value) for value in stage6_freeze["scope"]["institution_ids"]]
    )

    candidate_ids = sorted(counterfactuals["institution_id"].unique().tolist())

    prechecks = {
        "stage6a_freeze_passed": (
            stage6_freeze.get("gate_status") == "PASS_STAGE_6A_DESIGN_FREEZE"
        ),
        "stage6b_v2_passed": (
            stage6b_report.get("gate_status") == "PASS_STAGE_6B_RECONSTRUCTION_AUDIT_V2"
        ),
        "exactly_984_eligible_members": (len(eligible_manifest) == 984),
        "eligible_family_counts_match_frozen_stage4": (
            all(
                int(
                    family_counts.get(
                        family,
                        0,
                    )
                )
                == expected
                for family, expected in EXPECTED_ELIGIBLE_FAMILY_COUNTS.items()
            )
        ),
        "exactly_69_saved_stage5c_candidates": (len(counterfactuals) == 69),
        "exactly_19_stage5c_reachable_institutions": (len(reachable_stage5c) == 19),
        "candidate_institution_ids_match_frozen_stage6_scope": (
            candidate_ids == frozen_scope_ids
        ),
        "all_candidates_within_four_changes": bool((counterfactuals["l0"] <= 4).all()),
        "one_rank1_candidate_per_reachable_institution": (
            len(counterfactuals[counterfactuals["rank"] == 1]) == 19
        ),
    }

    print("\n=== CIPHER STAGE 6C — OFFICIAL ENSEMBLE ROBUSTNESS EVALUATION ===\n")
    print("Pre-evaluation checks:")
    for name, passed in prechecks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(prechecks.values()):
        print("\nGATE STATUS: FAIL_STAGE_6C_PRECHECK")
        raise SystemExit(1)

    candidate_matrix, metadata = build_candidate_matrix(
        counterfactuals=counterfactuals,
        primary=primary,
        id_column=id_column,
        features=features,
    )

    candidate_order = metadata["candidate_id"].tolist()

    candidate_matrix = candidate_matrix.loc[candidate_order]

    target_profiles = metadata["target_profile"].to_numpy(dtype=int)

    ensemble_cfg = cipher_config.get(
        "ensemble",
        {},
    )
    kmeans_n_init = int(
        ensemble_cfg.get(
            "kmeans_n_init",
            25,
        )
    )
    pca_threshold = float(
        ensemble_cfg.get(
            "pca_variance_threshold",
            0.85,
        )
    )

    # candidate x member binary support matrix
    support_matrix = np.zeros(
        (
            len(metadata),
            len(eligible_manifest),
        ),
        dtype=np.uint8,
    )

    member_ids = []
    member_families = []

    started = time.perf_counter()

    for member_position, (_, row) in enumerate(eligible_manifest.iterrows()):
        fitted = fit_member_predictor(
            row=row,
            data_by_id=data_by_id,
            kmeans_n_init=kmeans_n_init,
            pca_threshold=pca_threshold,
        )

        predictions = fitted["predict"](candidate_matrix)

        support = (predictions == target_profiles).astype(np.uint8)

        support_matrix[
            :,
            member_position,
        ] = support

        member_ids.append(fitted["member_id"])
        member_families.append(fitted["family"])

        if (member_position + 1) % 100 == 0:
            print(
                f"Evaluated {member_position + 1:03d}/984 ensemble members...",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    support_df = pd.DataFrame(
        support_matrix,
        index=candidate_order,
        columns=member_ids,
    )
    support_df.index.name = "candidate_id"
    support_df.to_csv(OUTPUT_DIR / "candidate_member_target_support_matrix.csv")

    candidate_summary = metadata.copy()

    candidate_summary["ensemble_support"] = support_matrix.mean(axis=1)

    for family in EXPECTED_ELIGIBLE_FAMILY_COUNTS:
        positions = [
            index
            for index, family_name in enumerate(member_families)
            if family_name == family
        ]

        if len(positions) != EXPECTED_ELIGIBLE_FAMILY_COUNTS[family]:
            raise ValueError(
                f"{family}: expected "
                f"{EXPECTED_ELIGIBLE_FAMILY_COUNTS[family]} eligible members, "
                f"found {len(positions)}."
            )

        candidate_summary[f"support_{family}"] = support_matrix[
            :,
            positions,
        ].mean(axis=1)

    family_support_columns = [
        f"support_{family}" for family in EXPECTED_ELIGIBLE_FAMILY_COUNTS
    ]

    candidate_summary["minimum_family_support"] = candidate_summary[
        family_support_columns
    ].min(axis=1)

    candidate_summary["maximum_family_support"] = candidate_summary[
        family_support_columns
    ].max(axis=1)

    candidate_summary["family_support_range"] = (
        candidate_summary["maximum_family_support"]
        - candidate_summary["minimum_family_support"]
    )

    for tau in all_taus:
        candidate_summary[f"valid_tau_{tau:.2f}"] = (
            candidate_summary["ensemble_support"] >= tau
        )

    candidate_summary.to_csv(
        OUTPUT_DIR / "candidate_ensemble_support.csv",
        index=False,
    )

    # -----------------------------
    # Institution-level aggregation
    # -----------------------------
    institution_rows = []

    for institution_id, group in candidate_summary.groupby(
        "institution_id",
        sort=True,
    ):
        group = group.copy()

        base_row = group.iloc[0]

        institution_result = {
            "institution_id": institution_id,
            "reference_profile": int(base_row["reference_profile"]),
            "target_profile": int(base_row["target_profile"]),
            "certainty_class": str(base_row["certainty_class"]),
            "saved_candidates": len(group),
            "maximum_ensemble_support": float(group["ensemble_support"].max()),
            "minimum_ensemble_support": float(group["ensemble_support"].min()),
            "median_ensemble_support": float(group["ensemble_support"].median()),
        }

        best_available = group.sort_values(
            [
                "ensemble_support",
                "total_cost",
                "plausibility_distance",
                "rank",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        ).iloc[0]

        institution_result["best_available_candidate_id"] = str(
            best_available["candidate_id"]
        )
        institution_result["best_available_support"] = float(
            best_available["ensemble_support"]
        )
        institution_result["best_available_cost"] = float(best_available["total_cost"])

        for tau in all_taus:
            valid = group[group[f"valid_tau_{tau:.2f}"].astype(bool)].copy()

            institution_result[f"robustly_reachable_tau_{tau:.2f}"] = len(valid) > 0
            institution_result[f"valid_candidate_count_tau_{tau:.2f}"] = len(valid)

            if len(valid):
                best = valid.sort_values(
                    [
                        "total_cost",
                        "ensemble_support",
                        "plausibility_distance",
                        "rank",
                    ],
                    ascending=[
                        True,
                        False,
                        True,
                        True,
                    ],
                ).iloc[0]

                institution_result[f"best_candidate_id_tau_{tau:.2f}"] = str(
                    best["candidate_id"]
                )
                institution_result[f"best_candidate_cost_tau_{tau:.2f}"] = float(
                    best["total_cost"]
                )
                institution_result[f"best_candidate_support_tau_{tau:.2f}"] = float(
                    best["ensemble_support"]
                )
                institution_result[
                    f"best_candidate_min_family_support_tau_{tau:.2f}"
                ] = float(best["minimum_family_support"])
            else:
                institution_result[f"best_candidate_id_tau_{tau:.2f}"] = ""
                institution_result[f"best_candidate_cost_tau_{tau:.2f}"] = np.nan
                institution_result[f"best_candidate_support_tau_{tau:.2f}"] = np.nan
                institution_result[
                    f"best_candidate_min_family_support_tau_{tau:.2f}"
                ] = np.nan

        institution_rows.append(institution_result)

    institution_summary = pd.DataFrame(institution_rows)

    institution_summary.to_csv(
        OUTPUT_DIR / "institution_robust_reachability.csv",
        index=False,
    )

    # -----------------------------
    # Group summaries
    # -----------------------------
    summary_frames = [
        summarize_group(
            institution_summary,
            "OVERALL",
            None,
            all_taus,
        ),
        summarize_group(
            institution_summary,
            "REFERENCE_PROFILE",
            "reference_profile",
            all_taus,
        ),
        summarize_group(
            institution_summary,
            "CERTAINTY_CLASS",
            "certainty_class",
            all_taus,
        ),
    ]

    group_summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    group_summary.to_csv(
        OUTPUT_DIR / "robust_reachability_summary.csv",
        index=False,
    )

    # -----------------------------
    # Candidate-level tau summary
    # -----------------------------
    candidate_tau_rows = []

    for tau in all_taus:
        candidate_tau_rows.append(
            {
                "tau": tau,
                "valid_candidate_rows": int(
                    candidate_summary[f"valid_tau_{tau:.2f}"].astype(bool).sum()
                ),
                "total_candidate_rows": len(candidate_summary),
                "candidate_valid_rate": float(
                    candidate_summary[f"valid_tau_{tau:.2f}"].astype(bool).mean()
                ),
                "robustly_reachable_institutions": int(
                    institution_summary[f"robustly_reachable_tau_{tau:.2f}"]
                    .astype(bool)
                    .sum()
                ),
                "reachable_stage5c_institutions": len(institution_summary),
                "conditional_robust_reachability_rate": float(
                    institution_summary[f"robustly_reachable_tau_{tau:.2f}"]
                    .astype(bool)
                    .mean()
                ),
                "unconditional_81_institution_rate": float(
                    institution_summary[f"robustly_reachable_tau_{tau:.2f}"]
                    .astype(bool)
                    .sum()
                    / 81
                ),
            }
        )

    tau_summary = pd.DataFrame(candidate_tau_rows)
    tau_summary.to_csv(
        OUTPUT_DIR / "tau_sensitivity_summary.csv",
        index=False,
    )

    # -----------------------------
    # Technical integrity checks only
    # -----------------------------
    checks = {
        "all_984_eligible_members_evaluated": (support_matrix.shape[1] == 984),
        "all_69_candidates_evaluated": (support_matrix.shape[0] == 69),
        "all_support_values_binary_at_member_level": bool(
            np.isin(
                support_matrix,
                [0, 1],
            ).all()
        ),
        "all_candidate_supports_in_unit_interval": bool(
            (
                (candidate_summary["ensemble_support"] >= 0)
                & (candidate_summary["ensemble_support"] <= 1)
            ).all()
        ),
        "all_four_family_supports_present": (
            all(
                column in candidate_summary.columns for column in family_support_columns
            )
        ),
        "all_19_institutions_summarized": (len(institution_summary) == 19),
        "no_new_candidates_generated": (
            len(candidate_summary) == len(counterfactuals) == 69
        ),
        "primary_tau_is_frozen_090": (abs(primary_tau - 0.90) <= 1e-12),
        "sensitivity_taus_are_frozen_080_095": (
            sorted(sensitivity_taus)
            == [
                0.80,
                0.95,
            ]
        ),
    }

    primary_robust_n = int(
        institution_summary[f"robustly_reachable_tau_{primary_tau:.2f}"]
        .astype(bool)
        .sum()
    )
    primary_conditional_rate = primary_robust_n / 19
    primary_unconditional_rate = primary_robust_n / 81

    report = {
        "eligible_ensemble_members": 984,
        "candidate_rows": 69,
        "stage5c_reachable_institutions": 19,
        "primary_tau": primary_tau,
        "sensitivity_taus": sensitivity_taus,
        "primary_robustly_reachable_institutions": (primary_robust_n),
        "primary_conditional_robust_reachability_rate": (primary_conditional_rate),
        "primary_unconditional_81_institution_rate": (primary_unconditional_rate),
        "candidate_ensemble_support_quantiles": {
            "q025": float(candidate_summary["ensemble_support"].quantile(0.025)),
            "q25": float(candidate_summary["ensemble_support"].quantile(0.25)),
            "median": float(candidate_summary["ensemble_support"].median()),
            "q75": float(candidate_summary["ensemble_support"].quantile(0.75)),
            "q975": float(candidate_summary["ensemble_support"].quantile(0.975)),
        },
        "elapsed_seconds": float(elapsed),
        "checks": checks,
        "scientific_result_gate": ("NONE_PREDECLARED_AFTER_STAGE5E_REVISION"),
        "interpretation": (
            "This stage reports ensemble robustness conditional on the 19 "
            "institutions that were exactly reachable under Stage 5C. "
            "No minimum robust-reachability percentage is imposed post hoc. "
            "Counterfactuals remain diagnostic profile-transition explanations, "
            "not causal interventions."
        ),
        "gate_status": (
            "PASS_STAGE_6C_OFFICIAL_EVALUATION"
            if all(checks.values())
            else "FAIL_STAGE_6C_TECHNICAL_INTEGRITY"
        ),
    }

    (OUTPUT_DIR / "stage6c_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (AUDIT_DIR / "stage6c_official_evaluation_audit.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== STAGE 6C — TAU SENSITIVITY SUMMARY ===\n")
    print(tau_summary.to_string(index=False))

    print("\n=== STAGE 6C — ROBUST REACHABILITY BY GROUP ===\n")
    print(group_summary.to_string(index=False))

    print("\n=== PRIMARY TAU=0.90 — INSTITUTION RESULTS ===\n")

    primary_columns = [
        "institution_id",
        "reference_profile",
        "certainty_class",
        "saved_candidates",
        "maximum_ensemble_support",
        f"robustly_reachable_tau_{primary_tau:.2f}",
        f"valid_candidate_count_tau_{primary_tau:.2f}",
        f"best_candidate_id_tau_{primary_tau:.2f}",
        f"best_candidate_cost_tau_{primary_tau:.2f}",
        f"best_candidate_support_tau_{primary_tau:.2f}",
        f"best_candidate_min_family_support_tau_{primary_tau:.2f}",
    ]

    print(
        institution_summary[primary_columns]
        .sort_values(
            [
                f"robustly_reachable_tau_{primary_tau:.2f}",
                "maximum_ensemble_support",
                "institution_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .to_string(index=False)
    )

    print("\n=== TOP 15 CANDIDATES BY ENSEMBLE SUPPORT ===\n")
    print(
        candidate_summary.sort_values(
            [
                "ensemble_support",
                "total_cost",
                "candidate_id",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .head(15)[
            [
                "candidate_id",
                "institution_id",
                "reference_profile",
                "certainty_class",
                "rank",
                "total_cost",
                "ensemble_support",
                "minimum_family_support",
                "family_support_range",
            ]
            + family_support_columns
        ]
        .to_string(index=False)
    )

    print("\n=== TECHNICAL INTEGRITY CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(
        "\nPrimary tau=0.90 robustly reachable:",
        f"{primary_robust_n}/19",
        f"({primary_conditional_rate:.4f} conditional)",
    )
    print(
        "Equivalent rate over full N=81:",
        f"{primary_unconditional_rate:.4f}",
    )
    print(
        "Elapsed seconds:",
        f"{elapsed:.2f}",
    )

    print(f"\nGATE STATUS: {report['gate_status']}")

    if report["gate_status"] == "PASS_STAGE_6C_OFFICIAL_EVALUATION":
        print(
            "Stage 6C is technically complete. The observed robustness rate is "
            "a scientific result, not a pass/fail target. Review before Stage 7."
        )
    else:
        print(
            "Do not proceed to Stage 7 until the technical integrity failure is reviewed."
        )


if __name__ == "__main__":
    main()
