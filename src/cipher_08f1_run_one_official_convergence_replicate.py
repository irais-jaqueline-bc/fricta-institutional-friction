from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)

import cipher_08d1_core_pipeline_smoke_v2 as core
import cipher_08d2_counterfactual_smoke_v3 as cf
from cipher_synthetic_generators import (
    FEATURE_NAMES,
    generate_scenario,
)

ROOT = Path(__file__).resolve().parents[1]

PLAN_PATH = ROOT / "cipher" / "design" / "stage8_official_run_plan_freeze_v1.json"

OFFICIAL_ROOT = (
    ROOT / "cipher" / "outputs" / "synthetic" / "official" / "convergence_audit"
)

AUDIT_ROOT = ROOT / "cipher" / "outputs" / "audit" / "stage8f"

MASTER_SEED = 20260807
STABILITY_ITERATIONS = 1000

SCENARIOS = [
    "S1_CONFIG_TWO_PROFILE",
    "S2_CORE_BOUNDARY",
    "S3_DIRECTIONAL_REACHABILITY",
    "S4_SEVERITY_CONTINUUM",
    "S5_GOVERNANCE_CONFOUNDED",
    "S6_NO_CLUSTER_NULL",
]

CF_SCENARIOS = {
    "S1_CONFIG_TWO_PROFILE",
    "S2_CORE_BOUNDARY",
    "S3_DIRECTIONAL_REACHABILITY",
    "S6_NO_CLUSTER_NULL",
}

OFFICIAL_AUDIT_REPLICATES = set(
    range(
        2,
        12,
    )
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen official CIPHER Stage-8 convergence-audit "
            "replicate across all six synthetic scenarios."
        )
    )

    parser.add_argument(
        "--replicate",
        type=int,
        required=True,
        help="Official convergence-audit replicate index, 2..11.",
    )

    return parser.parse_args()


def official_seed_scenario(
    scenario: str,
    replicate: int,
) -> str:
    # Existing core helper functions use the scenario string only for
    # deterministic random-state construction. Add the official replicate
    # index to that seed namespace without changing the synthetic scenario ID.
    return f"{scenario}__OFFICIAL_R{replicate:03d}"


def build_uncertainty_ensemble(
    X: np.ndarray,
    reference_profiles: np.ndarray,
    scenario: str,
    replicate: int,
    members_per_family: int,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    n = len(X)

    sample_size = int(round(n * 0.80))

    family_specs = [
        (
            "R0_WARD",
            "R0_STANDARDIZED",
            "HAC_WARD",
        ),
        (
            "R1_PCA85_WARD",
            "R1_PCA85",
            "HAC_WARD",
        ),
        (
            "R0_KMEANS",
            "R0_STANDARDIZED",
            "KMEANS",
        ),
        (
            "R1_PCA85_KMEANS",
            "R1_PCA85",
            "KMEANS",
        ),
    ]

    oob_predictions: list[list[int]] = [[] for _ in range(n)]

    family_oob_predictions: dict[
        str,
        list[list[int]],
    ] = {
        family: [[] for _ in range(n)]
        for (
            family,
            _,
            _,
        ) in family_specs
    }

    total_members = 0

    for (
        family,
        representation,
        algorithm,
    ) in family_specs:
        rng = np.random.default_rng(
            core.stable_seed(
                MASTER_SEED,
                "official_uncertainty_family",
                scenario,
                replicate,
                family,
                members_per_family,
            )
        )

        for member_index in range(members_per_family):
            sample = np.sort(
                rng.choice(
                    np.arange(
                        n,
                        dtype=int,
                    ),
                    size=sample_size,
                    replace=False,
                )
            )

            oob = np.setdiff1d(
                np.arange(
                    n,
                    dtype=int,
                ),
                sample,
                assume_unique=True,
            )

            feature_idx = np.sort(
                rng.choice(
                    np.arange(
                        X.shape[1],
                        dtype=int,
                    ),
                    size=11,
                    replace=False,
                )
            )

            X_sample = X[sample][:, feature_idx]

            X_oob = X[oob][:, feature_idx]

            Z_sample, scaler, pca = core.transform_representation(
                X_sample,
                representation,
            )

            model, native_labels = core.fit_cluster(
                Z_sample,
                algorithm,
                2,
                core.stable_seed(
                    MASTER_SEED,
                    "official_uncertainty_fit",
                    scenario,
                    replicate,
                    family,
                    members_per_family,
                    member_index,
                ),
            )

            mapping = core.align_member_labels(
                native_labels,
                reference_profiles[sample],
            )

            Z_oob = core.apply_representation(
                X_oob,
                scaler,
                pca,
            )

            if algorithm == "KMEANS":
                native_oob = model.predict(Z_oob)

            else:
                native_oob = core.ward_centroid_predict(
                    Z_sample,
                    native_labels,
                    Z_oob,
                )

            aligned_oob = np.array(
                [mapping[int(value)] for value in native_oob],
                dtype=int,
            )

            for row_index, prediction in zip(
                oob,
                aligned_oob,
            ):
                oob_predictions[int(row_index)].append(int(prediction))

                family_oob_predictions[family][int(row_index)].append(int(prediction))

            total_members += 1

    rows = []

    for row_index in range(n):
        predictions = np.array(
            oob_predictions[row_index],
            dtype=int,
        )

        if len(predictions) == 0:
            raise RuntimeError(f"No OOB uncertainty predictions for row {row_index}.")

        p1 = float(np.mean(predictions == 1))

        p2 = float(np.mean(predictions == 2))

        reference = int(reference_profiles[row_index])

        p_ref = p1 if reference == 1 else p2

        nonzero = np.array(
            [
                p1,
                p2,
            ],
            dtype=float,
        )

        nonzero = nonzero[nonzero > 0]

        entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(2.0))

        margin = float(2.0 * abs(p1 - 0.5))

        family_reference_probabilities = []

        for (
            family,
            _,
            _,
        ) in family_specs:
            family_predictions = np.array(
                family_oob_predictions[family][row_index],
                dtype=int,
            )

            if len(family_predictions) == 0:
                raise RuntimeError(
                    f"Missing OOB family predictions for row {row_index}, family {family}."
                )

            family_reference_probabilities.append(
                float(np.mean(family_predictions == reference))
            )

        family_consistency = float(min(family_reference_probabilities))

        if p_ref >= 0.90 and family_consistency >= 0.80:
            certainty_class = "CORE"

        elif p_ref < 0.75 or family_consistency < 0.60:
            certainty_class = "BOUNDARY"

        else:
            certainty_class = "HALO"

        rows.append(
            {
                "row_index": int(row_index),
                "reference_profile": reference,
                "n_oob_predictions": int(len(predictions)),
                "profile_1_probability": p1,
                "profile_2_probability": p2,
                "reference_profile_probability": p_ref,
                "normalized_entropy": entropy,
                "membership_margin": margin,
                "family_consistency": family_consistency,
                "certainty_class": certainty_class,
            }
        )

    certainty = pd.DataFrame(rows)

    report = {
        "ensemble_members": int(total_members),
        "members_per_family": int(members_per_family),
        "minimum_oob_predictions": int(certainty["n_oob_predictions"].min()),
        "median_oob_predictions": float(certainty["n_oob_predictions"].median()),
        "certainty_class_counts": {
            str(key): int(value)
            for (
                key,
                value,
            ) in certainty["certainty_class"]
            .value_counts()
            .items()
        },
    }

    return (
        certainty,
        report,
    )


def compare_uncertainty(
    certainty_200: pd.DataFrame,
    certainty_1000: pd.DataFrame,
) -> dict[str, Any]:
    left = certainty_200.sort_values("row_index")

    right = certainty_1000.sort_values("row_index")

    if not np.array_equal(
        left["row_index"].to_numpy(),
        right["row_index"].to_numpy(),
    ):
        raise RuntimeError("200/1000 uncertainty rows do not align.")

    rho = float(
        spearmanr(
            left["normalized_entropy"].to_numpy(dtype=float),
            right["normalized_entropy"].to_numpy(dtype=float),
        ).statistic
    )

    return {
        "normalized_entropy_spearman_200_vs_1000": rho,
        "passes_0_95": bool(rho >= 0.95),
    }


def build_robust_ensemble_sized(
    X: np.ndarray,
    reference_profiles: np.ndarray,
    scenario: str,
    replicate: int,
    members_per_family: int,
    minimum_eligible: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    old_rep = cf.SMOKE_REPLICATE
    old_members = cf.ENSEMBLE_MEMBERS_PER_FAMILY
    old_minimum = cf.MIN_ELIGIBLE_MEMBERS

    try:
        cf.SMOKE_REPLICATE = replicate

        cf.ENSEMBLE_MEMBERS_PER_FAMILY = members_per_family

        cf.MIN_ELIGIBLE_MEMBERS = minimum_eligible

        ensemble, report = cf.build_robust_ensemble(
            X,
            reference_profiles,
            scenario,
        )

    finally:
        cf.SMOKE_REPLICATE = old_rep
        cf.ENSEMBLE_MEMBERS_PER_FAMILY = old_members
        cf.MIN_ELIGIBLE_MEMBERS = old_minimum

    report["frozen_minimum_required"] = int(minimum_eligible)

    report["evaluable"] = bool(report["eligible_members"] >= minimum_eligible)

    return (
        ensemble,
        report,
    )


def robust_query_decisions(
    single_model_results: list[dict[str, Any]],
    ensemble: list[dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    robust_results = [
        cf.evaluate_robust_support(
            result,
            ensemble,
        )
        for result in single_model_results
    ]

    frame = cf.institution_summary_frame(robust_results)

    return (
        frame,
        robust_results,
    )


def compare_robust_decisions(
    frame_200: pd.DataFrame,
    frame_1000: pd.DataFrame,
) -> dict[str, Any]:
    left = frame_200[
        [
            "institution_id",
            "robust_reachable_tau_090",
        ]
    ].copy()

    right = frame_1000[
        [
            "institution_id",
            "robust_reachable_tau_090",
        ]
    ].copy()

    merged = left.merge(
        right,
        on="institution_id",
        suffixes=(
            "_200",
            "_1000",
        ),
        validate="one_to_one",
    )

    agreement = float(
        np.mean(
            merged["robust_reachable_tau_090_200"].astype(bool).to_numpy()
            == merged["robust_reachable_tau_090_1000"].astype(bool).to_numpy()
        )
    )

    return {
        "query_count": int(len(merged)),
        "tau_090_decision_agreement_200_vs_1000": agreement,
        "passes_0_95": bool(agreement >= 0.95),
        "robust_positive_200": int(
            merged["robust_reachable_tau_090_200"].astype(bool).sum()
        ),
        "robust_positive_1000": int(
            merged["robust_reachable_tau_090_1000"].astype(bool).sum()
        ),
    }


def save_single_model_candidates(
    scenario_dir: Path,
    single_results: list[dict[str, Any]],
) -> None:
    rows = []

    for result in single_results:
        for candidate in result["saved_candidates"]:
            row = {
                "institution_id": result["institution_id"],
                "source_profile": result["source_profile"],
                "target_profile": result["target_profile"],
                "candidate_id": candidate["candidate_id"],
                "rank": candidate["rank"],
                "l0": candidate["l0"],
                "weighted_l1": candidate["weighted_l1"],
                "total_cost": candidate["total_cost"],
                "plausibility_distance": candidate["plausibility_distance"],
                "plausibility_threshold": candidate["plausibility_threshold"],
                "edit_signature_json": json.dumps(candidate["edit_signature"]),
            }

            for j, feature in enumerate(FEATURE_NAMES):
                row[f"candidate__{feature}"] = candidate["candidate_vector"][j]

            rows.append(row)

    if rows:
        frame = pd.DataFrame(rows)
    else:
        frame = pd.DataFrame(
            columns=[
                "institution_id",
                "source_profile",
                "target_profile",
                "candidate_id",
                "rank",
                "l0",
                "weighted_l1",
                "total_cost",
                "plausibility_distance",
                "plausibility_threshold",
                "edit_signature_json",
            ]
            + [f"candidate__{feature}" for feature in FEATURE_NAMES]
        )

    frame.to_csv(
        scenario_dir / "single_model_saved_candidates_pretruth.csv",
        index=False,
    )


def run_one_scenario(
    scenario: str,
    replicate: int,
    replicate_root: Path,
) -> dict[str, Any]:
    scenario_seed_key = official_seed_scenario(
        scenario,
        replicate,
    )

    scenario_dir = replicate_root / scenario

    scenario_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    bundle = generate_scenario(
        scenario_id=scenario,
        replicate=replicate,
        master_seed=MASTER_SEED,
    )

    data_pretruth = bundle.data.copy()

    X = data_pretruth[FEATURE_NAMES].to_numpy(dtype=float)

    # -------------------------------------------------------------
    # Discovery and 1000-resample official stability.
    # -------------------------------------------------------------
    old_stability = core.STABILITY_ITERATIONS

    try:
        core.STABILITY_ITERATIONS = STABILITY_ITERATIONS

        metrics, labels_by_candidate = core.full_candidate_search(
            X,
            scenario_seed_key,
        )

        shortlist = core.build_shortlist(metrics)

        stability = core.run_stability(
            X,
            scenario_seed_key,
            shortlist,
            labels_by_candidate,
        )

    finally:
        core.STABILITY_ITERATIONS = old_stability

    selected, decision, diagnostics = core.choose_model(
        metrics,
        stability,
        labels_by_candidate,
    )

    selected_id = str(selected["candidate_id"])

    selected_k = int(selected["k_requested"])

    raw_labels = labels_by_candidate[selected_id].astype(int)

    stable_partition_claim = bool(
        selected["selection_eligible"] and float(selected["ari_median"]) >= 0.70
    )

    if selected_k == 2:
        severity_labels = core.remap_selected_labels_to_profiles(raw_labels)

    else:
        severity_labels = raw_labels

    severity = core.severity_audit(
        X,
        severity_labels,
        selected_k,
        scenario_seed_key,
    )

    labels_out = pd.DataFrame(
        {
            "institution_id": data_pretruth["institution_id"].astype(str),
            "cluster_id": raw_labels,
        }
    )

    governance = core.governance_audit(
        bundle.truth,
        labels_out,
        scenario_seed_key,
    )

    configurational_claim = bool(
        stable_partition_claim
        and not severity["severity_nearly_reconstructs_profiles"]
        and not governance["strong_governance_association"]
        and not governance["governance_nearly_reconstructs_profiles"]
    )

    # Save all discovery/pretruth outputs before truth metrics.
    data_pretruth.to_csv(
        scenario_dir / "data_pretruth.csv",
        index=False,
    )

    metrics.to_csv(
        scenario_dir / "candidate_metrics.csv",
        index=False,
    )

    shortlist.to_csv(
        scenario_dir / "stability_shortlist.csv",
        index=False,
    )

    stability.to_csv(
        scenario_dir / "stability_summary_1000.csv",
        index=False,
    )

    diagnostics.to_csv(
        scenario_dir / "selection_diagnostics.csv",
        index=False,
    )

    labels_out.to_csv(
        scenario_dir / "selected_labels.csv",
        index=False,
    )

    pretruth_report: dict[
        str,
        Any,
    ] = {
        "scenario": scenario,
        "replicate": int(replicate),
        "selected_candidate": selected_id,
        "selected_representation": str(selected["representation"]),
        "selected_algorithm": str(selected["algorithm"]),
        "selected_k": selected_k,
        "selected_ari_median": float(selected["ari_median"]),
        "selected_ari_p025": float(selected["ari_p025"]),
        "selected_silhouette": float(selected["silhouette"]),
        "selected_minimum_cluster_size": int(selected["minimum_cluster_size"]),
        "selected_minimum_resample_cluster_size": int(
            selected["minimum_resample_cluster_size_min"]
        ),
        "selection_decision": decision,
        "stable_partition_claim": stable_partition_claim,
        "severity": severity,
        "governance": governance,
        "configurational_profile_claim": configurational_claim,
        "uncertainty_applicable": bool(stable_partition_claim and selected_k == 2),
        "cf_applicable": bool(
            scenario in CF_SCENARIOS and stable_partition_claim and selected_k == 2
        ),
    }

    (scenario_dir / "pipeline_pretruth_report.json").write_text(
        json.dumps(
            pretruth_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # 200 vs 1000 uncertainty convergence, only for stable selected k=2.
    # -------------------------------------------------------------
    uncertainty_convergence = {
        "applicable": False,
        "status": "NOT_APPLICABLE_SELECTED_K_NOT_2_OR_UNSTABLE",
    }

    certainty_200 = None
    certainty_1000 = None

    reference_profiles = None

    if stable_partition_claim and selected_k == 2:
        reference_profiles = core.remap_selected_labels_to_profiles(raw_labels)

        certainty_200, report_200 = build_uncertainty_ensemble(
            X,
            reference_profiles,
            scenario,
            replicate,
            members_per_family=50,
        )

        certainty_1000, report_1000 = build_uncertainty_ensemble(
            X,
            reference_profiles,
            scenario,
            replicate,
            members_per_family=250,
        )

        certainty_200.insert(
            0,
            "institution_id",
            data_pretruth["institution_id"].astype(str).to_numpy(),
        )

        certainty_1000.insert(
            0,
            "institution_id",
            data_pretruth["institution_id"].astype(str).to_numpy(),
        )

        certainty_200.to_csv(
            scenario_dir / "membership_certainty_200_pretruth.csv",
            index=False,
        )

        certainty_1000.to_csv(
            scenario_dir / "membership_certainty_1000_pretruth.csv",
            index=False,
        )

        uncertainty_convergence = {
            "applicable": True,
            "status": "COMPLETED",
            "ensemble_200": report_200,
            "ensemble_1000": report_1000,
            **compare_uncertainty(
                certainty_200,
                certainty_1000,
            ),
        }

        (scenario_dir / "uncertainty_convergence_pretruth.json").write_text(
            json.dumps(
                uncertainty_convergence,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -------------------------------------------------------------
    # Single-model CF + 200/1000 robust convergence when applicable.
    # -------------------------------------------------------------
    cf_convergence: dict[
        str,
        Any,
    ] = {
        "applicable": False,
        "status": "NOT_APPLICABLE_SCENARIO_OR_SELECTED_K",
    }

    cf_single_results = None
    robust_200_frame = None
    robust_1000_frame = None

    if scenario in CF_SCENARIOS and stable_partition_claim and selected_k == 2:
        assert reference_profiles is not None

        predictor = cf.fit_reference_predictor(
            X,
            reference_profiles,
            str(selected["representation"]),
            str(selected["algorithm"]),
            scenario_seed_key,
            selected_id,
        )

        if predictor["partition_ari"] < 0.999999:
            raise RuntimeError(
                f"{scenario} R{replicate}: reference predictor "
                f"does not reproduce selected partition."
            )

        if (
            predictor["algorithm"] == "HAC_WARD"
            and predictor["inductive_fidelity"] < 0.95
        ):
            cf_convergence = {
                "applicable": False,
                "status": ("NOT_APPLICABLE_WARD_REFERENCE_EXTENSION_FIDELITY_FAIL"),
                "reference_inductive_fidelity": float(predictor["inductive_fidelity"]),
            }

        else:
            old_cf_rep = cf.SMOKE_REPLICATE

            try:
                cf.SMOKE_REPLICATE = replicate

                query_indices = cf.choose_queries(
                    data_pretruth["institution_id"].astype(str).to_numpy(),
                    reference_profiles,
                    scenario,
                )

            finally:
                cf.SMOKE_REPLICATE = old_cf_rep

            iqr_denominators = cf.compute_iqr_denominators(X)

            cf_single_results = []

            for source_index in query_indices:
                institution_id = str(
                    data_pretruth["institution_id"].iloc[int(source_index)]
                )

                result = cf.exact_single_model_cf(
                    int(source_index),
                    institution_id,
                    X,
                    reference_profiles,
                    predictor,
                    iqr_denominators,
                )

                cf_single_results.append(result)

            # Freeze single-model CF outputs before any truth use.
            (scenario_dir / "single_model_cf_pretruth.json").write_text(
                json.dumps(
                    {
                        "query_indices": [int(value) for value in query_indices],
                        "query_institution_ids": [
                            str(data_pretruth["institution_id"].iloc[int(value)])
                            for value in query_indices
                        ],
                        "reference_partition_ari": float(predictor["partition_ari"]),
                        "reference_inductive_fidelity": float(
                            predictor["inductive_fidelity"]
                        ),
                        "single_model_query_results": (cf_single_results),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            save_single_model_candidates(
                scenario_dir,
                cf_single_results,
            )

            ensemble_200, robust_report_200 = build_robust_ensemble_sized(
                X,
                reference_profiles,
                scenario,
                replicate,
                members_per_family=50,
                minimum_eligible=120,
            )

            ensemble_1000, robust_report_1000 = build_robust_ensemble_sized(
                X,
                reference_profiles,
                scenario,
                replicate,
                members_per_family=250,
                minimum_eligible=600,
            )

            cf_convergence = {
                "applicable": True,
                "status": "COMPLETED",
                "ensemble_200": (robust_report_200),
                "ensemble_1000": (robust_report_1000),
                "agreement_evaluable": False,
                "tau_090_decision_agreement_200_vs_1000": None,
                "passes_0_95": False,
            }

            if robust_report_200["evaluable"] and robust_report_1000["evaluable"]:
                robust_200_frame, robust_200_results = robust_query_decisions(
                    cf_single_results,
                    ensemble_200,
                )

                robust_1000_frame, robust_1000_results = robust_query_decisions(
                    cf_single_results,
                    ensemble_1000,
                )

                robust_200_frame.to_csv(
                    scenario_dir / "robust_query_decisions_200_pretruth.csv",
                    index=False,
                )

                robust_1000_frame.to_csv(
                    scenario_dir / "robust_query_decisions_1000_pretruth.csv",
                    index=False,
                )

                candidate_200 = cf.candidate_frame(
                    scenario,
                    robust_200_results,
                )

                candidate_1000 = cf.candidate_frame(
                    scenario,
                    robust_1000_results,
                )

                candidate_200.to_csv(
                    scenario_dir / "robust_candidates_200_pretruth.csv",
                    index=False,
                )

                candidate_1000.to_csv(
                    scenario_dir / "robust_candidates_1000_pretruth.csv",
                    index=False,
                )

                comparison = compare_robust_decisions(
                    robust_200_frame,
                    robust_1000_frame,
                )

                cf_convergence.update(
                    {
                        "agreement_evaluable": True,
                        **comparison,
                    }
                )

            (scenario_dir / "robust_convergence_pretruth.json").write_text(
                json.dumps(
                    cf_convergence,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    # -------------------------------------------------------------
    # Truth evaluation starts only after all pretruth outputs above.
    # -------------------------------------------------------------
    truth_report: dict[
        str,
        Any,
    ] = {
        "scenario": scenario,
        "replicate": int(replicate),
    }

    if bundle.truth["true_profile"].notna().all():
        true_profile = bundle.truth["true_profile"].astype(int).to_numpy()

        truth_report["ari_vs_true_profile"] = float(
            adjusted_rand_score(
                true_profile,
                raw_labels,
            )
        )

        truth_report["nmi_vs_true_profile"] = float(
            normalized_mutual_info_score(
                true_profile,
                raw_labels,
            )
        )

    else:
        truth_report["ari_vs_true_profile"] = None

        truth_report["nmi_vs_true_profile"] = None

    if (
        scenario == "S2_CORE_BOUNDARY"
        and certainty_200 is not None
        and certainty_1000 is not None
    ):
        boundary = bundle.truth["true_boundary"].astype(bool).astype(int).to_numpy()

        entropy_200 = certainty_200["normalized_entropy"].to_numpy(dtype=float)

        entropy_1000 = certainty_1000["normalized_entropy"].to_numpy(dtype=float)

        truth_report["boundary_uncertainty_200"] = {
            "auc": float(
                roc_auc_score(
                    boundary,
                    entropy_200,
                )
            ),
            "median_entropy_boundary": float(np.median(entropy_200[boundary == 1])),
            "median_entropy_core": float(np.median(entropy_200[boundary == 0])),
        }

        truth_report["boundary_uncertainty_1000"] = {
            "auc": float(
                roc_auc_score(
                    boundary,
                    entropy_1000,
                )
            ),
            "median_entropy_boundary": float(np.median(entropy_1000[boundary == 1])),
            "median_entropy_core": float(np.median(entropy_1000[boundary == 0])),
        }

    if scenario == "S3_DIRECTIONAL_REACHABILITY" and cf_single_results is not None:
        single_frame = cf.single_model_summary_frame(cf_single_results)

        truth_subset = bundle.truth[
            [
                "institution_id",
                "oracle_reachable",
            ]
        ]

        merged_single = single_frame.merge(
            truth_subset,
            on="institution_id",
            how="left",
            validate="one_to_one",
        )

        y_true = merged_single["oracle_reachable"].astype(bool).astype(int).to_numpy()

        y_single = (
            merged_single["single_model_reachable"].astype(bool).astype(int).to_numpy()
        )

        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
        )

        truth_report["s3_single_model_oracle"] = {
            "oracle_positive_queries": int(y_true.sum()),
            "precision": float(
                precision_score(
                    y_true,
                    y_single,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_single,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_single,
                    zero_division=0,
                )
            ),
        }

        if robust_200_frame is not None:
            truth_report["s3_truth_200"] = cf.evaluate_s3_truth(
                robust_200_frame,
                bundle.truth,
                reference_profiles,
                bundle.metadata,
            )

        if robust_1000_frame is not None:
            truth_report["s3_truth_1000"] = cf.evaluate_s3_truth(
                robust_1000_frame,
                bundle.truth,
                reference_profiles,
                bundle.metadata,
            )

    if scenario == "S4_SEVERITY_CONTINUUM":
        truth_report["false_configurational_claim"] = bool(configurational_claim)

        truth_report["severity_falsification"] = bool(
            severity["severity_nearly_reconstructs_profiles"]
        )

    if scenario == "S5_GOVERNANCE_CONFOUNDED":
        truth_report["false_configurational_claim"] = bool(configurational_claim)

        truth_report["governance_falsification"] = bool(
            governance["strong_governance_association"]
            or governance["governance_nearly_reconstructs_profiles"]
        )

    if scenario == "S6_NO_CLUSTER_NULL":
        truth_report["false_stable_profile_claim"] = bool(stable_partition_claim)

        if robust_200_frame is not None:
            truth_report["false_robust_cf_claim_200"] = bool(
                robust_200_frame["robust_reachable_tau_090"].astype(bool).any()
            )
        else:
            truth_report["false_robust_cf_claim_200"] = None

        if robust_1000_frame is not None:
            truth_report["false_robust_cf_claim_1000"] = bool(
                robust_1000_frame["robust_reachable_tau_090"].astype(bool).any()
            )
        else:
            truth_report["false_robust_cf_claim_1000"] = None

    bundle.truth.to_csv(
        scenario_dir / "truth_posthoc.csv",
        index=False,
    )

    (scenario_dir / "truth_evaluation_posthoc.json").write_text(
        json.dumps(
            truth_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    technical_checks = {
        "n_rows_80": len(data_pretruth) == 80,
        "candidate_grid_20": len(metrics) == 20,
        "shortlist_nonempty": len(shortlist) > 0,
        "official_stability_iterations_1000": all(
            stability["stability_iterations"].astype(int) == 1000
        ),
        "selected_k_valid": selected_k
        in {
            2,
            3,
            4,
            5,
            6,
        },
        "selected_candidate_eligible": bool(selected["selection_eligible"]),
        "binary_label_interface_valid": (
            selected_k != 2
            or set(int(value) for value in np.unique(severity_labels))
            == {
                1,
                2,
            }
        ),
        "uncertainty_convergence_complete_when_applicable": (
            not (stable_partition_claim and selected_k == 2)
            or uncertainty_convergence["status"] == "COMPLETED"
        ),
        "cf_pretruth_saved_when_applicable": (
            not (
                scenario in CF_SCENARIOS and stable_partition_claim and selected_k == 2
            )
            or (
                cf_convergence["status"]
                in {
                    "COMPLETED",
                    "NOT_APPLICABLE_WARD_REFERENCE_EXTENSION_FIDELITY_FAIL",
                }
            )
        ),
    }

    return {
        "scenario": scenario,
        "replicate": int(replicate),
        "selected_candidate": selected_id,
        "selected_k": selected_k,
        "selected_stability_ari_median": float(selected["ari_median"]),
        "stable_partition_claim": stable_partition_claim,
        "configurational_profile_claim": configurational_claim,
        "uncertainty_convergence": uncertainty_convergence,
        "cf_convergence": cf_convergence,
        "truth_report": truth_report,
        "technical_checks": technical_checks,
    }


def main() -> None:
    args = parse_args()

    replicate = int(args.replicate)

    plan = load_json(PLAN_PATH)

    prechecks = {
        "stage8e_plan_passed": (
            plan.get("gate_status") == "PASS_STAGE_8E_OFFICIAL_RUN_PLAN_FREEZE"
        ),
        "replicate_is_frozen_convergence_index": (
            replicate in OFFICIAL_AUDIT_REPLICATES
        ),
        "replicate_is_official": (
            replicate
            in set(plan["replicate_manifest"]["official_replicates_per_scenario"])
        ),
        "replicate_zero_one_excluded": replicate
        not in {
            0,
            1,
        },
    }

    final_root = OFFICIAL_ROOT / f"replicate_{replicate:03d}"

    working_root = OFFICIAL_ROOT / f"replicate_{replicate:03d}__WORKING"

    prechecks["final_output_absent"] = not final_root.exists()

    prechecks["working_output_absent"] = not working_root.exists()

    print("\n=== CIPHER STAGE 8F1 — ONE OFFICIAL CONVERGENCE-AUDIT REPLICATE ===\n")

    print(f"Official replicate: {replicate}")

    print(
        "This run is OFFICIAL. Its scientific outcomes must not be used to retune "
        "thresholds, generators, query budgets, or scenario geometry."
    )

    print("\n=== PRECHECKS ===\n")

    for name, passed in prechecks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(prechecks.values()):
        print("\nGATE STATUS: FAIL_STAGE_8F1_PRECHECK")

        raise SystemExit(1)

    working_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    scenario_results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario} / official replicate {replicate} ---")

        result = run_one_scenario(
            scenario,
            replicate,
            working_root,
        )

        scenario_results.append(result)

        print(
            "  selected:",
            result["selected_candidate"],
        )

        print(
            "  k:",
            result["selected_k"],
            "| stability ARI median:",
            f"{result['selected_stability_ari_median']:.4f}",
            "| stable claim:",
            result["stable_partition_claim"],
        )

        unc = result["uncertainty_convergence"]

        if unc["applicable"]:
            print(
                "  uncertainty convergence Spearman:",
                f"{unc['normalized_entropy_spearman_200_vs_1000']:.4f}",
                "| pass .95:",
                unc["passes_0_95"],
            )

        else:
            print(
                "  uncertainty convergence:",
                unc["status"],
            )

        cfr = result["cf_convergence"]

        if cfr["applicable"]:
            print(
                "  robust ensemble eligible 200/1000:",
                f"{cfr['ensemble_200']['eligible_members']}/"
                f"{cfr['ensemble_1000']['eligible_members']}",
            )

            print(
                "  robust agreement evaluable:",
                cfr["agreement_evaluable"],
            )

            if cfr["agreement_evaluable"]:
                print(
                    "  tau=.90 decision agreement:",
                    f"{cfr['tau_090_decision_agreement_200_vs_1000']:.4f}",
                    "| pass .95:",
                    cfr["passes_0_95"],
                )

        else:
            print(
                "  CF convergence:",
                cfr["status"],
            )

    technical_checks = {
        "prechecks_pass": all(prechecks.values()),
        "six_scenarios_completed": len(scenario_results) == 6,
        "all_scenario_technical_checks_pass": all(
            all(result["technical_checks"].values()) for result in scenario_results
        ),
        "replicate_identity_preserved": all(
            result["replicate"] == replicate for result in scenario_results
        ),
        "no_smoke_index_used": replicate
        not in {
            0,
            1,
        },
    }

    summary_rows = []

    for result in scenario_results:
        unc = result["uncertainty_convergence"]

        cfr = result["cf_convergence"]

        summary_rows.append(
            {
                "scenario": result["scenario"],
                "replicate": replicate,
                "selected_candidate": result["selected_candidate"],
                "selected_k": result["selected_k"],
                "stability_ari_median": result["selected_stability_ari_median"],
                "stable_partition_claim": result["stable_partition_claim"],
                "configurational_profile_claim": result[
                    "configurational_profile_claim"
                ],
                "uncertainty_applicable": unc["applicable"],
                "uncertainty_spearman_200_vs_1000": unc.get(
                    "normalized_entropy_spearman_200_vs_1000"
                ),
                "uncertainty_pass_0_95": unc.get("passes_0_95"),
                "cf_applicable": cfr["applicable"],
                "robust_eligible_200": (
                    cfr.get(
                        "ensemble_200",
                        {},
                    ).get("eligible_members")
                ),
                "robust_eligible_1000": (
                    cfr.get(
                        "ensemble_1000",
                        {},
                    ).get("eligible_members")
                ),
                "robust_agreement_evaluable": cfr.get("agreement_evaluable"),
                "robust_tau090_agreement": cfr.get(
                    "tau_090_decision_agreement_200_vs_1000"
                ),
                "robust_pass_0_95": cfr.get("passes_0_95"),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        working_root / "replicate_convergence_summary.csv",
        index=False,
    )

    gate_status = (
        "PASS_STAGE_8F1_OFFICIAL_REPLICATE"
        if all(technical_checks.values())
        else "FAIL_STAGE_8F1_OFFICIAL_REPLICATE"
    )

    audit = {
        "status": gate_status,
        "official_replicate": replicate,
        "technical_checks": technical_checks,
        "scenario_results": summary_rows,
        "scientific_thresholds_retuned": False,
        "smoke_replicates_included": False,
    }

    (working_root / "replicate_audit.json").write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== OFFICIAL REPLICATE CONVERGENCE SUMMARY ===\n")

    print(summary.to_string(index=False))

    print("\n=== TECHNICAL GATE CHECKS ===\n")

    for name, passed in technical_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {gate_status}")

    if gate_status != "PASS_STAGE_8F1_OFFICIAL_REPLICATE":
        print(
            "Working outputs were retained for debugging. "
            "Do not replace this official replicate with another index."
        )

        raise SystemExit(1)

    # Atomic-ish finalize: the complete working directory becomes the immutable
    # official replicate directory only after every technical gate passes.
    working_root.rename(final_root)

    AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        final_root / "replicate_audit.json",
        AUDIT_ROOT / f"replicate_{replicate:03d}_audit.json",
    )

    print(
        "FINALIZED OFFICIAL REPLICATE:",
        final_root.relative_to(ROOT),
    )

    print("Do not modify or overwrite this finalized directory.")


if __name__ == "__main__":
    main()
