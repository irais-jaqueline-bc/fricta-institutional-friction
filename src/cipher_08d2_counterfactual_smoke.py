from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    precision_score,
    recall_score,
)

from cipher_08d1_core_pipeline_smoke_v2 import (
    align_member_labels,
    apply_representation,
    fit_cluster,
    remap_selected_labels_to_profiles,
    stable_seed,
    transform_representation,
    ward_centroid_predict,
)
from cipher_synthetic_generators import (
    FEATURE_NAMES,
    generate_scenario,
)

ROOT = Path(__file__).resolve().parents[1]

CF_FREEZE_PATH = (
    ROOT / "cipher" / "design" / "stage8_counterfactual_evaluator_freeze_v1.json"
)
EVALUATOR_V2_PATH = ROOT / "cipher" / "design" / "stage8_evaluator_freeze_v2.json"
STAGE8D1_V2_AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8d1_core_pipeline_smoke_audit_v2.json"
)

CORE_SMOKE_ROOT = ROOT / "cipher" / "outputs" / "synthetic" / "performance_smoke_v2"

OUTPUT_ROOT = ROOT / "cipher" / "outputs" / "synthetic" / "counterfactual_smoke"

AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8d2_counterfactual_smoke_audit.json"
)

OFFICIAL_DIR = ROOT / "cipher" / "outputs" / "synthetic" / "official"

MASTER_SEED = 20260807
SMOKE_REPLICATE = 1

SCENARIOS = [
    "S1_CONFIG_TWO_PROFILE",
    "S2_CORE_BOUNDARY",
    "S3_DIRECTIONAL_REACHABILITY",
    "S4_SEVERITY_CONTINUUM",
    "S5_GOVERNANCE_CONFOUNDED",
    "S6_NO_CLUSTER_NULL",
]

CF_ALLOWED_SCENARIOS = {
    "S1_CONFIG_TWO_PROFILE",
    "S2_CORE_BOUNDARY",
    "S3_DIRECTIONAL_REACHABILITY",
    "S6_NO_CLUSTER_NULL",
}

QUERY_COUNT = 20
MAX_CHANGED_FEATURES = 4
TARGET_ANCHORS = 3
L0_PENALTY = 0.25

ENSEMBLE_MEMBERS_PER_FAMILY = 50
ENSEMBLE_SAMPLE_FRACTION = 0.80
ENSEMBLE_FEATURE_COUNT = 11
WARD_FIDELITY_MIN = 0.95
MIN_ELIGIBLE_MEMBERS = 120

PRIMARY_TAU = 0.90
SENSITIVITY_TAUS = [0.80, 0.95]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def official_results_absent() -> bool:
    if not OFFICIAL_DIR.exists():
        return True

    return not any(path.is_file() for path in OFFICIAL_DIR.rglob("*"))


def canonical_reference_labels(
    raw_labels: np.ndarray,
) -> np.ndarray:
    return remap_selected_labels_to_profiles(raw_labels.astype(int))


def align_native_to_reference(
    native_labels: np.ndarray,
    reference_profiles: np.ndarray,
) -> dict[int, int]:
    return align_member_labels(
        native_labels.astype(int),
        reference_profiles.astype(int),
    )


def fit_reference_predictor(
    X: np.ndarray,
    reference_profiles: np.ndarray,
    representation: str,
    algorithm: str,
    scenario: str,
    candidate_id: str,
) -> dict[str, Any]:
    Z, scaler, pca = transform_representation(
        X,
        representation,
    )

    model, native_labels = fit_cluster(
        Z,
        algorithm,
        2,
        stable_seed(
            MASTER_SEED,
            "full",
            scenario,
            candidate_id,
        ),
    )

    mapping = align_native_to_reference(
        native_labels,
        reference_profiles,
    )

    aligned_native = np.array(
        [mapping[int(value)] for value in native_labels],
        dtype=int,
    )

    partition_ari = float(
        adjusted_rand_score(
            reference_profiles,
            aligned_native,
        )
    )

    if algorithm == "KMEANS":
        training_native_prediction = model.predict(Z)

        training_prediction = np.array(
            [mapping[int(value)] for value in training_native_prediction],
            dtype=int,
        )

        inductive_fidelity = float(np.mean(training_prediction == reference_profiles))

        predictor_payload = {
            "algorithm": algorithm,
            "model": model,
        }

    elif algorithm == "HAC_WARD":
        centroid_native = ward_centroid_predict(
            Z,
            native_labels,
            Z,
        )

        centroid_aligned = np.array(
            [mapping[int(value)] for value in centroid_native],
            dtype=int,
        )

        inductive_fidelity = float(np.mean(centroid_aligned == reference_profiles))

        predictor_payload = {
            "algorithm": algorithm,
            "Z_train": Z,
            "native_labels": native_labels,
        }

    else:
        raise KeyError(algorithm)

    return {
        "representation": representation,
        "algorithm": algorithm,
        "scaler": scaler,
        "pca": pca,
        "mapping": mapping,
        "partition_ari": partition_ari,
        "inductive_fidelity": inductive_fidelity,
        **predictor_payload,
    }


def predict_reference(
    predictor: dict[str, Any],
    X_new: np.ndarray,
) -> np.ndarray:
    Z_new = apply_representation(
        X_new,
        predictor["scaler"],
        predictor["pca"],
    )

    if predictor["algorithm"] == "KMEANS":
        native = predictor["model"].predict(Z_new)

    elif predictor["algorithm"] == "HAC_WARD":
        native = ward_centroid_predict(
            predictor["Z_train"],
            predictor["native_labels"],
            Z_new,
        )

    else:
        raise KeyError(predictor["algorithm"])

    return np.array(
        [predictor["mapping"][int(value)] for value in native],
        dtype=int,
    )


def within_profile_5nn_threshold(
    X_target: np.ndarray,
) -> float:
    if len(X_target) < 6:
        raise ValueError(
            "At least six target-profile observations "
            "are required for within-profile 5NN."
        )

    distances = np.sqrt(
        (
            (
                X_target[
                    :,
                    None,
                    :,
                ]
                - X_target[
                    None,
                    :,
                    :,
                ]
            )
            ** 2
        ).sum(axis=2)
    )

    ordered = np.sort(
        distances,
        axis=1,
    )

    fifth_other_neighbor = ordered[
        :,
        5,
    ]

    return float(
        np.quantile(
            fifth_other_neighbor,
            0.95,
        )
    )


def candidate_5nn_distance(
    candidates: np.ndarray,
    X_target: np.ndarray,
) -> np.ndarray:
    distances = np.sqrt(
        (
            (
                candidates[
                    :,
                    None,
                    :,
                ]
                - X_target[
                    None,
                    :,
                    :,
                ]
            )
            ** 2
        ).sum(axis=2)
    )

    ordered = np.sort(
        distances,
        axis=1,
    )

    return ordered[
        :,
        4,
    ]


def compute_iqr_denominators(
    X: np.ndarray,
) -> np.ndarray:
    q75 = np.quantile(
        X,
        0.75,
        axis=0,
    )

    q25 = np.quantile(
        X,
        0.25,
        axis=0,
    )

    return np.maximum(
        q75 - q25,
        1e-6,
    )


def choose_queries(
    institution_ids: np.ndarray,
    reference_profiles: np.ndarray,
    scenario: str,
) -> np.ndarray:
    rng = np.random.default_rng(
        stable_seed(
            MASTER_SEED,
            "cf_query_selection",
            scenario,
            SMOKE_REPLICATE,
        )
    )

    by_profile = {
        profile: np.flatnonzero(reference_profiles == profile) for profile in [1, 2]
    }

    shuffled = {}

    for profile in [1, 2]:
        values = by_profile[profile].copy()

        rng.shuffle(values)

        shuffled[profile] = values

    n1 = len(shuffled[1])
    n2 = len(shuffled[2])

    if n1 >= 10 and n2 >= 10:
        selected = np.concatenate(
            [
                shuffled[1][:10],
                shuffled[2][:10],
            ]
        )

    else:
        smaller = 1 if n1 < n2 else 2
        larger = 2 if smaller == 1 else 1

        small_take = min(
            10,
            len(shuffled[smaller]),
        )

        large_take = QUERY_COUNT - small_take

        if len(shuffled[larger]) < large_take:
            raise RuntimeError("Unable to select 20 truth-blind queries.")

        selected = np.concatenate(
            [
                shuffled[smaller][:small_take],
                shuffled[larger][:large_take],
            ]
        )

    selected = selected.astype(int)

    if len(selected) != QUERY_COUNT:
        raise RuntimeError(f"Expected {QUERY_COUNT} queries, " f"got {len(selected)}.")

    if len(np.unique(selected)) != QUERY_COUNT:
        raise RuntimeError("Duplicate query indices.")

    # Stable order for reproducible files.
    return np.array(
        sorted(
            selected.tolist(),
            key=lambda idx: str(institution_ids[idx]),
        ),
        dtype=int,
    )


def nearest_target_anchors(
    source: np.ndarray,
    X_target: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    distances = np.sqrt(
        (
            (
                X_target
                - source[
                    None,
                    :,
                ]
            )
            ** 2
        ).sum(axis=1)
    )

    order = np.lexsort(
        (
            target_indices,
            distances,
        )
    )

    return target_indices[order[:TARGET_ANCHORS]]


def alternatives_for_feature(
    current: float,
    anchor_values: np.ndarray,
) -> list[float]:
    values = [
        float(value)
        for value in anchor_values
        if not np.isclose(
            float(value),
            float(current),
            atol=1e-12,
            rtol=0,
        )
    ]

    # Deterministic de-duplication.
    unique = []

    for value in values:
        if not any(
            np.isclose(
                value,
                existing,
                atol=1e-12,
                rtol=0,
            )
            for existing in unique
        ):
            unique.append(value)

    return unique


def signed_edit_signature(
    source: np.ndarray,
    candidate: np.ndarray,
) -> tuple[str, ...]:
    signature = []

    for j, feature in enumerate(FEATURE_NAMES):
        delta = candidate[j] - source[j]

        if np.isclose(
            delta,
            0.0,
            atol=1e-12,
            rtol=0,
        ):
            continue

        direction = "UP" if delta > 0 else "DOWN"

        signature.append(f"{feature}:{direction}")

    return tuple(signature)


def pareto_front_exact(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda row: (
            row["total_cost"],
            row["l0"],
            row["plausibility_distance"],
            row["edit_signature"],
        ),
    )

    # Prior lower-cost minima by exact L0.
    prior_min_plausibility = {
        l0: np.inf
        for l0 in range(
            1,
            MAX_CHANGED_FEATURES + 1,
        )
    }

    pareto = []
    start = 0

    while start < len(ordered):
        cost = ordered[start]["total_cost"]

        end = start + 1

        while end < len(ordered) and ordered[end]["total_cost"] == cost:
            end += 1

        group = ordered[start:end]

        group_min_by_l0 = {
            l0: np.inf
            for l0 in range(
                1,
                MAX_CHANGED_FEATURES + 1,
            )
        }

        for row in group:
            l0 = int(row["l0"])

            group_min_by_l0[l0] = min(
                group_min_by_l0[l0],
                float(row["plausibility_distance"]),
            )

        for row in group:
            l0 = int(row["l0"])
            plausibility = float(row["plausibility_distance"])

            lower_cost_prefix = min(
                prior_min_plausibility[value]
                for value in range(
                    1,
                    l0 + 1,
                )
            )

            dominated_by_lower_cost = bool(lower_cost_prefix <= plausibility)

            smaller_l0_same_cost = min(
                (
                    group_min_by_l0[value]
                    for value in range(
                        1,
                        l0,
                    )
                ),
                default=np.inf,
            )

            same_l0_best_plausibility = group_min_by_l0[l0]

            dominated_same_cost = bool(
                smaller_l0_same_cost <= plausibility
                or same_l0_best_plausibility < plausibility
            )

            if not (dominated_by_lower_cost or dominated_same_cost):
                pareto.append(row)

        for l0 in range(
            1,
            MAX_CHANGED_FEATURES + 1,
        ):
            prior_min_plausibility[l0] = min(
                prior_min_plausibility[l0],
                group_min_by_l0[l0],
            )

        start = end

    return pareto


def select_diverse_pareto(
    pareto: list[dict[str, Any]],
    max_saved: int = 5,
) -> list[dict[str, Any]]:
    ordered = sorted(
        pareto,
        key=lambda row: (
            row["total_cost"],
            row["l0"],
            row["plausibility_distance"],
            row["edit_signature"],
        ),
    )

    selected = []
    seen_signatures = set()

    for row in ordered:
        signature = tuple(row["edit_signature"])

        if signature in seen_signatures:
            continue

        selected.append(row)
        seen_signatures.add(signature)

        if len(selected) >= max_saved:
            return selected

    for row in ordered:
        if row in selected:
            continue

        selected.append(row)

        if len(selected) >= max_saved:
            break

    return selected


def exact_single_model_cf(
    source_index: int,
    institution_id: str,
    X: np.ndarray,
    reference_profiles: np.ndarray,
    predictor: dict[str, Any],
    iqr_denominators: np.ndarray,
) -> dict[str, Any]:
    source = X[source_index].copy()

    source_profile = int(reference_profiles[source_index])

    target_profile = 2 if source_profile == 1 else 1

    target_indices = np.flatnonzero(reference_profiles == target_profile)

    if len(target_indices) < 6:
        return {
            "institution_id": institution_id,
            "source_index": int(source_index),
            "source_profile": source_profile,
            "target_profile": target_profile,
            "status": "NOT_APPLICABLE_TARGET_PROFILE_TOO_SMALL_FOR_5NN",
            "candidate_count_evaluated": 0,
            "valid_candidate_count": 0,
            "pareto_candidate_count": 0,
            "single_model_reachable": False,
            "saved_candidates": [],
        }

    X_target = X[target_indices]

    threshold = within_profile_5nn_threshold(X_target)

    anchor_indices = nearest_target_anchors(
        source,
        X_target,
        target_indices,
    )

    anchor_matrix = X[anchor_indices]

    alternatives = {
        j: alternatives_for_feature(
            source[j],
            anchor_matrix[:, j],
        )
        for j in range(len(FEATURE_NAMES))
    }

    valid_rows = []
    candidate_count = 0

    for l0 in range(
        1,
        MAX_CHANGED_FEATURES + 1,
    ):
        for feature_indices in itertools.combinations(
            range(len(FEATURE_NAMES)),
            l0,
        ):
            if any(len(alternatives[feature]) == 0 for feature in feature_indices):
                continue

            value_lists = [alternatives[feature] for feature in feature_indices]

            products = list(itertools.product(*value_lists))

            if not products:
                continue

            candidates = np.repeat(
                source[
                    None,
                    :,
                ],
                len(products),
                axis=0,
            )

            for row_index, values in enumerate(products):
                for feature, value in zip(
                    feature_indices,
                    values,
                ):
                    candidates[
                        row_index,
                        feature,
                    ] = value

            candidate_count += len(candidates)

            predicted = predict_reference(
                predictor,
                candidates,
            )

            transition_mask = predicted == target_profile

            if not transition_mask.any():
                continue

            transitioned = candidates[transition_mask]

            plausibility = candidate_5nn_distance(
                transitioned,
                X_target,
            )

            plausible_mask = plausibility <= threshold

            if not plausible_mask.any():
                continue

            plausible_candidates = transitioned[plausible_mask]
            plausible_distances = plausibility[plausible_mask]

            changed = ~np.isclose(
                plausible_candidates,
                source[
                    None,
                    :,
                ],
                atol=1e-12,
                rtol=0,
            )

            weighted_l1 = (
                np.abs(
                    plausible_candidates
                    - source[
                        None,
                        :,
                    ]
                )
                / iqr_denominators[None, :]
            ).sum(axis=1)

            l0_counts = changed.sum(axis=1).astype(int)

            total_costs = weighted_l1 + L0_PENALTY * l0_counts

            for (
                candidate,
                plaus_distance,
                l0_count,
                weighted,
                total_cost,
            ) in zip(
                plausible_candidates,
                plausible_distances,
                l0_counts,
                weighted_l1,
                total_costs,
            ):
                signature = signed_edit_signature(
                    source,
                    candidate,
                )

                valid_rows.append(
                    {
                        "candidate_vector": candidate.copy(),
                        "l0": int(l0_count),
                        "weighted_l1": float(weighted),
                        "total_cost": float(total_cost),
                        "plausibility_distance": float(plaus_distance),
                        "plausibility_threshold": float(threshold),
                        "edit_signature": signature,
                    }
                )

    pareto = pareto_front_exact(valid_rows)

    saved = select_diverse_pareto(
        pareto,
        max_saved=5,
    )

    serialized = []

    for rank, row in enumerate(
        saved,
        start=1,
    ):
        candidate = row["candidate_vector"]

        serialized.append(
            {
                "rank": int(rank),
                "candidate_id": (f"{institution_id}__CF{rank:02d}"),
                "l0": int(row["l0"]),
                "weighted_l1": float(row["weighted_l1"]),
                "total_cost": float(row["total_cost"]),
                "plausibility_distance": float(row["plausibility_distance"]),
                "plausibility_threshold": float(row["plausibility_threshold"]),
                "edit_signature": list(row["edit_signature"]),
                "candidate_vector": [float(value) for value in candidate],
            }
        )

    return {
        "institution_id": institution_id,
        "source_index": int(source_index),
        "source_profile": source_profile,
        "target_profile": target_profile,
        "status": "COMPLETED",
        "target_profile_size": int(len(target_indices)),
        "target_anchor_indices": [int(value) for value in anchor_indices],
        "target_anchor_count": int(len(anchor_indices)),
        "candidate_count_evaluated": int(candidate_count),
        "valid_candidate_count": int(len(valid_rows)),
        "pareto_candidate_count": int(len(pareto)),
        "single_model_reachable": bool(len(valid_rows) > 0),
        "saved_candidates": serialized,
    }


def build_robust_ensemble(
    X: np.ndarray,
    reference_profiles: np.ndarray,
    scenario: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    n = len(X)

    sample_size = int(round(n * ENSEMBLE_SAMPLE_FRACTION))

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

    members = []
    family_counts = {}

    for (
        family,
        representation,
        algorithm,
    ) in family_specs:
        rng = np.random.default_rng(
            stable_seed(
                MASTER_SEED,
                "cf_ensemble_family",
                scenario,
                family,
                SMOKE_REPLICATE,
            )
        )

        accepted = 0
        excluded_fidelity = 0

        for member_index in range(ENSEMBLE_MEMBERS_PER_FAMILY):
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

            feature_idx = np.sort(
                rng.choice(
                    np.arange(
                        X.shape[1],
                        dtype=int,
                    ),
                    size=ENSEMBLE_FEATURE_COUNT,
                    replace=False,
                )
            )

            X_sample = X[sample][:, feature_idx]

            Z_sample, scaler, pca = transform_representation(
                X_sample,
                representation,
            )

            model, native_labels = fit_cluster(
                Z_sample,
                algorithm,
                2,
                stable_seed(
                    MASTER_SEED,
                    "cf_ensemble_fit",
                    scenario,
                    family,
                    member_index,
                ),
            )

            mapping = align_member_labels(
                native_labels,
                reference_profiles[sample],
            )

            if algorithm == "HAC_WARD":
                native_inductive = ward_centroid_predict(
                    Z_sample,
                    native_labels,
                    Z_sample,
                )

                aligned_inductive = np.array(
                    [mapping[int(value)] for value in native_inductive],
                    dtype=int,
                )

                fidelity = float(
                    np.mean(aligned_inductive == reference_profiles[sample])
                )

                if fidelity < WARD_FIDELITY_MIN:
                    excluded_fidelity += 1
                    continue

                payload = {
                    "Z_train": Z_sample,
                    "native_labels": native_labels,
                }

            else:
                fidelity = 1.0
                payload = {
                    "model": model,
                }

            members.append(
                {
                    "member_id": (f"{family}__{member_index:03d}"),
                    "family": family,
                    "representation": representation,
                    "algorithm": algorithm,
                    "feature_idx": feature_idx,
                    "scaler": scaler,
                    "pca": pca,
                    "mapping": mapping,
                    "fidelity": fidelity,
                    **payload,
                }
            )

            accepted += 1

        family_counts[family] = {
            "accepted": int(accepted),
            "excluded_fidelity": int(excluded_fidelity),
        }

    report = {
        "eligible_members": int(len(members)),
        "minimum_required": int(MIN_ELIGIBLE_MEMBERS),
        "family_counts": family_counts,
    }

    return members, report


def predict_member(
    member: dict[str, Any],
    candidate: np.ndarray,
) -> int:
    feature_idx = member["feature_idx"]

    X_sub = candidate[
        None,
        feature_idx,
    ]

    Z_new = apply_representation(
        X_sub,
        member["scaler"],
        member["pca"],
    )

    if member["algorithm"] == "KMEANS":
        native = int(member["model"].predict(Z_new)[0])

    else:
        native = int(
            ward_centroid_predict(
                member["Z_train"],
                member["native_labels"],
                Z_new,
            )[0]
        )

    return int(member["mapping"][native])


def evaluate_robust_support(
    institution_result: dict[str, Any],
    ensemble: list[dict[str, Any]],
) -> dict[str, Any]:
    saved = institution_result["saved_candidates"]

    candidate_results = []

    for candidate in saved:
        vector = np.array(
            candidate["candidate_vector"],
            dtype=float,
        )

        target_profile = int(institution_result["target_profile"])

        predictions = np.array(
            [
                predict_member(
                    member,
                    vector,
                )
                for member in ensemble
            ],
            dtype=int,
        )

        support = float(np.mean(predictions == target_profile))

        family_support = {}

        for family in sorted({member["family"] for member in ensemble}):
            mask = np.array(
                [member["family"] == family for member in ensemble],
                dtype=bool,
            )

            family_support[family] = float(np.mean(predictions[mask] == target_profile))

        candidate_results.append(
            {
                **candidate,
                "ensemble_support": support,
                "minimum_family_support": float(min(family_support.values())),
                "family_support": family_support,
                "robust_tau_080": bool(support >= 0.80),
                "robust_tau_090": bool(support >= 0.90),
                "robust_tau_095": bool(support >= 0.95),
            }
        )

    robust_080 = any(row["robust_tau_080"] for row in candidate_results)

    robust_090 = any(row["robust_tau_090"] for row in candidate_results)

    robust_095 = any(row["robust_tau_095"] for row in candidate_results)

    return {
        **{
            key: value
            for key, value in institution_result.items()
            if key != "saved_candidates"
        },
        "saved_candidates": candidate_results,
        "robust_reachable_tau_080": bool(robust_080),
        "robust_reachable_tau_090": bool(robust_090),
        "robust_reachable_tau_095": bool(robust_095),
    }


def overlap_map_reference_to_latent(
    reference_profiles: np.ndarray,
    latent_profiles: np.ndarray,
) -> dict[int, str]:
    reference_values = np.array(
        [1, 2],
        dtype=int,
    )

    latent_values = np.array(
        ["A", "B"],
        dtype=object,
    )

    contingency = np.zeros(
        (
            2,
            2,
        ),
        dtype=int,
    )

    for i, reference in enumerate(reference_values):
        for j, latent in enumerate(latent_values):
            contingency[i, j] = int(
                np.sum((reference_profiles == reference) & (latent_profiles == latent))
            )

    rows, cols = linear_sum_assignment(-contingency)

    return {
        int(reference_values[row]): str(latent_values[col])
        for row, col in zip(
            rows,
            cols,
        )
    }


def evaluate_s3_truth(
    institution_rows: pd.DataFrame,
    truth: pd.DataFrame,
    reference_profiles: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    truth_subset = truth[
        [
            "institution_id",
            "oracle_reachable",
            "latent_profile",
            "true_profile",
        ]
    ].copy()

    merged = institution_rows.merge(
        truth_subset,
        on="institution_id",
        how="left",
        validate="one_to_one",
    )

    y_true = merged["oracle_reachable"].astype(bool).astype(int).to_numpy()

    y_single = merged["single_model_reachable"].astype(bool).astype(int).to_numpy()

    y_robust = merged["robust_reachable_tau_090"].astype(bool).astype(int).to_numpy()

    single = {
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

    robust = {
        "precision": float(
            precision_score(
                y_true,
                y_robust,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_robust,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_robust,
                zero_division=0,
            )
        ),
    }

    latent_all = truth["latent_profile"].astype(str).to_numpy()

    mapping = overlap_map_reference_to_latent(
        reference_profiles,
        latent_all,
    )

    robust_rates = (
        merged.groupby(
            "source_profile",
            observed=True,
        )["robust_reachable_tau_090"]
        .mean()
        .to_dict()
    )

    rate_1 = float(
        robust_rates.get(
            1,
            0.0,
        )
    )

    rate_2 = float(
        robust_rates.get(
            2,
            0.0,
        )
    )

    if np.isclose(
        rate_1,
        rate_2,
        atol=1e-12,
        rtol=0,
    ):
        predicted_source_profile = None
        predicted_source_latent = None
        direction_recovered = False

    else:
        predicted_source_profile = 1 if rate_1 > rate_2 else 2

        predicted_source_latent = mapping[predicted_source_profile]

        direction_recovered = bool(
            predicted_source_latent == str(metadata["accessible_source_latent"])
        )

    # Frozen numeric-label swap sanity check:
    # swap only true_profile; oracle_reachable and latent A/B remain unchanged.
    swapped_truth = truth_subset.copy()

    swapped_truth["true_profile"] = swapped_truth["true_profile"].map(
        {
            1: 2,
            2: 1,
        }
    )

    swapped_merged = institution_rows.merge(
        swapped_truth,
        on="institution_id",
        how="left",
        validate="one_to_one",
    )

    swapped_y_true = (
        swapped_merged["oracle_reachable"].astype(bool).astype(int).to_numpy()
    )

    swap_single_f1 = float(
        f1_score(
            swapped_y_true,
            y_single,
            zero_division=0,
        )
    )

    swap_robust_precision = float(
        precision_score(
            swapped_y_true,
            y_robust,
            zero_division=0,
        )
    )

    label_swap_invariant = bool(
        np.isclose(
            swap_single_f1,
            single["f1"],
            atol=1e-12,
            rtol=0,
        )
        and np.isclose(
            swap_robust_precision,
            robust["precision"],
            atol=1e-12,
            rtol=0,
        )
        and direction_recovered == direction_recovered
    )

    return {
        "queried_oracle_positive_count": int(y_true.sum()),
        "queried_oracle_negative_count": int(len(y_true) - y_true.sum()),
        "single_model": single,
        "robust_tau_090": robust,
        "reference_profile_to_latent_mapping": {
            str(key): value for key, value in mapping.items()
        },
        "robust_reachable_rate_by_reference_profile": {
            "1": rate_1,
            "2": rate_2,
        },
        "predicted_accessible_source_profile": predicted_source_profile,
        "predicted_accessible_source_latent": predicted_source_latent,
        "planted_accessible_source_latent": str(metadata["accessible_source_latent"]),
        "direction_recovered": bool(direction_recovered),
        "numeric_true_profile_label_swap_invariant": label_swap_invariant,
    }


def institution_summary_frame(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for result in results:
        candidate_supports = [
            float(candidate["ensemble_support"])
            for candidate in result["saved_candidates"]
        ]

        rows.append(
            {
                "institution_id": result["institution_id"],
                "source_index": result["source_index"],
                "source_profile": result["source_profile"],
                "target_profile": result["target_profile"],
                "status": result["status"],
                "candidate_count_evaluated": result["candidate_count_evaluated"],
                "valid_candidate_count": result["valid_candidate_count"],
                "pareto_candidate_count": result["pareto_candidate_count"],
                "saved_candidate_count": len(result["saved_candidates"]),
                "single_model_reachable": result["single_model_reachable"],
                "best_ensemble_support": (
                    max(candidate_supports) if candidate_supports else np.nan
                ),
                "robust_reachable_tau_080": result.get(
                    "robust_reachable_tau_080",
                    False,
                ),
                "robust_reachable_tau_090": result.get(
                    "robust_reachable_tau_090",
                    False,
                ),
                "robust_reachable_tau_095": result.get(
                    "robust_reachable_tau_095",
                    False,
                ),
            }
        )

    return pd.DataFrame(rows)


def candidate_frame(
    scenario: str,
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for result in results:
        for candidate in result["saved_candidates"]:
            row = {
                "scenario": scenario,
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
                "ensemble_support": candidate["ensemble_support"],
                "minimum_family_support": candidate["minimum_family_support"],
                "family_support_json": json.dumps(
                    candidate["family_support"],
                    sort_keys=True,
                ),
                "robust_tau_080": candidate["robust_tau_080"],
                "robust_tau_090": candidate["robust_tau_090"],
                "robust_tau_095": candidate["robust_tau_095"],
            }

            for j, feature in enumerate(FEATURE_NAMES):
                row[f"candidate__{feature}"] = candidate["candidate_vector"][j]

            rows.append(row)

    return pd.DataFrame(rows)


def run_applicable_scenario(
    scenario: str,
    pretruth: dict[str, Any],
    bundle,
) -> dict[str, Any]:
    scenario_core_dir = CORE_SMOKE_ROOT / scenario

    labels = pd.read_csv(scenario_core_dir / "selected_labels.csv")

    data = bundle.data.copy()

    merged = data.merge(
        labels,
        on="institution_id",
        how="left",
        validate="one_to_one",
    )

    if merged["cluster_id"].isna().any():
        raise RuntimeError(f"{scenario}: missing selected labels after merge.")

    X = merged[FEATURE_NAMES].to_numpy(dtype=float)

    institution_ids = merged["institution_id"].astype(str).to_numpy()

    raw_labels = merged["cluster_id"].astype(int).to_numpy()

    reference_profiles = canonical_reference_labels(raw_labels)

    predictor = fit_reference_predictor(
        X,
        reference_profiles,
        str(pretruth["selected_representation"]),
        str(pretruth["selected_algorithm"]),
        scenario,
        str(pretruth["selected_candidate"]),
    )

    if predictor["partition_ari"] < 0.999999:
        raise RuntimeError(
            f"{scenario}: fitted reference model does not reproduce "
            f"stored selected partition (ARI={predictor['partition_ari']:.6f})."
        )

    if (
        predictor["algorithm"] == "HAC_WARD"
        and predictor["inductive_fidelity"] < WARD_FIDELITY_MIN
    ):
        return {
            "status": "NOT_APPLICABLE_WARD_REFERENCE_EXTENSION_FIDELITY_FAIL",
            "scenario": scenario,
            "reference_partition_ari": predictor["partition_ari"],
            "reference_inductive_fidelity": predictor["inductive_fidelity"],
            "query_results": [],
            "ensemble_report": None,
            "truth_evaluation": None,
        }

    query_indices = choose_queries(
        institution_ids,
        reference_profiles,
        scenario,
    )

    iqr_denominators = compute_iqr_denominators(X)

    single_results = []

    print(
        f"  selected 20 truth-blind queries: "
        f"P1={(reference_profiles[query_indices] == 1).sum()}, "
        f"P2={(reference_profiles[query_indices] == 2).sum()}"
    )

    for query_number, source_index in enumerate(
        query_indices,
        start=1,
    ):
        institution_id = str(institution_ids[source_index])

        result = exact_single_model_cf(
            int(source_index),
            institution_id,
            X,
            reference_profiles,
            predictor,
            iqr_denominators,
        )

        single_results.append(result)

        print(
            f"  [{query_number:02d}/20] {institution_id} "
            f"P{result['source_profile']}→P{result['target_profile']} "
            f"evaluated={result['candidate_count_evaluated']} "
            f"valid={result['valid_candidate_count']} "
            f"saved={len(result['saved_candidates'])}"
        )

    scenario_out = OUTPUT_ROOT / scenario

    scenario_out.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Save the complete single-model outputs BEFORE building truth evaluation.
    single_pretruth = {
        "scenario": scenario,
        "selected_candidate": pretruth["selected_candidate"],
        "selected_k": 2,
        "reference_partition_ari": predictor["partition_ari"],
        "reference_inductive_fidelity": predictor["inductive_fidelity"],
        "query_indices": [int(value) for value in query_indices],
        "query_institution_ids": [
            str(institution_ids[value]) for value in query_indices
        ],
        "iqr_denominators": {
            feature: float(iqr_denominators[j])
            for j, feature in enumerate(FEATURE_NAMES)
        },
        "single_model_query_results": single_results,
    }

    (scenario_out / "single_model_pretruth.json").write_text(
        json.dumps(
            single_pretruth,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ensemble, ensemble_report = build_robust_ensemble(
        X,
        reference_profiles,
        scenario,
    )

    if ensemble_report["eligible_members"] < MIN_ELIGIBLE_MEMBERS:
        raise RuntimeError(
            f"{scenario}: only "
            f"{ensemble_report['eligible_members']} eligible ensemble members."
        )

    robust_results = [
        evaluate_robust_support(
            result,
            ensemble,
        )
        for result in single_results
    ]

    institution_frame = institution_summary_frame(robust_results)

    candidate_results = candidate_frame(
        scenario,
        robust_results,
    )

    institution_frame.to_csv(
        scenario_out / "institution_cf_summary_pretruth.csv",
        index=False,
    )

    candidate_results.to_csv(
        scenario_out / "saved_counterfactual_candidates_pretruth.csv",
        index=False,
    )

    robust_pretruth = {
        "scenario": scenario,
        "ensemble_report": ensemble_report,
        "single_model_reachable_count": int(
            institution_frame["single_model_reachable"].sum()
        ),
        "robust_reachable_tau_080_count": int(
            institution_frame["robust_reachable_tau_080"].sum()
        ),
        "robust_reachable_tau_090_count": int(
            institution_frame["robust_reachable_tau_090"].sum()
        ),
        "robust_reachable_tau_095_count": int(
            institution_frame["robust_reachable_tau_095"].sum()
        ),
    }

    (scenario_out / "robust_pretruth_report.json").write_text(
        json.dumps(
            robust_pretruth,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Truth may be accessed only after every pipeline/CF output above is frozen.
    truth_evaluation = None

    if scenario == "S3_DIRECTIONAL_REACHABILITY":
        truth_evaluation = evaluate_s3_truth(
            institution_frame,
            bundle.truth,
            reference_profiles,
            bundle.metadata,
        )

    elif scenario == "S6_NO_CLUSTER_NULL":
        truth_evaluation = {
            "false_robust_cf_claim_tau_090": bool(
                institution_frame["robust_reachable_tau_090"].any()
            ),
            "cf_applicable": True,
            "queried_institutions": int(len(institution_frame)),
        }

    elif scenario in {
        "S1_CONFIG_TWO_PROFILE",
        "S2_CORE_BOUNDARY",
    }:
        truth_evaluation = {
            "oracle_reachability_metrics": "NOT_DEFINED",
            "scientific_role": "DIAGNOSTIC_ONLY",
        }

    (scenario_out / "truth_evaluation_smoke_only.json").write_text(
        json.dumps(
            truth_evaluation,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "status": "COMPLETED",
        "scenario": scenario,
        "reference_partition_ari": predictor["partition_ari"],
        "reference_inductive_fidelity": predictor["inductive_fidelity"],
        "query_results": robust_results,
        "ensemble_report": ensemble_report,
        "truth_evaluation": truth_evaluation,
    }


def main() -> None:
    cf_freeze = load_json(CF_FREEZE_PATH)

    evaluator_v2 = load_json(EVALUATOR_V2_PATH)

    stage8d1_v2 = load_json(STAGE8D1_V2_AUDIT_PATH)

    prechecks = {
        "stage8c2_cf_freeze_passed": (
            cf_freeze.get("gate_status")
            == "PASS_STAGE_8C2_COUNTERFACTUAL_EVALUATOR_FREEZE"
        ),
        "stage8c1_evaluator_v2_passed": (
            evaluator_v2.get("gate_status") == "PASS_STAGE_8C1_MULTICLASS_AMENDMENT"
        ),
        "stage8d1_v2_core_smoke_passed": (
            stage8d1_v2.get("status") == "PASS_STAGE_8D1_V2_CORE_PIPELINE_SMOKE"
        ),
        "core_smoke_outputs_exist": (CORE_SMOKE_ROOT.exists()),
        "cf_smoke_output_absent": (not OUTPUT_ROOT.exists()),
        "official_outputs_absent": official_results_absent(),
    }

    print("\n=== CIPHER STAGE 8D2 — COUNTERFACTUAL / ROBUST-REACHABILITY SMOKE ===\n")

    print(
        "NON-OFFICIAL smoke only. "
        "No discovery model is reselected and no smoke value is a scientific result."
    )

    print("\n=== PRECHECKS ===\n")

    for name, passed in prechecks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(prechecks.values()):
        print("\nGATE STATUS: FAIL_STAGE_8D2_PRECHECK")
        raise SystemExit(1)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    scenario_results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario} ---")

        core_dir = CORE_SMOKE_ROOT / scenario

        pretruth = load_json(core_dir / "pipeline_pretruth_report.json")

        selected_k = int(pretruth["selected_k"])

        stable_claim = bool(pretruth["stable_partition_claim"])

        if scenario not in CF_ALLOWED_SCENARIOS:
            status = "NOT_APPLICABLE_SCENARIO_POLICY"

            print(
                "  CF status:",
                status,
            )

            scenario_results.append(
                {
                    "scenario": scenario,
                    "status": status,
                    "selected_k": selected_k,
                    "stable_claim": stable_claim,
                }
            )

            continue

        if not stable_claim:
            status = "NOT_APPLICABLE_NO_STABLE_PARTITION_CLAIM"

            print(
                "  CF status:",
                status,
            )

            scenario_results.append(
                {
                    "scenario": scenario,
                    "status": status,
                    "selected_k": selected_k,
                    "stable_claim": stable_claim,
                }
            )

            continue

        if selected_k != 2:
            status = "NOT_APPLICABLE_SELECTED_K_NOT_2"

            print(
                "  CF status:",
                status,
            )

            scenario_results.append(
                {
                    "scenario": scenario,
                    "status": status,
                    "selected_k": selected_k,
                    "stable_claim": stable_claim,
                }
            )

            continue

        bundle = generate_scenario(
            scenario_id=scenario,
            replicate=SMOKE_REPLICATE,
            master_seed=MASTER_SEED,
        )

        result = run_applicable_scenario(
            scenario,
            pretruth,
            bundle,
        )

        result["selected_k"] = selected_k
        result["stable_claim"] = stable_claim

        scenario_results.append(result)

        if result["status"] == "COMPLETED":
            frame = institution_summary_frame(result["query_results"])

            print(
                "  reference partition ARI:",
                f"{result['reference_partition_ari']:.4f}",
            )

            print(
                "  reference inductive fidelity:",
                f"{result['reference_inductive_fidelity']:.4f}",
            )

            print(
                "  eligible ensemble members:",
                result["ensemble_report"]["eligible_members"],
            )

            print(
                "  single-model reachable:",
                f"{int(frame['single_model_reachable'].sum())}/20",
            )

            print(
                "  robust reachable tau=.80/.90/.95:",
                f"{int(frame['robust_reachable_tau_080'].sum())}/"
                f"{int(frame['robust_reachable_tau_090'].sum())}/"
                f"{int(frame['robust_reachable_tau_095'].sum())}",
            )

            if scenario == "S3_DIRECTIONAL_REACHABILITY":
                truth = result["truth_evaluation"]

                print(
                    "  SMOKE S3 oracle positives in query set:",
                    truth["queried_oracle_positive_count"],
                )

                print(
                    "  SMOKE single-model oracle F1:",
                    f"{truth['single_model']['f1']:.4f}",
                )

                print(
                    "  SMOKE robust precision/F1:",
                    f"{truth['robust_tau_090']['precision']:.4f}/"
                    f"{truth['robust_tau_090']['f1']:.4f}",
                )

                print(
                    "  SMOKE direction recovered:",
                    truth["direction_recovered"],
                )

                print(
                    "  SMOKE numeric-label-swap invariant:",
                    truth["numeric_true_profile_label_swap_invariant"],
                )

            if scenario == "S6_NO_CLUSTER_NULL":
                print(
                    "  SMOKE false robust-CF claim:",
                    result["truth_evaluation"]["false_robust_cf_claim_tau_090"],
                )

    completed = [
        result for result in scenario_results if result["status"] == "COMPLETED"
    ]

    expected_completed = {
        "S1_CONFIG_TWO_PROFILE",
        "S3_DIRECTIONAL_REACHABILITY",
        "S6_NO_CLUSTER_NULL",
    }

    actually_completed = {result["scenario"] for result in completed}

    technical_checks = {
        "prechecks_pass": all(prechecks.values()),
        "expected_smoke_applicability": (actually_completed == expected_completed),
        "s2_abstains_due_selected_k3": any(
            result["scenario"] == "S2_CORE_BOUNDARY"
            and result["status"] == "NOT_APPLICABLE_SELECTED_K_NOT_2"
            for result in scenario_results
        ),
        "s4_s5_abstain_by_policy": all(
            any(
                result["scenario"] == scenario
                and result["status"] == "NOT_APPLICABLE_SCENARIO_POLICY"
                for result in scenario_results
            )
            for scenario in [
                "S4_SEVERITY_CONTINUUM",
                "S5_GOVERNANCE_CONFOUNDED",
            ]
        ),
        "all_completed_have_20_queries": all(
            len(result["query_results"]) == 20 for result in completed
        ),
        "all_reference_partitions_reproduced": all(
            result["reference_partition_ari"] >= 0.999999 for result in completed
        ),
        "all_ward_reference_fidelities_pass": all(
            (result["reference_inductive_fidelity"] >= WARD_FIDELITY_MIN)
            for result in completed
            if (
                load_json(
                    CORE_SMOKE_ROOT
                    / result["scenario"]
                    / "pipeline_pretruth_report.json"
                )["selected_algorithm"]
                == "HAC_WARD"
            )
        ),
        "all_ensembles_meet_minimum": all(
            result["ensemble_report"]["eligible_members"] >= MIN_ELIGIBLE_MEMBERS
            for result in completed
        ),
        "saved_candidates_at_most_5": all(
            len(institution["saved_candidates"]) <= 5
            for result in completed
            for institution in result["query_results"]
        ),
        "all_saved_candidates_respect_max_l0": all(
            candidate["l0"] <= MAX_CHANGED_FEATURES
            for result in completed
            for institution in result["query_results"]
            for candidate in institution["saved_candidates"]
        ),
        "all_saved_candidates_pass_plausibility": all(
            candidate["plausibility_distance"]
            <= candidate["plausibility_threshold"] + 1e-12
            for result in completed
            for institution in result["query_results"]
            for candidate in institution["saved_candidates"]
        ),
        "s3_truth_eval_present": any(
            result["scenario"] == "S3_DIRECTIONAL_REACHABILITY"
            and result["truth_evaluation"] is not None
            for result in completed
        ),
        "official_outputs_untouched": official_results_absent(),
    }

    summary_rows = []

    for result in scenario_results:
        row = {
            "scenario": result["scenario"],
            "status": result["status"],
            "selected_k": result["selected_k"],
            "stable_claim": result["stable_claim"],
        }

        if result["status"] == "COMPLETED":
            frame = institution_summary_frame(result["query_results"])

            row.update(
                {
                    "eligible_ensemble_members": result["ensemble_report"][
                        "eligible_members"
                    ],
                    "single_model_reachable_n": int(
                        frame["single_model_reachable"].sum()
                    ),
                    "robust_tau_080_n": int(frame["robust_reachable_tau_080"].sum()),
                    "robust_tau_090_n": int(frame["robust_reachable_tau_090"].sum()),
                    "robust_tau_095_n": int(frame["robust_reachable_tau_095"].sum()),
                }
            )

            if result["scenario"] == "S3_DIRECTIONAL_REACHABILITY":
                truth = result["truth_evaluation"]

                row.update(
                    {
                        "s3_oracle_positive_queries_smoke_only": truth[
                            "queried_oracle_positive_count"
                        ],
                        "s3_single_model_f1_smoke_only": truth["single_model"]["f1"],
                        "s3_robust_precision_smoke_only": truth["robust_tau_090"][
                            "precision"
                        ],
                        "s3_robust_f1_smoke_only": truth["robust_tau_090"]["f1"],
                        "s3_direction_recovered_smoke_only": truth[
                            "direction_recovered"
                        ],
                        "s3_label_swap_invariant_smoke_only": truth[
                            "numeric_true_profile_label_swap_invariant"
                        ],
                    }
                )

            if result["scenario"] == "S6_NO_CLUSTER_NULL":
                row["s6_false_robust_cf_claim_smoke_only"] = result["truth_evaluation"][
                    "false_robust_cf_claim_tau_090"
                ]

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_ROOT / "stage8d2_smoke_summary.csv",
        index=False,
    )

    gate_status = (
        "PASS_STAGE_8D2_COUNTERFACTUAL_SMOKE"
        if all(technical_checks.values())
        else "FAIL_STAGE_8D2_COUNTERFACTUAL_SMOKE"
    )

    report = {
        "status": gate_status,
        "scientific_interpretation_allowed": False,
        "scenario_results": summary_rows,
        "technical_checks": technical_checks,
    }

    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== NON-OFFICIAL CF SMOKE SUMMARY ===\n")

    print(summary.to_string(index=False))

    print("\n=== TECHNICAL GATE CHECKS ===\n")

    for name, passed in technical_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {gate_status}")

    if gate_status == "PASS_STAGE_8D2_COUNTERFACTUAL_SMOKE":
        print(
            "Counterfactual mechanics pass. "
            "Do NOT interpret smoke F1/precision/direction/false-claim values scientifically."
        )
        print(
            "Next after review: freeze official replicate indices, convergence-audit "
            "indices, and official-run execution plan."
        )
    else:
        print(
            "Do not proceed. Repair only failed technical mechanics; "
            "do not tune scientific thresholds or scenario geometry."
        )


if __name__ == "__main__":
    main()
