from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FRICTA_CONFIG_PATH = ROOT / "icdm" / "design" / "experiment_config.json"
CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
METHOD_FREEZE_PATH = ROOT / "cipher" / "design" / "counterfactual_method_freeze.json"

PRIMARY_MATRIX_PATH = ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
FINAL_LABELS_PATH = (
    ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
CERTAINTY_PATH = ROOT / "cipher" / "outputs" / "certainty" / "institution_certainty.csv"
FEATURE_GRID_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "preparation"
    / "feature_grid_summary.csv"
)
PLAUSIBILITY_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "counterfactuals"
    / "preparation"
    / "plausibility_thresholds.csv"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "counterfactuals" / "smoke"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"

PROFILE_1 = 1
PROFILE_2 = 2


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def align_labels(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    predicted_values = list(np.unique(predicted))
    reference_values = list(np.unique(reference))

    contingency = np.zeros(
        (len(predicted_values), len(reference_values)),
        dtype=int,
    )

    for i, p in enumerate(predicted_values):
        for j, r in enumerate(reference_values):
            contingency[i, j] = int(np.sum((predicted == p) & (reference == r)))

    rows, cols = linear_sum_assignment(-contingency)

    mapping = {
        int(predicted_values[row]): int(reference_values[col])
        for row, col in zip(rows, cols)
    }

    aligned = np.array(
        [mapping[int(value)] for value in predicted],
        dtype=int,
    )

    return aligned, mapping


def reconstruct_selected_model(
    X: np.ndarray,
    y_reference: np.ndarray,
    pca_threshold: float,
    seed: int,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=pca_threshold,
        svd_solver="full",
        random_state=seed,
    )
    Z = pca.fit_transform(X_scaled)

    ward = AgglomerativeClustering(
        n_clusters=2,
        linkage="ward",
    )
    raw_labels = ward.fit_predict(Z)

    aligned_labels, _ = align_labels(
        raw_labels,
        y_reference,
    )

    if not np.array_equal(aligned_labels, y_reference):
        raise ValueError(
            "Selected-model reconstruction no longer matches the frozen partition."
        )

    centroids = {
        profile: Z[aligned_labels == profile].mean(axis=0)
        for profile in sorted(np.unique(aligned_labels))
    }

    return scaler, pca, centroids


def predict_centroid_profiles(
    X_candidates: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    centroids: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    Z = pca.transform(scaler.transform(X_candidates))

    labels = sorted(centroids)
    centroid_matrix = np.vstack([centroids[label] for label in labels])

    distances = np.sqrt(
        ((Z[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    )

    nearest = np.argmin(distances, axis=1)

    predictions = np.array(
        [labels[idx] for idx in nearest],
        dtype=int,
    )

    return predictions, distances


def mean_knn_to_target(
    X_candidates: np.ndarray,
    X_target: np.ndarray,
    k: int,
) -> np.ndarray:
    # IMPORTANT: Stage 5A froze Euclidean distance in the original
    # normalized 13-D feature space. Sum squared differences first,
    # then take the square root.
    distances = np.sqrt(
        ((X_candidates[:, None, :] - X_target[None, :, :]) ** 2).sum(axis=2)
    )

    nearest = np.partition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return nearest.mean(axis=1)


def state_key(x: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(x.astype(float), 12))


def candidate_cost(
    x0: np.ndarray,
    x1: np.ndarray,
    denominators: np.ndarray,
    l0_penalty: float,
) -> tuple[float, float, int]:
    diff = np.abs(x1 - x0)
    changed = diff > 1e-12
    weighted_l1 = float(np.sum(diff / denominators))
    l0 = int(np.sum(changed))
    total = weighted_l1 + l0_penalty * l0
    return total, weighted_l1, l0


def target_margin_from_distances(
    distances: np.ndarray,
    current_profile: int,
    target_profile: int,
    label_order: list[int],
) -> np.ndarray:
    current_idx = label_order.index(current_profile)
    target_idx = label_order.index(target_profile)

    # Positive means the target centroid is closer.
    return distances[:, current_idx] - distances[:, target_idx]


def evaluate_candidates(
    candidates: np.ndarray,
    x0: np.ndarray,
    current_profile: int,
    target_profile: int,
    scaler: StandardScaler,
    pca: PCA,
    centroids: dict[int, np.ndarray],
    X_target: np.ndarray,
    plausibility_threshold: float,
    plausibility_k: int,
    denominators: np.ndarray,
    l0_penalty: float,
) -> pd.DataFrame:
    predictions, distances = predict_centroid_profiles(
        candidates,
        scaler,
        pca,
        centroids,
    )

    plausibility_distance = mean_knn_to_target(
        candidates,
        X_target,
        k=plausibility_k,
    )

    label_order = sorted(centroids)
    target_margin = target_margin_from_distances(
        distances,
        current_profile=current_profile,
        target_profile=target_profile,
        label_order=label_order,
    )

    rows = []

    for i, candidate in enumerate(candidates):
        total, weighted_l1, l0 = candidate_cost(
            x0,
            candidate,
            denominators,
            l0_penalty,
        )

        rows.append(
            {
                "candidate_index": i,
                "predicted_profile": int(predictions[i]),
                "valid_transition": bool(predictions[i] == target_profile),
                "plausibility_distance": float(plausibility_distance[i]),
                "plausible": bool(plausibility_distance[i] <= plausibility_threshold),
                "target_margin": float(target_margin[i]),
                "weighted_l1": weighted_l1,
                "l0": l0,
                "total_cost": total,
            }
        )

    return pd.DataFrame(rows)


def changed_feature_info(
    x0: np.ndarray,
    x1: np.ndarray,
    features: list[str],
) -> tuple[list[str], list[dict[str, float]]]:
    changes = []
    names = []

    for j, feature in enumerate(features):
        if abs(float(x1[j]) - float(x0[j])) <= 1e-12:
            continue

        names.append(feature)
        changes.append(
            {
                "feature": feature,
                "from": float(x0[j]),
                "to": float(x1[j]),
                "delta": float(x1[j] - x0[j]),
            }
        )

    return names, changes


def select_diverse_pareto(
    candidates: list[dict[str, Any]],
    max_results: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    # Keep the best candidate for each changed-feature set.
    best_by_feature_set: dict[tuple[str, ...], dict[str, Any]] = {}

    for item in candidates:
        key = tuple(sorted(item["changed_features"]))

        previous = best_by_feature_set.get(key)

        if previous is None or (
            item["total_cost"],
            item["plausibility_distance"],
        ) < (
            previous["total_cost"],
            previous["plausibility_distance"],
        ):
            best_by_feature_set[key] = item

    unique = list(best_by_feature_set.values())

    nondominated = []

    for i, a in enumerate(unique):
        dominated = False

        for j, b in enumerate(unique):
            if i == j:
                continue

            no_worse = (
                b["weighted_l1"] <= a["weighted_l1"] + 1e-12
                and b["l0"] <= a["l0"]
                and b["plausibility_distance"] <= a["plausibility_distance"] + 1e-12
            )

            strictly_better = (
                b["weighted_l1"] < a["weighted_l1"] - 1e-12
                or b["l0"] < a["l0"]
                or b["plausibility_distance"] < a["plausibility_distance"] - 1e-12
            )

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            nondominated.append(a)

    nondominated.sort(
        key=lambda item: (
            item["total_cost"],
            item["l0"],
            item["plausibility_distance"],
            tuple(item["changed_features"]),
        )
    )

    return nondominated[:max_results]


def beam_search(
    x0: np.ndarray,
    current_profile: int,
    target_profile: int,
    features: list[str],
    feature_grids: list[np.ndarray],
    scaler: StandardScaler,
    pca: PCA,
    centroids: dict[int, np.ndarray],
    X_target: np.ndarray,
    plausibility_threshold: float,
    plausibility_k: int,
    denominators: np.ndarray,
    l0_penalty: float,
    max_changed_features: int,
    beam_width: int,
    max_results: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Each beam state is (vector, tuple_of_changed_feature_indices).
    beam = [(x0.copy(), tuple())]
    seen = {state_key(x0)}

    valid_candidates: list[dict[str, Any]] = []
    evaluated_total = 0
    generated_total = 0

    label_order = sorted(centroids)

    for depth in range(1, max_changed_features + 1):
        expansions = []

        for state, changed_indices in beam:
            changed_set = set(changed_indices)

            for j in range(len(features)):
                if j in changed_set:
                    continue

                for value in feature_grids[j]:
                    if abs(float(value) - float(x0[j])) <= 1e-12:
                        continue

                    candidate = state.copy()
                    candidate[j] = float(value)

                    key = state_key(candidate)

                    if key in seen:
                        continue

                    seen.add(key)
                    generated_total += 1

                    new_changed = tuple(sorted(changed_indices + (j,)))

                    expansions.append((candidate, new_changed))

        if not expansions:
            break

        candidate_matrix = np.vstack([item[0] for item in expansions])

        evaluation = evaluate_candidates(
            candidate_matrix,
            x0=x0,
            current_profile=current_profile,
            target_profile=target_profile,
            scaler=scaler,
            pca=pca,
            centroids=centroids,
            X_target=X_target,
            plausibility_threshold=plausibility_threshold,
            plausibility_k=plausibility_k,
            denominators=denominators,
            l0_penalty=l0_penalty,
        )

        evaluated_total += len(evaluation)

        scored_states = []

        for idx, (candidate, changed_indices) in enumerate(expansions):
            row = evaluation.iloc[idx]

            changed_names, changes = changed_feature_info(
                x0,
                candidate,
                features,
            )

            if bool(row["valid_transition"]) and bool(row["plausible"]):
                valid_candidates.append(
                    {
                        "candidate": candidate.copy(),
                        "changed_features": changed_names,
                        "changes": changes,
                        "weighted_l1": float(row["weighted_l1"]),
                        "l0": int(row["l0"]),
                        "total_cost": float(row["total_cost"]),
                        "plausibility_distance": float(row["plausibility_distance"]),
                        "target_margin": float(row["target_margin"]),
                        "search_depth": depth,
                    }
                )

            # Beam ranking:
            # 1) candidates that already cross the target boundary;
            # 2) candidates closer to crossing (higher target margin);
            # 3) lower cost.
            transition_penalty = 0 if bool(row["valid_transition"]) else 1

            heuristic_key = (
                transition_penalty,
                -float(row["target_margin"]),
                float(row["total_cost"]),
                float(row["plausibility_distance"]),
                state_key(candidate),
            )

            scored_states.append(
                (
                    heuristic_key,
                    candidate,
                    changed_indices,
                )
            )

        scored_states.sort(key=lambda item: item[0])

        beam = [
            (candidate, changed_indices)
            for _, candidate, changed_indices in scored_states[:beam_width]
        ]

    selected = select_diverse_pareto(
        valid_candidates,
        max_results=max_results,
    )

    diagnostics = {
        "generated_total": generated_total,
        "evaluated_total": evaluated_total,
        "valid_plausible_candidates": len(valid_candidates),
        "selected_counterfactuals": len(selected),
    }

    return selected, diagnostics


def exact_search(
    x0: np.ndarray,
    current_profile: int,
    target_profile: int,
    features: list[str],
    feature_grids: list[np.ndarray],
    scaler: StandardScaler,
    pca: PCA,
    centroids: dict[int, np.ndarray],
    X_target: np.ndarray,
    plausibility_threshold: float,
    plausibility_k: int,
    denominators: np.ndarray,
    l0_penalty: float,
    max_changed_features: int,
    max_results: int,
    batch_size: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_candidates: list[dict[str, Any]] = []
    evaluated_total = 0

    for depth in range(1, max_changed_features + 1):
        for feature_subset in itertools.combinations(
            range(len(features)),
            depth,
        ):
            alternative_values = []

            for j in feature_subset:
                alternatives = [
                    float(v)
                    for v in feature_grids[j]
                    if abs(float(v) - float(x0[j])) > 1e-12
                ]

                if not alternatives:
                    alternative_values = []
                    break

                alternative_values.append(alternatives)

            if not alternative_values:
                continue

            product_iter = itertools.product(*alternative_values)

            batch_vectors = []
            batch_changes = []

            def flush_batch():
                nonlocal evaluated_total
                if not batch_vectors:
                    return

                matrix = np.vstack(batch_vectors)

                evaluation = evaluate_candidates(
                    matrix,
                    x0=x0,
                    current_profile=current_profile,
                    target_profile=target_profile,
                    scaler=scaler,
                    pca=pca,
                    centroids=centroids,
                    X_target=X_target,
                    plausibility_threshold=plausibility_threshold,
                    plausibility_k=plausibility_k,
                    denominators=denominators,
                    l0_penalty=l0_penalty,
                )

                evaluated_total += len(evaluation)

                for idx, candidate in enumerate(batch_vectors):
                    row = evaluation.iloc[idx]

                    if not (bool(row["valid_transition"]) and bool(row["plausible"])):
                        continue

                    changed_names, changes = changed_feature_info(
                        x0,
                        candidate,
                        features,
                    )

                    valid_candidates.append(
                        {
                            "candidate": candidate.copy(),
                            "changed_features": changed_names,
                            "changes": changes,
                            "weighted_l1": float(row["weighted_l1"]),
                            "l0": int(row["l0"]),
                            "total_cost": float(row["total_cost"]),
                            "plausibility_distance": float(
                                row["plausibility_distance"]
                            ),
                            "target_margin": float(row["target_margin"]),
                            "search_depth": depth,
                        }
                    )

                batch_vectors.clear()
                batch_changes.clear()

            for values in product_iter:
                candidate = x0.copy()

                for j, value in zip(
                    feature_subset,
                    values,
                ):
                    candidate[j] = value

                batch_vectors.append(candidate)
                batch_changes.append(feature_subset)

                if len(batch_vectors) >= batch_size:
                    flush_batch()

            flush_batch()

    selected = select_diverse_pareto(
        valid_candidates,
        max_results=max_results,
    )

    diagnostics = {
        "evaluated_total": evaluated_total,
        "valid_plausible_candidates": len(valid_candidates),
        "selected_counterfactuals": len(selected),
    }

    return selected, diagnostics


def select_smoke_institutions(
    certainty: pd.DataFrame,
) -> pd.DataFrame:
    certainty = certainty.copy()

    # Two lowest-certainty institutions overall.
    lowest = certainty.sort_values(
        [
            "reference_profile_probability",
            "family_consistency",
            "institution_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).head(2)

    # Two highest-certainty institutions from each frozen profile.
    cores = []

    for profile in (PROFILE_1, PROFILE_2):
        profile_rows = certainty[certainty["reference_profile"] == profile].copy()

        selected = profile_rows.sort_values(
            [
                "reference_profile_probability",
                "family_consistency",
                "consensus_gap",
                "institution_id",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        ).head(2)

        cores.append(selected)

    smoke = pd.concat(
        [lowest] + cores,
        ignore_index=True,
    ).drop_duplicates(subset=["institution_id"])

    # If a low-certainty case duplicates a selected core due to ties,
    # fill deterministically from the next-lowest rows.
    if len(smoke) < 6:
        remaining = certainty[
            ~certainty["institution_id"].isin(smoke["institution_id"])
        ].sort_values(
            [
                "reference_profile_probability",
                "family_consistency",
                "institution_id",
            ]
        )

        smoke = pd.concat(
            [
                smoke,
                remaining.head(6 - len(smoke)),
            ],
            ignore_index=True,
        )

    return smoke.head(6)


def serialize_counterfactual(
    institution_id: str,
    current_profile: int,
    target_profile: int,
    method: str,
    rank: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "institution_id": institution_id,
        "current_profile": current_profile,
        "target_profile": target_profile,
        "method": method,
        "rank": rank,
        "weighted_l1": result["weighted_l1"],
        "l0": result["l0"],
        "total_cost": result["total_cost"],
        "plausibility_distance": result["plausibility_distance"],
        "target_margin": result["target_margin"],
        "search_depth": result["search_depth"],
        "changed_features_json": json.dumps(result["changed_features"]),
        "changes_json": json.dumps(result["changes"]),
    }


def json_safe(value):
    """Recursively convert NumPy/Pandas scalar types into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    method_freeze = load_json(METHOD_FREEZE_PATH)

    if method_freeze.get("gate_status") != "PASS_STAGE_5A":
        raise ValueError("Stage 5A has not passed.")

    id_column = fricta_config["id_column"]
    features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[[id_column, "cluster_id"]]
    certainty = pd.read_csv(CERTAINTY_PATH)
    feature_grid = pd.read_csv(FEATURE_GRID_PATH)
    plausibility = pd.read_csv(PLAUSIBILITY_PATH)

    primary[id_column] = primary[id_column].astype(str)
    labels[id_column] = labels[id_column].astype(str)
    certainty["institution_id"] = certainty["institution_id"].astype(str)

    data = primary[[id_column] + features].merge(
        labels,
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError("Expected 81 institutions.")

    X = data[features].to_numpy(dtype=float)
    y = data["cluster_id"].astype(int).to_numpy()

    scaler, pca, centroids = reconstruct_selected_model(
        X,
        y_reference=y,
        pca_threshold=float(cipher_config["ensemble"]["pca_variance_threshold"]),
        seed=int(cipher_config["random_seed"]),
    )

    feature_grid = feature_grid.set_index("feature").loc[features].reset_index()

    feature_grids = [
        np.array(
            json.loads(row["grid_values_json"]),
            dtype=float,
        )
        for _, row in feature_grid.iterrows()
    ]

    denominators = feature_grid["cost_denominator"].to_numpy(dtype=float)

    max_changed = int(cipher_config["counterfactuals"]["maximum_changed_features"])
    beam_width = int(cipher_config["counterfactuals"]["beam_width"])
    max_results = int(
        cipher_config["counterfactuals"]["max_diverse_counterfactuals_per_institution"]
    )
    l0_penalty = float(cipher_config["counterfactuals"]["l0_penalty"])
    plausibility_k = int(cipher_config["counterfactuals"]["plausibility_neighbors"])

    threshold_by_profile = {
        int(row["profile"]): float(row["plausibility_threshold"])
        for _, row in plausibility.iterrows()
    }

    smoke = select_smoke_institutions(certainty)

    smoke.to_csv(
        OUTPUT_DIR / "smoke_institutions.csv",
        index=False,
    )

    data_by_id = data.set_index(id_column)

    result_rows = []
    diagnostic_rows = []

    # Exact-search audit on all six smoke institutions.
    # The smoke stage is an algorithmic correctness audit, not the
    # population-level feasibility/coverage test. Running exact search on all
    # six prevents cherry-picking and lets us verify both solution existence
    # and minimum cost.
    exact_audit_ids = smoke["institution_id"].astype(str).tolist()

    started_all = time.perf_counter()

    for _, smoke_row in smoke.iterrows():
        institution_id = str(smoke_row["institution_id"])

        source = data_by_id.loc[institution_id]

        current_profile = int(source["cluster_id"])
        target_profile = PROFILE_2 if current_profile == PROFILE_1 else PROFILE_1

        x0 = source[features].to_numpy(dtype=float)

        X_target = X[y == target_profile]

        plausibility_threshold = threshold_by_profile[target_profile]

        started = time.perf_counter()

        beam_results, beam_diag = beam_search(
            x0=x0,
            current_profile=current_profile,
            target_profile=target_profile,
            features=features,
            feature_grids=feature_grids,
            scaler=scaler,
            pca=pca,
            centroids=centroids,
            X_target=X_target,
            plausibility_threshold=plausibility_threshold,
            plausibility_k=plausibility_k,
            denominators=denominators,
            l0_penalty=l0_penalty,
            max_changed_features=max_changed,
            beam_width=beam_width,
            max_results=max_results,
        )

        beam_seconds = time.perf_counter() - started

        for rank, result in enumerate(
            beam_results,
            start=1,
        ):
            result_rows.append(
                serialize_counterfactual(
                    institution_id,
                    current_profile,
                    target_profile,
                    "BEAM",
                    rank,
                    result,
                )
            )

        exact_results = []
        exact_diag = {}
        exact_seconds = np.nan

        if institution_id in exact_audit_ids:
            started_exact = time.perf_counter()

            exact_results, exact_diag = exact_search(
                x0=x0,
                current_profile=current_profile,
                target_profile=target_profile,
                features=features,
                feature_grids=feature_grids,
                scaler=scaler,
                pca=pca,
                centroids=centroids,
                X_target=X_target,
                plausibility_threshold=plausibility_threshold,
                plausibility_k=plausibility_k,
                denominators=denominators,
                l0_penalty=l0_penalty,
                max_changed_features=max_changed,
                max_results=max_results,
            )

            exact_seconds = time.perf_counter() - started_exact

            for rank, result in enumerate(
                exact_results,
                start=1,
            ):
                result_rows.append(
                    serialize_counterfactual(
                        institution_id,
                        current_profile,
                        target_profile,
                        "EXACT",
                        rank,
                        result,
                    )
                )

        beam_best = beam_results[0]["total_cost"] if beam_results else np.nan
        exact_best = exact_results[0]["total_cost"] if exact_results else np.nan

        beam_exists = bool(len(beam_results) > 0)
        exact_exists = bool(len(exact_results) > 0)
        existence_matches_exact = bool(beam_exists == exact_exists)

        if beam_exists and exact_exists:
            optimality_gap = float(beam_best - exact_best)
            beam_matches_exact_best = bool(abs(optimality_gap) <= 1e-12)
        elif not beam_exists and not exact_exists:
            optimality_gap = np.nan
            beam_matches_exact_best = True
        else:
            optimality_gap = np.nan
            beam_matches_exact_best = False

        diagnostic_rows.append(
            {
                "institution_id": institution_id,
                "certainty_class": smoke_row["certainty_class"],
                "reference_profile_probability": float(
                    smoke_row["reference_profile_probability"]
                ),
                "current_profile": current_profile,
                "target_profile": target_profile,
                "beam_counterfactuals": len(beam_results),
                "beam_evaluated_candidates": int(beam_diag["evaluated_total"]),
                "beam_valid_plausible_candidates": int(
                    beam_diag["valid_plausible_candidates"]
                ),
                "beam_seconds": beam_seconds,
                "exact_audited": (institution_id in exact_audit_ids),
                "exact_counterfactuals": len(exact_results),
                "exact_evaluated_candidates": (
                    int(exact_diag["evaluated_total"]) if exact_diag else 0
                ),
                "exact_valid_plausible_candidates": (
                    int(exact_diag["valid_plausible_candidates"]) if exact_diag else 0
                ),
                "exact_seconds": exact_seconds,
                "beam_best_cost": beam_best,
                "exact_best_cost": exact_best,
                "optimality_gap": optimality_gap,
                "beam_solution_exists": beam_exists,
                "exact_solution_exists": exact_exists,
                "existence_matches_exact": existence_matches_exact,
                "beam_matches_exact_best": (beam_matches_exact_best),
            }
        )

    elapsed_all = time.perf_counter() - started_all

    result_columns = [
        "institution_id",
        "current_profile",
        "target_profile",
        "method",
        "rank",
        "weighted_l1",
        "l0",
        "total_cost",
        "plausibility_distance",
        "target_margin",
        "search_depth",
        "changed_features_json",
        "changes_json",
    ]

    diagnostic_columns = [
        "institution_id",
        "certainty_class",
        "reference_profile_probability",
        "current_profile",
        "target_profile",
        "beam_counterfactuals",
        "beam_evaluated_candidates",
        "beam_valid_plausible_candidates",
        "beam_seconds",
        "exact_audited",
        "exact_counterfactuals",
        "exact_evaluated_candidates",
        "exact_valid_plausible_candidates",
        "exact_seconds",
        "beam_best_cost",
        "exact_best_cost",
        "optimality_gap",
        "beam_solution_exists",
        "exact_solution_exists",
        "existence_matches_exact",
        "beam_matches_exact_best",
    ]

    # Preserve the expected schema even when no valid counterfactual is found.
    # This lets the smoke test fail scientifically/diagnostically instead of
    # crashing in the reporting layer with KeyError: 'method'.
    results = pd.DataFrame(
        result_rows,
        columns=result_columns,
    )
    diagnostics = pd.DataFrame(
        diagnostic_rows,
        columns=diagnostic_columns,
    )

    results.to_csv(
        OUTPUT_DIR / "smoke_counterfactuals.csv",
        index=False,
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "smoke_search_diagnostics.csv",
        index=False,
    )

    beam_only = results[results["method"] == "BEAM"].copy()

    beam_coverage = (diagnostics["beam_counterfactuals"] > 0).mean()

    exact_audited = diagnostics[diagnostics["exact_audited"]].copy()

    exact_coverage = float((exact_audited["exact_counterfactuals"] > 0).mean())

    solution_cases = exact_audited[
        exact_audited["exact_solution_exists"].astype(bool)
    ].copy()

    euclidean_self_test = float(
        mean_knn_to_target(
            np.array([[0.0, 0.0]]),
            np.array([[3.0, 4.0]]),
            k=1,
        )[0]
    )

    checks = {
        "six_smoke_institutions": (len(smoke) == 6),
        "both_transition_directions_tested": (
            set(
                zip(
                    diagnostics["current_profile"],
                    diagnostics["target_profile"],
                )
            )
            == {
                (1, 2),
                (2, 1),
            }
        ),
        "plausibility_metric_is_frozen_euclidean": (
            abs(euclidean_self_test - 5.0) <= 1e-12
        ),
        "all_six_exact_audited": (
            len(exact_audited) == 6 and bool(exact_audited["exact_audited"].all())
        ),
        "beam_exact_solution_existence_matches_all_six": bool(
            exact_audited["existence_matches_exact"].astype(bool).all()
        ),
        "every_beam_result_within_four_features": (
            bool((beam_only["l0"] <= max_changed).all()) if len(beam_only) else True
        ),
        "every_beam_result_plausible": (
            bool(
                (
                    beam_only["plausibility_distance"]
                    <= beam_only["target_profile"].map(threshold_by_profile) + 1e-12
                ).all()
            )
            if len(beam_only)
            else True
        ),
        "beam_matches_exact_best_on_every_solvable_smoke_case": (
            bool(solution_cases["beam_matches_exact_best"].astype(bool).all())
            if len(solution_cases)
            else True
        ),
    }

    report = {
        "smoke_institutions": smoke[
            [
                "institution_id",
                "reference_profile",
                "reference_profile_probability",
                "family_consistency",
                "certainty_class",
            ]
        ].to_dict(orient="records"),
        "exact_audit_ids": (exact_audit_ids),
        "beam_coverage_descriptive": float(beam_coverage),
        "exact_coverage_descriptive": float(exact_coverage),
        "coverage_gate_note": (
            "Smoke coverage is descriptive only. The frozen >=70% "
            "counterfactual coverage criterion is evaluated on the full "
            "81-institution official run."
        ),
        "elapsed_seconds": float(elapsed_all),
        "checks": checks,
        "gate_status": (
            "PASS_STAGE_5B_SMOKE" if all(checks.values()) else "FAIL_STAGE_5B_SMOKE"
        ),
    }

    (OUTPUT_DIR / "smoke_report.json").write_text(
        json.dumps(
            json_safe(report),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 5B — COUNTERFACTUAL SMOKE SEARCH ===\n")

    print("Smoke institutions:\n")
    print(
        smoke[
            [
                "institution_id",
                "reference_profile",
                "reference_profile_probability",
                "family_consistency",
                "certainty_class",
            ]
        ].to_string(index=False)
    )

    print("\n=== SEARCH DIAGNOSTICS ===\n")
    print(diagnostics.to_string(index=False))

    print("\n=== BEAM COUNTERFACTUALS ===\n")

    if len(beam_only):
        print(
            beam_only[
                [
                    "institution_id",
                    "current_profile",
                    "target_profile",
                    "rank",
                    "weighted_l1",
                    "l0",
                    "total_cost",
                    "plausibility_distance",
                    "target_margin",
                    "changed_features_json",
                    "changes_json",
                ]
            ].to_string(index=False)
        )
    else:
        print("No beam counterfactuals found.")

    print("\n=== SMOKE FEASIBILITY — DESCRIPTIVE ONLY ===\n")
    print(
        "Beam solution coverage:",
        f"{beam_coverage:.4f}",
    )
    print(
        "Exact solution coverage:",
        f"{exact_coverage:.4f}",
    )
    print(
        "NOTE: The frozen >=70% coverage gate is reserved for the "
        "official 81-institution run."
    )

    print("\n=== GATE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {report['gate_status']}")

    if report["gate_status"] == "PASS_STAGE_5B_SMOKE":
        print(
            "Beam search passed algorithmic smoke validation against "
            "exact search. Official Stage 5 coverage may be evaluated after review."
        )


if __name__ == "__main__":
    main()
