from __future__ import annotations

import argparse
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
METHOD_FREEZE_V2_PATH = (
    ROOT / "cipher" / "design" / "counterfactual_method_freeze_v2.json"
)

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
ACTIONABILITY_PATH = ROOT / "cipher" / "design" / "actionability_manifest.csv"

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "counterfactuals" / "official_exact"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"

PROFILE_1 = 1
PROFILE_2 = 2


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value):
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


def align_labels(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    predicted_values = list(np.unique(predicted))
    reference_values = list(np.unique(reference))

    contingency = np.zeros(
        (len(predicted_values), len(reference_values)),
        dtype=int,
    )

    for i, pred_value in enumerate(predicted_values):
        for j, ref_value in enumerate(reference_values):
            contingency[i, j] = int(
                np.sum((predicted == pred_value) & (reference == ref_value))
            )

    rows, cols = linear_sum_assignment(-contingency)

    mapping = {
        int(predicted_values[row]): int(reference_values[col])
        for row, col in zip(rows, cols)
    }

    return np.array(
        [mapping[int(value)] for value in predicted],
        dtype=int,
    )


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
    aligned_labels = align_labels(raw_labels, y_reference)

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
    # Frozen Stage 5A metric: Euclidean distance in original normalized 13-D space.
    distances = np.sqrt(
        ((X_candidates[:, None, :] - X_target[None, :, :]) ** 2).sum(axis=2)
    )

    nearest = np.partition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return nearest.mean(axis=1)


def candidate_cost(
    x0: np.ndarray,
    x1: np.ndarray,
    denominators: np.ndarray,
    l0_penalty: float,
) -> tuple[float, float, int]:
    diff = np.abs(x1 - x0)
    changed = diff > 1e-12
    weighted_l1 = float(np.sum(diff / denominators))
    l0 = int(changed.sum())
    total_cost = weighted_l1 + l0_penalty * l0
    return total_cost, weighted_l1, l0


def change_payload(
    x0: np.ndarray,
    x1: np.ndarray,
    features: list[str],
    actionability_lookup: dict[str, dict[str, str]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    changed_features = []
    signed_items = []
    changes = []

    for j, feature in enumerate(features):
        old = float(x0[j])
        new = float(x1[j])

        if abs(new - old) <= 1e-12:
            continue

        direction = "UP" if new > old else "DOWN"
        symbol = "↑" if direction == "UP" else "↓"

        changed_features.append(feature)
        signed_items.append(f"{feature}{symbol}")

        meta = actionability_lookup[feature]

        changes.append(
            {
                "feature": feature,
                "from": old,
                "to": new,
                "delta": float(new - old),
                "signed_direction": direction,
                "actionability_class": meta["actionability_class"],
                "realistic_improvement_direction": meta[
                    "realistic_improvement_direction"
                ],
            }
        )

    return changed_features, signed_items, changes


def expected_exact_candidate_count(
    x0: np.ndarray,
    feature_grids: list[np.ndarray],
    max_changed: int,
) -> int:
    alternative_counts = []

    for j, grid in enumerate(feature_grids):
        count = sum(abs(float(value) - float(x0[j])) > 1e-12 for value in grid)
        alternative_counts.append(int(count))

    total = 0

    for depth in range(1, max_changed + 1):
        for subset in itertools.combinations(
            range(len(feature_grids)),
            depth,
        ):
            product = 1
            for j in subset:
                product *= alternative_counts[j]
            total += product

    return int(total)


def representative_better(
    candidate: dict[str, Any],
    previous: dict[str, Any] | None,
) -> bool:
    if previous is None:
        return True

    candidate_key = (
        candidate["total_cost"],
        candidate["plausibility_distance"],
        -candidate["target_margin"],
        tuple(candidate["changed_features"]),
    )

    previous_key = (
        previous["total_cost"],
        previous["plausibility_distance"],
        -previous["target_margin"],
        tuple(previous["changed_features"]),
    )

    return candidate_key < previous_key


def select_diverse_pareto(
    representatives: list[dict[str, Any]],
    max_results: int,
) -> list[dict[str, Any]]:
    if not representatives:
        return []

    nondominated = []

    for i, a in enumerate(representatives):
        dominated = False

        for j, b in enumerate(representatives):
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
            -item["target_margin"],
            tuple(item["changed_features"]),
        )
    )

    return nondominated[:max_results]


def exact_search_one(
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
    max_changed: int,
    max_results: int,
    actionability_lookup: dict[str, dict[str, str]],
    batch_size: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = sorted(centroids)
    current_idx = labels.index(current_profile)
    target_idx = labels.index(target_profile)

    best_by_feature_set: dict[tuple[str, ...], dict[str, Any]] = {}

    depth_stats = {
        depth: {
            "evaluated": 0,
            "transition_count": 0,
            "plausible_count": 0,
            "valid_plausible_count": 0,
        }
        for depth in range(1, max_changed + 1)
    }

    global_best: dict[str, Any] | None = None

    for depth in range(1, max_changed + 1):
        for subset in itertools.combinations(
            range(len(features)),
            depth,
        ):
            alternative_values = []

            for j in subset:
                alternatives = [
                    float(value)
                    for value in feature_grids[j]
                    if abs(float(value) - float(x0[j])) > 1e-12
                ]

                if not alternatives:
                    alternative_values = []
                    break

                alternative_values.append(alternatives)

            if not alternative_values:
                continue

            batch_vectors = []

            def flush_batch():
                nonlocal global_best

                if not batch_vectors:
                    return

                M = np.vstack(batch_vectors)

                predictions, distances = predict_centroid_profiles(
                    M,
                    scaler,
                    pca,
                    centroids,
                )

                plausibility_distance = mean_knn_to_target(
                    M,
                    X_target,
                    plausibility_k,
                )

                target_margin = distances[:, current_idx] - distances[:, target_idx]

                transition_mask = predictions == target_profile
                plausible_mask = plausibility_distance <= plausibility_threshold
                valid_mask = transition_mask & plausible_mask

                depth_stats[depth]["evaluated"] += len(M)
                depth_stats[depth]["transition_count"] += int(transition_mask.sum())
                depth_stats[depth]["plausible_count"] += int(plausible_mask.sum())
                depth_stats[depth]["valid_plausible_count"] += int(valid_mask.sum())

                valid_indices = np.where(valid_mask)[0]

                for idx in valid_indices:
                    candidate = M[idx]

                    total_cost, weighted_l1, l0 = candidate_cost(
                        x0,
                        candidate,
                        denominators,
                        l0_penalty,
                    )

                    changed_features, signed_items, changes = change_payload(
                        x0,
                        candidate,
                        features,
                        actionability_lookup,
                    )

                    result = {
                        "candidate": candidate.copy(),
                        "changed_features": changed_features,
                        "signed_items": signed_items,
                        "changes": changes,
                        "weighted_l1": weighted_l1,
                        "l0": l0,
                        "total_cost": total_cost,
                        "plausibility_distance": float(plausibility_distance[idx]),
                        "target_margin": float(target_margin[idx]),
                        "search_depth": depth,
                    }

                    feature_key = tuple(sorted(changed_features))

                    if representative_better(
                        result,
                        best_by_feature_set.get(feature_key),
                    ):
                        best_by_feature_set[feature_key] = result

                    if global_best is None or (
                        result["total_cost"],
                        result["plausibility_distance"],
                        -result["target_margin"],
                        tuple(result["changed_features"]),
                    ) < (
                        global_best["total_cost"],
                        global_best["plausibility_distance"],
                        -global_best["target_margin"],
                        tuple(global_best["changed_features"]),
                    ):
                        global_best = result

                batch_vectors.clear()

            for values in itertools.product(*alternative_values):
                candidate = x0.copy()

                for j, value in zip(subset, values):
                    candidate[j] = value

                batch_vectors.append(candidate)

                if len(batch_vectors) >= batch_size:
                    flush_batch()

            flush_batch()

    representatives = list(best_by_feature_set.values())

    selected = select_diverse_pareto(
        representatives,
        max_results=max_results,
    )

    if global_best is not None:
        selected_keys = {tuple(np.round(item["candidate"], 12)) for item in selected}
        global_key = tuple(np.round(global_best["candidate"], 12))

        # Guarantee that rank 1 is the exact global minimum-cost valid candidate.
        if global_key not in selected_keys:
            selected = [global_best] + selected[: max(0, max_results - 1)]
        else:
            selected.sort(
                key=lambda item: (
                    0 if tuple(np.round(item["candidate"], 12)) == global_key else 1,
                    item["total_cost"],
                    item["l0"],
                    item["plausibility_distance"],
                )
            )

    total_evaluated = sum(row["evaluated"] for row in depth_stats.values())
    total_transitions = sum(row["transition_count"] for row in depth_stats.values())
    total_plausible = sum(row["plausible_count"] for row in depth_stats.values())
    total_valid = sum(row["valid_plausible_count"] for row in depth_stats.values())

    diagnostics = {
        "evaluated_total": int(total_evaluated),
        "transition_candidates": int(total_transitions),
        "plausible_candidates": int(total_plausible),
        "valid_plausible_candidates": int(total_valid),
        "representative_feature_sets": int(len(representatives)),
        "selected_counterfactuals": int(len(selected)),
        "global_best_cost": (
            float(global_best["total_cost"]) if global_best is not None else np.nan
        ),
        "global_best_l0": (
            int(global_best["l0"]) if global_best is not None else np.nan
        ),
        "depth_stats": depth_stats,
    }

    return selected, diagnostics


def serialize_result(
    institution_id: str,
    reference_profile: int,
    target_profile: int,
    certainty_class: str,
    reference_probability: float,
    family_consistency: float,
    rank: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "institution_id": institution_id,
        "reference_profile": reference_profile,
        "target_profile": target_profile,
        "certainty_class": certainty_class,
        "reference_profile_probability": reference_probability,
        "family_consistency": family_consistency,
        "rank": rank,
        "is_exact_global_minimum": rank == 1,
        "weighted_l1": result["weighted_l1"],
        "l0": result["l0"],
        "total_cost": result["total_cost"],
        "plausibility_distance": result["plausibility_distance"],
        "target_margin": result["target_margin"],
        "search_depth": result["search_depth"],
        "changed_features_json": json.dumps(
            result["changed_features"],
            ensure_ascii=False,
        ),
        "signed_items_json": json.dumps(
            result["signed_items"],
            ensure_ascii=False,
        ),
        "changes_json": json.dumps(
            result["changes"],
            ensure_ascii=False,
        ),
    }


def save_checkpoints(
    result_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(result_rows).to_csv(
        OUTPUT_DIR / "checkpoint_counterfactuals.csv",
        index=False,
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT_DIR / "checkpoint_diagnostics.csv",
        index=False,
    )


def load_checkpoints() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    results_path = OUTPUT_DIR / "checkpoint_counterfactuals.csv"
    diagnostics_path = OUTPUT_DIR / "checkpoint_diagnostics.csv"

    if not diagnostics_path.exists():
        return [], [], set()

    diagnostic_df = pd.read_csv(diagnostics_path)
    diagnostic_rows = diagnostic_df.to_dict(orient="records")

    if results_path.exists() and results_path.stat().st_size > 0:
        try:
            result_df = pd.read_csv(results_path)
            result_rows = result_df.to_dict(orient="records")
        except pd.errors.EmptyDataError:
            result_rows = []
    else:
        result_rows = []

    completed = set(diagnostic_df["institution_id"].astype(str))

    return result_rows, diagnostic_rows, completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint files if a previous run was interrupted.",
    )
    args = parser.parse_args()

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
    method_v2 = load_json(METHOD_FREEZE_V2_PATH)

    if method_v2.get("gate_status") != "PASS_STAGE_5B_METHOD_AMENDMENT":
        raise ValueError("Stage 5B exact-search method amendment has not passed.")

    if (
        method_v2.get("search_space", {}).get("official_search_optimizer")
        != "EXACT_EXHAUSTIVE_ENUMERATION"
    ):
        raise ValueError(
            "Official counterfactual optimizer is not frozen to exact enumeration."
        )

    id_column = fricta_config["id_column"]
    features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[[id_column, "cluster_id"]]
    certainty = pd.read_csv(CERTAINTY_PATH)
    feature_grid = pd.read_csv(FEATURE_GRID_PATH)
    plausibility = pd.read_csv(PLAUSIBILITY_PATH)
    actionability = pd.read_csv(ACTIONABILITY_PATH)

    primary[id_column] = primary[id_column].astype(str)
    labels[id_column] = labels[id_column].astype(str)
    certainty["institution_id"] = certainty["institution_id"].astype(str)

    data = primary[[id_column] + features].merge(
        labels,
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    data = data.merge(
        certainty[
            [
                "institution_id",
                "certainty_class",
                "reference_profile_probability",
                "family_consistency",
            ]
        ],
        left_on=id_column,
        right_on="institution_id",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 institutions; found {len(data)}.")

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

    actionability_lookup = (
        actionability.set_index("feature")[
            [
                "actionability_class",
                "realistic_improvement_direction",
            ]
        ]
        .astype(str)
        .to_dict(orient="index")
    )

    thresholds = {
        int(row["profile"]): float(row["plausibility_threshold"])
        for _, row in plausibility.iterrows()
    }

    max_changed = int(cipher_config["counterfactuals"]["maximum_changed_features"])
    max_results = int(
        cipher_config["counterfactuals"]["max_diverse_counterfactuals_per_institution"]
    )
    l0_penalty = float(cipher_config["counterfactuals"]["l0_penalty"])
    plausibility_k = int(cipher_config["counterfactuals"]["plausibility_neighbors"])

    if args.resume:
        result_rows, diagnostic_rows, completed_ids = load_checkpoints()
        print(f"Resuming: {len(completed_ids)} institutions already completed.")
    else:
        existing_final = OUTPUT_DIR / "official_counterfactuals.csv"
        if existing_final.exists():
            raise FileExistsError(
                "Official Stage 5C outputs already exist. "
                "Do not overwrite a completed official run."
            )
        result_rows = []
        diagnostic_rows = []
        completed_ids = set()

    started_all = time.perf_counter()

    for row_number, row in data.iterrows():
        institution_id = str(row[id_column])

        if institution_id in completed_ids:
            continue

        current_profile = int(row["cluster_id"])
        target_profile = PROFILE_2 if current_profile == PROFILE_1 else PROFILE_1

        x0 = row[features].to_numpy(dtype=float)

        X_target = X[y == target_profile]

        expected_count = expected_exact_candidate_count(
            x0,
            feature_grids,
            max_changed,
        )

        started = time.perf_counter()

        selected, diag = exact_search_one(
            x0=x0,
            current_profile=current_profile,
            target_profile=target_profile,
            features=features,
            feature_grids=feature_grids,
            scaler=scaler,
            pca=pca,
            centroids=centroids,
            X_target=X_target,
            plausibility_threshold=thresholds[target_profile],
            plausibility_k=plausibility_k,
            denominators=denominators,
            l0_penalty=l0_penalty,
            max_changed=max_changed,
            max_results=max_results,
            actionability_lookup=actionability_lookup,
        )

        elapsed = time.perf_counter() - started

        if diag["evaluated_total"] != expected_count:
            raise ValueError(
                f"{institution_id}: exact enumeration mismatch. "
                f"Expected {expected_count}, evaluated {diag['evaluated_total']}."
            )

        for rank, result in enumerate(
            selected,
            start=1,
        ):
            result_rows.append(
                serialize_result(
                    institution_id=institution_id,
                    reference_profile=current_profile,
                    target_profile=target_profile,
                    certainty_class=str(row["certainty_class"]),
                    reference_probability=float(row["reference_profile_probability"]),
                    family_consistency=float(row["family_consistency"]),
                    rank=rank,
                    result=result,
                )
            )

        if diag["valid_plausible_candidates"] > 0:
            failure_mode = "SOLVABLE"
        elif diag["transition_candidates"] > 0:
            failure_mode = "TRANSITIONS_FAIL_PLAUSIBILITY"
        else:
            failure_mode = "NO_TRANSITION_WITHIN_4_FEATURES"

        depth_stats = diag["depth_stats"]

        diagnostic_rows.append(
            {
                "institution_id": institution_id,
                "reference_profile": current_profile,
                "target_profile": target_profile,
                "certainty_class": str(row["certainty_class"]),
                "reference_profile_probability": float(
                    row["reference_profile_probability"]
                ),
                "family_consistency": float(row["family_consistency"]),
                "expected_exact_candidates": expected_count,
                "evaluated_exact_candidates": diag["evaluated_total"],
                "transition_candidates": diag["transition_candidates"],
                "plausible_candidates": diag["plausible_candidates"],
                "valid_plausible_candidates": diag["valid_plausible_candidates"],
                "selected_counterfactuals": diag["selected_counterfactuals"],
                "global_best_cost": diag["global_best_cost"],
                "global_best_l0": diag["global_best_l0"],
                "depth1_valid": depth_stats[1]["valid_plausible_count"],
                "depth2_valid": depth_stats[2]["valid_plausible_count"],
                "depth3_valid": depth_stats[3]["valid_plausible_count"],
                "depth4_valid": depth_stats[4]["valid_plausible_count"],
                "failure_mode": failure_mode,
                "elapsed_seconds": elapsed,
            }
        )

        save_checkpoints(
            result_rows,
            diagnostic_rows,
        )

        completed_now = len({str(item["institution_id"]) for item in diagnostic_rows})

        print(
            f"[{completed_now:02d}/81] {institution_id} "
            f"{current_profile}->{target_profile} "
            f"{failure_mode} "
            f"valid={diag['valid_plausible_candidates']} "
            f"best_cost={diag['global_best_cost']} "
            f"seconds={elapsed:.2f}",
            flush=True,
        )

    elapsed_all = time.perf_counter() - started_all

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    if len(diagnostics) != 81:
        raise ValueError(
            f"Expected 81 completed diagnostics; found {len(diagnostics)}."
        )

    results.to_csv(
        OUTPUT_DIR / "official_counterfactuals.csv",
        index=False,
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "institution_counterfactual_diagnostics.csv",
        index=False,
    )

    best = results[results["rank"] == 1].copy()

    coverage = float((diagnostics["failure_mode"] == "SOLVABLE").mean())

    coverage_rows = []

    coverage_rows.append(
        {
            "group_type": "OVERALL",
            "group_value": "ALL",
            "n": len(diagnostics),
            "solvable": int((diagnostics["failure_mode"] == "SOLVABLE").sum()),
            "coverage": coverage,
        }
    )

    for profile, group in diagnostics.groupby(
        "reference_profile",
        sort=True,
    ):
        solvable = int((group["failure_mode"] == "SOLVABLE").sum())
        coverage_rows.append(
            {
                "group_type": "REFERENCE_PROFILE",
                "group_value": str(profile),
                "n": len(group),
                "solvable": solvable,
                "coverage": solvable / len(group),
            }
        )

    for certainty_class, group in diagnostics.groupby(
        "certainty_class",
        sort=True,
    ):
        solvable = int((group["failure_mode"] == "SOLVABLE").sum())
        coverage_rows.append(
            {
                "group_type": "CERTAINTY_CLASS",
                "group_value": str(certainty_class),
                "n": len(group),
                "solvable": solvable,
                "coverage": solvable / len(group),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(
        OUTPUT_DIR / "coverage_summary.csv",
        index=False,
    )

    cost_rows = []

    if len(best):
        for grouping, column in [
            ("OVERALL", None),
            ("REFERENCE_PROFILE", "reference_profile"),
            ("CERTAINTY_CLASS", "certainty_class"),
        ]:
            if column is None:
                groups = [("ALL", best)]
            else:
                groups = best.groupby(
                    column,
                    sort=True,
                )

            for value, group in groups:
                cost_rows.append(
                    {
                        "group_type": grouping,
                        "group_value": str(value),
                        "n_solved": len(group),
                        "median_exact_min_cost": float(group["total_cost"].median()),
                        "q025_exact_min_cost": float(
                            group["total_cost"].quantile(0.025)
                        ),
                        "q975_exact_min_cost": float(
                            group["total_cost"].quantile(0.975)
                        ),
                        "median_changed_features": float(group["l0"].median()),
                    }
                )

    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(
        OUTPUT_DIR / "minimum_cost_summary.csv",
        index=False,
    )

    failure_counts = diagnostics["failure_mode"].value_counts().to_dict()

    exact_count_match = bool(
        (
            diagnostics["expected_exact_candidates"]
            == diagnostics["evaluated_exact_candidates"]
        ).all()
    )

    selected_within_four = (
        bool((results["l0"] <= max_changed).all()) if len(results) else True
    )

    selected_plausible = (
        bool(
            (
                results["plausibility_distance"]
                <= results["target_profile"].map(thresholds) + 1e-12
            ).all()
        )
        if len(results)
        else True
    )

    rank1_one_per_solved = len(best) == int(
        (diagnostics["failure_mode"] == "SOLVABLE").sum()
    )

    checks = {
        "method_amendment_passed": (
            method_v2.get("gate_status") == "PASS_STAGE_5B_METHOD_AMENDMENT"
        ),
        "81_institutions_completed": (len(diagnostics) == 81),
        "exact_candidate_count_matches_combinatorial_space": (exact_count_match),
        "all_selected_counterfactuals_within_4_features": (selected_within_four),
        "all_selected_counterfactuals_pass_frozen_plausibility": (selected_plausible),
        "one_exact_global_minimum_per_solved_institution": (rank1_one_per_solved),
        "official_single_model_coverage_at_least_70_percent": (coverage >= 0.70),
    }

    report = {
        "optimizer": "EXACT_EXHAUSTIVE_ENUMERATION",
        "institutions": 81,
        "coverage": coverage,
        "solvable_institutions": int((diagnostics["failure_mode"] == "SOLVABLE").sum()),
        "failure_counts": {
            str(key): int(value) for key, value in failure_counts.items()
        },
        "elapsed_seconds_current_session": float(elapsed_all),
        "checks": checks,
        "gate_status": ("PASS_STAGE_5C" if all(checks.values()) else "FAIL_STAGE_5C"),
        "interpretation": (
            "Counterfactuals are exact minimum-cost diagnostic profile-transition "
            "explanations within the frozen discrete search space. They are not "
            "causal interventions or treatment recommendations."
        ),
    }

    (OUTPUT_DIR / "stage5c_report.json").write_text(
        json.dumps(
            json_safe(report),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 5C — OFFICIAL EXACT COUNTERFACTUALS ===\n")

    print(
        "Institutions completed:",
        len(diagnostics),
    )
    print(
        "Solvable institutions:",
        report["solvable_institutions"],
    )
    print(
        "Official single-model coverage:",
        f"{coverage:.4f}",
    )
    print(
        "Failure modes:",
        report["failure_counts"],
    )

    print("\n=== COVERAGE SUMMARY ===\n")
    print(coverage_df.to_string(index=False))

    print("\n=== EXACT MINIMUM-COST SUMMARY ===\n")
    if len(cost_df):
        print(cost_df.to_string(index=False))
    else:
        print("No solvable institutions.")

    print("\n=== 15 LOWEST EXACT MINIMUM COSTS ===\n")
    if len(best):
        print(
            best.sort_values(
                [
                    "total_cost",
                    "institution_id",
                ]
            )
            .head(15)[
                [
                    "institution_id",
                    "reference_profile",
                    "target_profile",
                    "certainty_class",
                    "total_cost",
                    "l0",
                    "plausibility_distance",
                    "signed_items_json",
                ]
            ]
            .to_string(index=False)
        )
    else:
        print("None.")

    print("\n=== 15 HIGHEST EXACT MINIMUM COSTS AMONG SOLVABLE ===\n")
    if len(best):
        print(
            best.sort_values(
                [
                    "total_cost",
                    "institution_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(15)[
                [
                    "institution_id",
                    "reference_profile",
                    "target_profile",
                    "certainty_class",
                    "total_cost",
                    "l0",
                    "plausibility_distance",
                    "signed_items_json",
                ]
            ]
            .to_string(index=False)
        )
    else:
        print("None.")

    print("\n=== GATE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {report['gate_status']}")

    if report["gate_status"] == "PASS_STAGE_5C":
        print(
            "Stage 5 single-model exact counterfactuals are complete. "
            "Stage 6 ensemble-robust validation may begin after review."
        )
    else:
        print("Do not proceed to Stage 6 until the failed Stage 5C gate is reviewed.")


if __name__ == "__main__":
    main()
