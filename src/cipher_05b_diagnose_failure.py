from __future__ import annotations

import itertools
import json
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
SMOKE_INSTITUTIONS_PATH = (
    ROOT / "cipher" / "outputs" / "counterfactuals" / "smoke" / "smoke_institutions.csv"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "counterfactuals" / "smoke_diagnostic"

PROFILE_1 = 1
PROFILE_2 = 2


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def align_labels(predicted: np.ndarray, reference: np.ndarray):
    pvals = list(np.unique(predicted))
    rvals = list(np.unique(reference))
    table = np.zeros((len(pvals), len(rvals)), dtype=int)

    for i, p in enumerate(pvals):
        for j, r in enumerate(rvals):
            table[i, j] = int(np.sum((predicted == p) & (reference == r)))

    rows, cols = linear_sum_assignment(-table)
    mapping = {int(pvals[r]): int(rvals[c]) for r, c in zip(rows, cols)}
    aligned = np.array([mapping[int(v)] for v in predicted], dtype=int)
    return aligned


def reconstruct_selected_model(
    X: np.ndarray,
    y_reference: np.ndarray,
    pca_threshold: float,
    seed: int,
):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(
        n_components=pca_threshold,
        svd_solver="full",
        random_state=seed,
    )
    Z = pca.fit_transform(Xs)

    ward = AgglomerativeClustering(n_clusters=2, linkage="ward")
    raw = ward.fit_predict(Z)
    aligned = align_labels(raw, y_reference)

    if not np.array_equal(aligned, y_reference):
        raise ValueError(
            "Selected Ward reconstruction no longer matches frozen labels."
        )

    centroids = {
        profile: Z[aligned == profile].mean(axis=0)
        for profile in sorted(np.unique(aligned))
    }
    return scaler, pca, centroids


def predict_with_margin(
    Xc: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    centroids: dict[int, np.ndarray],
    current_profile: int,
    target_profile: int,
):
    Z = pca.transform(scaler.transform(Xc))
    labels = sorted(centroids)
    C = np.vstack([centroids[label] for label in labels])
    distances = np.sqrt(((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    nearest = np.argmin(distances, axis=1)
    pred = np.array([labels[i] for i in nearest], dtype=int)

    current_idx = labels.index(current_profile)
    target_idx = labels.index(target_profile)
    margin = distances[:, current_idx] - distances[:, target_idx]
    return pred, margin


def mean_knn_distance(Xc: np.ndarray, Xt: np.ndarray, k: int):
    distances = np.sqrt(((Xc[:, None, :] - Xt[None, :, :]) ** 2).sum(axis=2))
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def candidate_cost(x0, x1, denominators, l0_penalty):
    diff = np.abs(x1 - x0)
    changed = diff > 1e-12
    weighted_l1 = float(np.sum(diff / denominators))
    l0 = int(changed.sum())
    return weighted_l1 + l0_penalty * l0, weighted_l1, l0


def changes_json(x0, x1, features):
    rows = []
    for j, feature in enumerate(features):
        if abs(float(x1[j]) - float(x0[j])) > 1e-12:
            rows.append(
                {
                    "feature": feature,
                    "from": float(x0[j]),
                    "to": float(x1[j]),
                    "delta": float(x1[j] - x0[j]),
                }
            )
    return json.dumps(rows)


def exact_diagnose(
    x0,
    current_profile,
    target_profile,
    features,
    feature_grids,
    scaler,
    pca,
    centroids,
    X_target,
    plausibility_threshold,
    plausibility_k,
    denominators,
    l0_penalty,
    max_changed,
    batch_size=4096,
):
    depth_rows = []
    best_transition = None
    best_plausible_margin = None
    best_overall_margin = None

    for depth in range(1, max_changed + 1):
        total = 0
        transitions = 0
        plausible = 0
        both = 0

        depth_best_transition_plaus = np.inf
        depth_best_transition_cost = np.inf
        depth_best_plausible_margin = -np.inf
        depth_best_margin = -np.inf

        for subset in itertools.combinations(range(len(features)), depth):
            alt_lists = []
            for j in subset:
                vals = [
                    float(v)
                    for v in feature_grids[j]
                    if abs(float(v) - float(x0[j])) > 1e-12
                ]
                if not vals:
                    alt_lists = []
                    break
                alt_lists.append(vals)

            if not alt_lists:
                continue

            batch = []

            def flush():
                nonlocal total, transitions, plausible, both
                nonlocal depth_best_transition_plaus
                nonlocal depth_best_transition_cost
                nonlocal depth_best_plausible_margin
                nonlocal depth_best_margin
                nonlocal best_transition
                nonlocal best_plausible_margin
                nonlocal best_overall_margin

                if not batch:
                    return

                M = np.vstack(batch)
                pred, margin = predict_with_margin(
                    M,
                    scaler,
                    pca,
                    centroids,
                    current_profile,
                    target_profile,
                )
                plaus_dist = mean_knn_distance(
                    M,
                    X_target,
                    plausibility_k,
                )

                total += len(M)
                transition_mask = pred == target_profile
                plausible_mask = plaus_dist <= plausibility_threshold
                both_mask = transition_mask & plausible_mask

                transitions += int(transition_mask.sum())
                plausible += int(plausible_mask.sum())
                both += int(both_mask.sum())

                depth_best_margin = max(
                    depth_best_margin,
                    float(np.max(margin)),
                )

                if np.any(plausible_mask):
                    local = float(np.max(margin[plausible_mask]))
                    depth_best_plausible_margin = max(
                        depth_best_plausible_margin,
                        local,
                    )

                transition_indices = np.where(transition_mask)[0]
                for idx in transition_indices:
                    candidate = M[idx]
                    cost, weighted_l1, l0 = candidate_cost(
                        x0,
                        candidate,
                        denominators,
                        l0_penalty,
                    )

                    depth_best_transition_plaus = min(
                        depth_best_transition_plaus,
                        float(plaus_dist[idx]),
                    )
                    depth_best_transition_cost = min(
                        depth_best_transition_cost,
                        cost,
                    )

                    item = {
                        "candidate": candidate.copy(),
                        "cost": cost,
                        "weighted_l1": weighted_l1,
                        "l0": l0,
                        "plausibility_distance": float(plaus_dist[idx]),
                        "plausibility_ratio": float(
                            plaus_dist[idx] / plausibility_threshold
                        ),
                        "target_margin": float(margin[idx]),
                        "changes_json": changes_json(
                            x0,
                            candidate,
                            features,
                        ),
                    }

                    if best_transition is None or (
                        item["plausibility_ratio"],
                        item["cost"],
                    ) < (
                        best_transition["plausibility_ratio"],
                        best_transition["cost"],
                    ):
                        best_transition = item

                plausible_indices = np.where(plausible_mask)[0]
                for idx in plausible_indices:
                    item = {
                        "candidate": M[idx].copy(),
                        "target_margin": float(margin[idx]),
                        "plausibility_distance": float(plaus_dist[idx]),
                        "changes_json": changes_json(
                            x0,
                            M[idx],
                            features,
                        ),
                    }
                    if (
                        best_plausible_margin is None
                        or item["target_margin"]
                        > best_plausible_margin["target_margin"]
                    ):
                        best_plausible_margin = item

                best_idx = int(np.argmax(margin))
                overall_item = {
                    "candidate": M[best_idx].copy(),
                    "target_margin": float(margin[best_idx]),
                    "plausibility_distance": float(plaus_dist[best_idx]),
                    "changes_json": changes_json(
                        x0,
                        M[best_idx],
                        features,
                    ),
                }
                if (
                    best_overall_margin is None
                    or overall_item["target_margin"]
                    > best_overall_margin["target_margin"]
                ):
                    best_overall_margin = overall_item

                batch.clear()

            for values in itertools.product(*alt_lists):
                candidate = x0.copy()
                for j, value in zip(subset, values):
                    candidate[j] = value
                batch.append(candidate)
                if len(batch) >= batch_size:
                    flush()
            flush()

        depth_rows.append(
            {
                "depth": depth,
                "evaluated": total,
                "transition_count": transitions,
                "plausible_count": plausible,
                "valid_and_plausible_count": both,
                "best_target_margin": (
                    depth_best_margin if np.isfinite(depth_best_margin) else np.nan
                ),
                "best_plausible_target_margin": (
                    depth_best_plausible_margin
                    if np.isfinite(depth_best_plausible_margin)
                    else np.nan
                ),
                "minimum_plausibility_distance_among_transitions": (
                    depth_best_transition_plaus
                    if np.isfinite(depth_best_transition_plaus)
                    else np.nan
                ),
                "minimum_cost_among_transitions": (
                    depth_best_transition_cost
                    if np.isfinite(depth_best_transition_cost)
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(depth_rows),
        best_transition,
        best_plausible_margin,
        best_overall_margin,
    )


def target_exemplar_diagnostic(
    x0,
    X_target,
    target_ids,
    features,
    denominators,
    l0_penalty,
):
    rows = []

    for target_id, candidate in zip(target_ids, X_target):
        diff = np.abs(candidate - x0)
        l0 = int(np.sum(diff > 1e-12))
        total_cost, weighted_l1, _ = candidate_cost(
            x0,
            candidate,
            denominators,
            l0_penalty,
        )
        rows.append(
            {
                "target_institution_id": target_id,
                "l0": l0,
                "weighted_l1": weighted_l1,
                "total_cost": total_cost,
                "changes_json": changes_json(
                    x0,
                    candidate,
                    features,
                ),
            }
        )

    df = pd.DataFrame(rows)
    by_l0 = df.sort_values(["l0", "total_cost", "target_institution_id"]).iloc[0]
    by_cost = df.sort_values(["total_cost", "l0", "target_institution_id"]).iloc[0]

    return by_l0, by_cost


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    freeze = load_json(METHOD_FREEZE_PATH)

    if freeze.get("gate_status") != "PASS_STAGE_5A":
        raise ValueError("Stage 5A must pass before diagnostics.")

    id_col = fricta_config["id_column"]
    features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[[id_col, "cluster_id"]]
    grid_df = pd.read_csv(FEATURE_GRID_PATH)
    plaus_df = pd.read_csv(PLAUSIBILITY_PATH)
    smoke = pd.read_csv(SMOKE_INSTITUTIONS_PATH)

    primary[id_col] = primary[id_col].astype(str)
    labels[id_col] = labels[id_col].astype(str)
    smoke["institution_id"] = smoke["institution_id"].astype(str)

    data = primary[[id_col] + features].merge(
        labels,
        on=id_col,
        how="inner",
        validate="one_to_one",
    )

    X = data[features].to_numpy(dtype=float)
    y = data["cluster_id"].astype(int).to_numpy()

    scaler, pca, centroids = reconstruct_selected_model(
        X,
        y_reference=y,
        pca_threshold=float(cipher_config["ensemble"]["pca_variance_threshold"]),
        seed=int(cipher_config["random_seed"]),
    )

    grid_df = grid_df.set_index("feature").loc[features].reset_index()
    feature_grids = [
        np.array(json.loads(row["grid_values_json"]), dtype=float)
        for _, row in grid_df.iterrows()
    ]
    denominators = grid_df["cost_denominator"].to_numpy(dtype=float)

    l0_penalty = float(cipher_config["counterfactuals"]["l0_penalty"])
    max_changed = int(cipher_config["counterfactuals"]["maximum_changed_features"])
    plausibility_k = int(cipher_config["counterfactuals"]["plausibility_neighbors"])

    thresholds = {
        int(row["profile"]): float(row["plausibility_threshold"])
        for _, row in plaus_df.iterrows()
    }

    data_by_id = data.set_index(id_col)

    summary_rows = []
    depth_outputs = []
    transition_examples = []
    plausible_examples = []
    margin_examples = []
    exemplar_rows = []

    print("\n=== CIPHER STAGE 5B — FAILURE DIAGNOSTIC ===\n")

    for _, smoke_row in smoke.iterrows():
        institution_id = str(smoke_row["institution_id"])
        source = data_by_id.loc[institution_id]

        current_profile = int(source["cluster_id"])
        target_profile = PROFILE_2 if current_profile == PROFILE_1 else PROFILE_1
        x0 = source[features].to_numpy(dtype=float)

        target_mask = y == target_profile
        X_target = X[target_mask]
        target_ids = data.loc[target_mask, id_col].astype(str).tolist()
        threshold = thresholds[target_profile]

        depth_df, best_transition, best_plausible, best_margin = exact_diagnose(
            x0=x0,
            current_profile=current_profile,
            target_profile=target_profile,
            features=features,
            feature_grids=feature_grids,
            scaler=scaler,
            pca=pca,
            centroids=centroids,
            X_target=X_target,
            plausibility_threshold=threshold,
            plausibility_k=plausibility_k,
            denominators=denominators,
            l0_penalty=l0_penalty,
            max_changed=max_changed,
        )

        depth_df.insert(0, "institution_id", institution_id)
        depth_df.insert(1, "current_profile", current_profile)
        depth_df.insert(2, "target_profile", target_profile)
        depth_outputs.append(depth_df)

        by_l0, by_cost = target_exemplar_diagnostic(
            x0=x0,
            X_target=X_target,
            target_ids=target_ids,
            features=features,
            denominators=denominators,
            l0_penalty=l0_penalty,
        )

        total_transitions = int(depth_df["transition_count"].sum())
        total_plausible = int(depth_df["plausible_count"].sum())
        total_both = int(depth_df["valid_and_plausible_count"].sum())

        if total_transitions == 0:
            failure_mode = "NO_TRANSITION_WITHIN_4_FEATURES"
        elif total_both == 0:
            failure_mode = "TRANSITIONS_EXIST_BUT_FAIL_PLAUSIBILITY"
        else:
            failure_mode = "VALID_PLAUSIBLE_EXISTS"

        summary_rows.append(
            {
                "institution_id": institution_id,
                "certainty_class": smoke_row["certainty_class"],
                "current_profile": current_profile,
                "target_profile": target_profile,
                "transition_candidates_upto_4": total_transitions,
                "plausible_candidates_upto_4": total_plausible,
                "valid_plausible_candidates_upto_4": total_both,
                "best_target_margin_upto_4": float(
                    depth_df["best_target_margin"].max()
                ),
                "best_plausible_target_margin_upto_4": (
                    float(depth_df["best_plausible_target_margin"].max())
                    if depth_df["best_plausible_target_margin"].notna().any()
                    else np.nan
                ),
                "minimum_target_exemplar_l0": int(by_l0["l0"]),
                "minimum_target_exemplar_cost": float(by_l0["total_cost"]),
                "failure_mode": failure_mode,
            }
        )

        exemplar_rows.append(
            {
                "institution_id": institution_id,
                "criterion": "MINIMUM_L0_TARGET_EXEMPLAR",
                **by_l0.to_dict(),
            }
        )
        exemplar_rows.append(
            {
                "institution_id": institution_id,
                "criterion": "MINIMUM_COST_TARGET_EXEMPLAR",
                **by_cost.to_dict(),
            }
        )

        if best_transition is not None:
            transition_examples.append(
                {
                    "institution_id": institution_id,
                    **{k: v for k, v in best_transition.items() if k != "candidate"},
                }
            )

        if best_plausible is not None:
            plausible_examples.append(
                {
                    "institution_id": institution_id,
                    **{k: v for k, v in best_plausible.items() if k != "candidate"},
                }
            )

        if best_margin is not None:
            margin_examples.append(
                {
                    "institution_id": institution_id,
                    **{k: v for k, v in best_margin.items() if k != "candidate"},
                }
            )

    summary = pd.DataFrame(summary_rows)
    depth_all = pd.concat(depth_outputs, ignore_index=True)
    exemplars = pd.DataFrame(exemplar_rows)
    best_transition_df = pd.DataFrame(transition_examples)
    best_plausible_df = pd.DataFrame(plausible_examples)
    best_margin_df = pd.DataFrame(margin_examples)

    summary.to_csv(OUTPUT_DIR / "failure_summary.csv", index=False)
    depth_all.to_csv(OUTPUT_DIR / "depth_diagnostics.csv", index=False)
    exemplars.to_csv(OUTPUT_DIR / "target_exemplar_diagnostics.csv", index=False)
    best_transition_df.to_csv(
        OUTPUT_DIR / "best_transition_candidates.csv", index=False
    )
    best_plausible_df.to_csv(OUTPUT_DIR / "best_plausible_candidates.csv", index=False)
    best_margin_df.to_csv(OUTPUT_DIR / "best_margin_candidates.csv", index=False)

    print("=== FAILURE SUMMARY ===\n")
    print(summary.to_string(index=False))

    print("\n=== BY-DEPTH COUNTS ===\n")
    print(
        depth_all[
            [
                "institution_id",
                "depth",
                "evaluated",
                "transition_count",
                "plausible_count",
                "valid_and_plausible_count",
                "best_target_margin",
                "best_plausible_target_margin",
                "minimum_plausibility_distance_among_transitions",
            ]
        ].to_string(index=False)
    )

    print("\n=== TARGET EXEMPLAR MINIMUM L0 ===\n")
    print(
        exemplars[exemplars["criterion"] == "MINIMUM_L0_TARGET_EXEMPLAR"][
            [
                "institution_id",
                "target_institution_id",
                "l0",
                "weighted_l1",
                "total_cost",
            ]
        ].to_string(index=False)
    )

    modes = summary["failure_mode"].value_counts().to_dict()

    report = {
        "failure_modes": {str(k): int(v) for k, v in modes.items()},
        "all_six_diagnosed": len(summary) == 6,
        "any_transition_within_4": bool(
            (summary["transition_candidates_upto_4"] > 0).any()
        ),
        "any_valid_plausible_within_4": bool(
            (summary["valid_plausible_candidates_upto_4"] > 0).any()
        ),
        "minimum_target_exemplar_l0_overall": int(
            summary["minimum_target_exemplar_l0"].min()
        ),
        "gate_status": "STAGE_5B_FAILURE_DIAGNOSED_REVIEW_REQUIRED",
    }

    (OUTPUT_DIR / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nGATE STATUS: STAGE_5B_FAILURE_DIAGNOSED_REVIEW_REQUIRED")


if __name__ == "__main__":
    main()
