from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FRICTA_CONFIG_PATH = ROOT / "icdm" / "design" / "experiment_config.json"
CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
PRIMARY_MATRIX_PATH = ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
FINAL_LABELS_PATH = (
    ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
SELECTED_MODEL_PATH = ROOT / "icdm" / "outputs" / "clustering" / "selected_model.json"
ACTIONABILITY_PATH = ROOT / "cipher" / "design" / "actionability_manifest.csv"
STAGE4_REPORT_PATH = (
    ROOT / "cipher" / "outputs" / "inductive_validation" / "inductive_report.json"
)

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "counterfactuals" / "preparation"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"
METHOD_FREEZE_PATH = ROOT / "cipher" / "design" / "counterfactual_method_freeze.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_labels(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    predicted_values = list(np.unique(predicted))
    reference_values = list(np.unique(reference))

    if len(predicted_values) != len(reference_values):
        raise ValueError(
            "Predicted and reference partitions have different numbers of labels."
        )

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

    aligned = np.array(
        [mapping[int(value)] for value in predicted],
        dtype=int,
    )

    return aligned, mapping


def nearest_centroid_predict(
    Z: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    labels = sorted(centroids)
    centroid_matrix = np.vstack([centroids[label] for label in labels])

    distances = np.sqrt(
        ((Z[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    )

    nearest_idx = np.argmin(distances, axis=1)

    predicted = np.array(
        [labels[idx] for idx in nearest_idx],
        dtype=int,
    )

    return predicted, distances


def mean_knn_distance(
    X_query: np.ndarray,
    X_reference: np.ndarray,
    k: int,
    exclude_self: bool,
) -> np.ndarray:
    distances = np.sqrt(
        ((X_query[:, None, :] - X_reference[None, :, :]) ** 2).sum(axis=2)
    )

    if exclude_self:
        if len(X_query) != len(X_reference):
            raise ValueError(
                "exclude_self=True requires query and reference arrays of equal length."
            )
        np.fill_diagonal(distances, np.inf)

    if X_reference.shape[0] - int(exclude_self) < k:
        raise ValueError(f"Not enough reference points for k={k} nearest neighbors.")

    nearest = np.partition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return nearest.mean(axis=1)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if METHOD_FREEZE_PATH.exists():
        raise FileExistsError(
            "Counterfactual method is already frozen. "
            f"Existing file: {METHOD_FREEZE_PATH}"
        )

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)
    selected_model = load_json(SELECTED_MODEL_PATH)
    stage4_report = load_json(STAGE4_REPORT_PATH)

    if stage4_report.get("gate_status") != "PASS_STAGE_4":
        raise ValueError(
            "Stage 4 has not passed. Counterfactual preparation is not authorized."
        )

    if selected_model.get("candidate_id") != cipher_config["reference_model"]:
        raise ValueError(
            "Selected ICDM model does not match the frozen CIPHER reference model."
        )

    id_column = fricta_config["id_column"]
    features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[[id_column, "cluster_id"]]
    actionability = pd.read_csv(ACTIONABILITY_PATH)

    primary[id_column] = primary[id_column].astype(str)
    labels[id_column] = labels[id_column].astype(str)

    if set(actionability["feature"]) != set(features):
        raise ValueError(
            "Actionability manifest does not contain exactly the 13 frozen features."
        )

    if not (actionability["review_status"].astype(str) == "CONFIRMED").all():
        raise ValueError("Every actionability row must be CONFIRMED before Stage 5.")

    included = actionability.loc[
        actionability["included_in_diagnostic_counterfactual_search"].astype(bool),
        "feature",
    ].tolist()

    if set(included) != set(features):
        raise ValueError(
            "Stage 5 expects all 13 frozen primary indicators to remain "
            "eligible for diagnostic counterfactual search."
        )

    data = primary[[id_column] + features].merge(
        labels,
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 aligned institutions; found {len(data)}.")

    X = (
        data[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    if np.isnan(X).any():
        raise ValueError("Primary feature matrix contains missing values.")

    if float(X.min()) < 0 or float(X.max()) > 1:
        raise ValueError("Primary feature matrix must remain within [0,1].")

    y_reference = data["cluster_id"].astype(int).to_numpy()

    # Reconstruct the selected full-data representation.
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(X)

    pca_threshold = float(cipher_config["ensemble"]["pca_variance_threshold"])

    pca = PCA(
        n_components=pca_threshold,
        svd_solver="full",
        random_state=int(cipher_config["random_seed"]),
    )
    Z = pca.fit_transform(Z_scaled)

    ward = AgglomerativeClustering(
        n_clusters=2,
        linkage="ward",
    )
    ward_raw = ward.fit_predict(Z)

    ward_aligned, mapping = align_labels(
        ward_raw,
        y_reference,
    )

    ward_ari = float(
        adjusted_rand_score(
            y_reference,
            ward_aligned,
        )
    )
    exact_reference_match = bool(
        np.array_equal(
            y_reference,
            ward_aligned,
        )
    )

    centroids: dict[int, np.ndarray] = {
        profile: Z[ward_aligned == profile].mean(axis=0)
        for profile in sorted(np.unique(ward_aligned))
    }

    centroid_predictions, centroid_distances = nearest_centroid_predict(
        Z,
        centroids,
    )

    centroid_fidelity_to_ward = float(np.mean(centroid_predictions == ward_aligned))

    centroid_fidelity_to_reference = float(np.mean(centroid_predictions == y_reference))

    profile_labels = sorted(np.unique(y_reference).tolist())

    if profile_labels != [1, 2]:
        raise ValueError(f"Expected reference profiles [1,2]; found {profile_labels}.")

    # Feature grids and primary counterfactual cost scales.
    feature_rows = []

    iqr_epsilon = float(cipher_config["counterfactuals"]["iqr_epsilon"])

    for feature_index, feature in enumerate(features):
        values = np.sort(np.unique(X[:, feature_index]))
        q1 = float(
            np.quantile(
                X[:, feature_index],
                0.25,
            )
        )
        q3 = float(
            np.quantile(
                X[:, feature_index],
                0.75,
            )
        )
        iqr = q3 - q1

        action_row = actionability.loc[actionability["feature"] == feature].iloc[0]

        feature_rows.append(
            {
                "feature": feature,
                "grid_levels": int(len(values)),
                "grid_values_json": json.dumps([float(v) for v in values]),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "cost_denominator": float(iqr + iqr_epsilon),
                "actionability_class": str(action_row["actionability_class"]),
                "diagnostic_direction": str(
                    action_row["diagnostic_counterfactual_direction"]
                ),
                "realistic_improvement_direction": str(
                    action_row["realistic_improvement_direction"]
                ),
            }
        )

    feature_grid = pd.DataFrame(feature_rows)
    feature_grid.to_csv(
        OUTPUT_DIR / "feature_grid_summary.csv",
        index=False,
    )

    # Plausibility is measured in original normalized 13-D feature space.
    # Metric is frozen here, before any counterfactual result is generated.
    plausibility_k = int(cipher_config["counterfactuals"]["plausibility_neighbors"])
    plausibility_percentile = float(
        cipher_config["counterfactuals"]["plausibility_percentile"]
    )

    plausibility_rows = []
    institution_plausibility_rows = []

    for profile in profile_labels:
        X_profile = X[y_reference == profile]

        within_mean_knn = mean_knn_distance(
            X_profile,
            X_profile,
            k=plausibility_k,
            exclude_self=True,
        )

        threshold = float(
            np.quantile(
                within_mean_knn,
                plausibility_percentile,
            )
        )

        plausibility_rows.append(
            {
                "profile": int(profile),
                "n_institutions": int(len(X_profile)),
                "metric": "EUCLIDEAN_ORIGINAL_NORMALIZED_13D",
                "k_neighbors": plausibility_k,
                "threshold_percentile": plausibility_percentile,
                "mean_knn_distance_min": float(within_mean_knn.min()),
                "mean_knn_distance_median": float(np.median(within_mean_knn)),
                "mean_knn_distance_max": float(within_mean_knn.max()),
                "plausibility_threshold": threshold,
            }
        )

        profile_ids = (
            data.loc[
                data["cluster_id"] == profile,
                id_column,
            ]
            .astype(str)
            .tolist()
        )

        for institution_id, distance_value in zip(
            profile_ids,
            within_mean_knn,
        ):
            institution_plausibility_rows.append(
                {
                    "institution_id": institution_id,
                    "profile": int(profile),
                    "within_profile_mean_5nn_distance": float(distance_value),
                }
            )

    plausibility = pd.DataFrame(plausibility_rows)
    plausibility.to_csv(
        OUTPUT_DIR / "plausibility_thresholds.csv",
        index=False,
    )

    pd.DataFrame(institution_plausibility_rows).to_csv(
        OUTPUT_DIR / "institution_plausibility_reference.csv",
        index=False,
    )

    # Record selected-model centroid geometry for later reproducibility.
    geometry_rows = []

    label_order = sorted(centroids)

    for i, institution_id in enumerate(data[id_column].astype(str)):
        distance_by_profile = {
            profile: float(
                centroid_distances[
                    i,
                    label_order.index(profile),
                ]
            )
            for profile in label_order
        }

        current_profile = int(y_reference[i])
        target_profile = PROFILE_2 if current_profile == PROFILE_1 else PROFILE_1

        geometry_rows.append(
            {
                "institution_id": institution_id,
                "reference_profile": current_profile,
                "centroid_prediction": int(centroid_predictions[i]),
                "distance_profile_1": distance_by_profile[1],
                "distance_profile_2": distance_by_profile[2],
                "target_profile": target_profile,
                "target_margin": float(
                    distance_by_profile[current_profile]
                    - distance_by_profile[target_profile]
                ),
            }
        )

    geometry = pd.DataFrame(geometry_rows)
    geometry.to_csv(
        OUTPUT_DIR / "selected_model_geometry.csv",
        index=False,
    )

    checks = {
        "stage4_passed": (stage4_report.get("gate_status") == "PASS_STAGE_4"),
        "selected_model_matches_frozen_reference": (
            selected_model.get("candidate_id") == cipher_config["reference_model"]
        ),
        "81_institutions": (len(data) == 81),
        "13_features": (len(features) == 13),
        "all_actionability_confirmed": bool(
            (actionability["review_status"] == "CONFIRMED").all()
        ),
        "all_13_features_counterfactually_eligible": (set(included) == set(features)),
        "reconstructed_ward_ari_1": (abs(ward_ari - 1.0) <= 1e-12),
        "reconstructed_ward_exact_reference_match": (exact_reference_match),
        "selected_model_centroid_extension_fidelity_at_least_095": (
            centroid_fidelity_to_ward >= 0.95
        ),
        "every_feature_has_at_least_2_observed_levels": bool(
            (feature_grid["grid_levels"] >= 2).all()
        ),
        "plausibility_thresholds_finite": bool(
            np.isfinite(
                plausibility["plausibility_threshold"].to_numpy(dtype=float)
            ).all()
        ),
    }

    method_freeze = {
        "status": ("COUNTERFACTUAL_METHOD_FROZEN_BEFORE_COUNTERFACTUAL_RESULTS"),
        "selected_reference_model": (cipher_config["reference_model"]),
        "selected_model_reconstruction": {
            "scaler": "StandardScaler fit on all 81 institutions",
            "pca": {
                "variance_threshold": pca_threshold,
                "svd_solver": "full",
                "components_retained": int(pca.n_components_),
                "variance_explained": float(pca.explained_variance_ratio_.sum()),
            },
            "clustering": ("AgglomerativeClustering(" "n_clusters=2, linkage='ward')"),
            "label_alignment": ("Hungarian overlap alignment to frozen profile IDs"),
            "ward_ari_vs_frozen_reference": ward_ari,
            "exact_reference_match": exact_reference_match,
        },
        "inductive_extension": {
            "method": ("nearest centroid in selected PCA representation"),
            "training_fidelity_to_reconstructed_ward": (centroid_fidelity_to_ward),
            "training_fidelity_to_frozen_reference": (centroid_fidelity_to_reference),
            "interpretation": (
                "diagnostic inductive extension of a transductive Ward "
                "partition; not claimed to be native Ward prediction"
            ),
        },
        "search_space": {
            "features": features,
            "candidate_values": ("empirically observed normalized levels per feature"),
            "maximum_changed_features": int(
                cipher_config["counterfactuals"]["maximum_changed_features"]
            ),
            "beam_width": int(cipher_config["counterfactuals"]["beam_width"]),
            "max_diverse_counterfactuals_per_institution": int(
                cipher_config["counterfactuals"][
                    "max_diverse_counterfactuals_per_institution"
                ]
            ),
        },
        "cost": {
            "primary": (
                "sum_j |x_j-x'_j|/(IQR_j+epsilon) "
                "+ l0_penalty * number_changed_features"
            ),
            "iqr_epsilon": iqr_epsilon,
            "l0_penalty": float(cipher_config["counterfactuals"]["l0_penalty"]),
        },
        "plausibility": {
            "metric": "Euclidean distance in original normalized 13-D feature space",
            "k_neighbors": plausibility_k,
            "target_profile_threshold": (
                "95th percentile of within-profile mean 5-NN distance"
            ),
            "percentile": plausibility_percentile,
        },
        "interpretation": (
            "Counterfactuals are diagnostic profile-transition explanations, "
            "not causal intervention or treatment recommendations."
        ),
        "source_hashes": {
            "primary_matrix": sha256_file(PRIMARY_MATRIX_PATH),
            "final_labels": sha256_file(FINAL_LABELS_PATH),
            "actionability_manifest": sha256_file(ACTIONABILITY_PATH),
            "stage4_report": sha256_file(STAGE4_REPORT_PATH),
        },
        "checks": checks,
        "gate_status": ("PASS_STAGE_5A" if all(checks.values()) else "FAIL_STAGE_5A"),
    }

    METHOD_FREEZE_PATH.write_text(
        json.dumps(
            method_freeze,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "selected_model_inductive_audit.json").write_text(
        json.dumps(
            {
                "ward_ari_vs_frozen_reference": ward_ari,
                "exact_reference_match": exact_reference_match,
                "centroid_extension_fidelity_to_ward": (centroid_fidelity_to_ward),
                "centroid_extension_fidelity_to_reference": (
                    centroid_fidelity_to_reference
                ),
                "pca_components": int(pca.n_components_),
                "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                "label_mapping": mapping,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 5A — COUNTERFACTUAL METHOD PREPARATION ===\n")

    print(
        "Selected model:",
        cipher_config["reference_model"],
    )
    print(
        "Reconstructed Ward ARI vs frozen reference:",
        f"{ward_ari:.4f}",
    )
    print(
        "Exact frozen-label match:",
        exact_reference_match,
    )
    print(
        "PCA components retained:",
        int(pca.n_components_),
    )
    print(
        "PCA variance explained:",
        f"{pca.explained_variance_ratio_.sum():.4f}",
    )
    print(
        "Nearest-centroid extension fidelity to Ward:",
        f"{centroid_fidelity_to_ward:.4f}",
    )
    print(
        "Nearest-centroid extension fidelity to frozen reference:",
        f"{centroid_fidelity_to_reference:.4f}",
    )

    print("\n=== FEATURE GRID SUMMARY ===\n")
    print(
        feature_grid[
            [
                "feature",
                "grid_levels",
                "iqr",
                "cost_denominator",
                "actionability_class",
            ]
        ].to_string(index=False)
    )

    print("\n=== PLAUSIBILITY THRESHOLDS ===\n")
    print(plausibility.to_string(index=False))

    print("\n=== GATE CHECKS ===\n")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nGATE STATUS: {method_freeze['gate_status']}")

    if method_freeze["gate_status"] == "PASS_STAGE_5A":
        print(
            "Counterfactual search geometry is frozen. "
            "Stage 5B smoke search may begin after review."
        )


PROFILE_1 = 1
PROFILE_2 = 2


if __name__ == "__main__":
    main()
