from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRICTA_CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
CIPHER_FROZEN_CONFIG_PATH = (
    PROJECT_ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
)
ANALYSIS_FREEZE_PATH = PROJECT_ROOT / "cipher" / "design" / "analysis_freeze.json"
HASH_MANIFEST_PATH = PROJECT_ROOT / "cipher" / "outputs" / "audit" / "input_hashes.json"

PRIMARY_MATRIX_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
)
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)

ENSEMBLE_ROOT = PROJECT_ROOT / "cipher" / "outputs" / "ensemble"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    algorithm: str
    use_pca: bool


FAMILIES = [
    FamilySpec("R0_WARD", "ward", False),
    FamilySpec("R1_PCA85_WARD", "ward", True),
    FamilySpec("R0_KMEANS", "kmeans", False),
    FamilySpec("R1_PCA85_KMEANS", "kmeans", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CIPHER heterogeneous stability ensemble."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "official"],
        default="smoke",
        help="Smoke builds 2 accepted members per family; official uses the frozen targets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory for the chosen mode.",
    )
    parser.add_argument(
        "--smoke-members-per-family",
        type=int,
        default=2,
        help="Accepted members per family in smoke mode.",
    )
    return parser.parse_args()


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


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_stage0_and_hashes() -> (
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
):
    freeze = load_json(ANALYSIS_FREEZE_PATH)
    if freeze.get("status") != "CIPHER_STAGE_0_FROZEN":
        raise ValueError(
            "CIPHER Stage 0 is not frozen. Expected status CIPHER_STAGE_0_FROZEN."
        )

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_FROZEN_CONFIG_PATH)
    hash_manifest = load_json(HASH_MANIFEST_PATH)

    if cipher_config.get("design_status") != "FROZEN_BEFORE_CIPHER_RESULTS":
        raise ValueError("Frozen CIPHER configuration has an unexpected design_status.")

    mismatches: list[str] = []
    for relative_path, metadata in hash_manifest["files"].items():
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            mismatches.append(f"MISSING: {relative_path}")
            continue
        actual = sha256_file(path)
        expected = metadata["sha256"]
        if actual != expected:
            mismatches.append(
                f"CHANGED: {relative_path}\n"
                f"  expected={expected}\n"
                f"  actual={actual}"
            )

    if mismatches:
        raise ValueError(
            "Frozen Stage 0 inputs changed after hashing:\n- " + "\n- ".join(mismatches)
        )

    return fricta_config, cipher_config, freeze


def load_aligned_data(
    fricta_config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], str]:
    id_column = fricta_config["id_column"]
    features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)

    required_primary = [id_column] + features
    missing = [column for column in required_primary if column not in primary.columns]
    if missing:
        raise KeyError(f"Primary matrix is missing columns: {missing}")

    if "cluster_id" not in labels.columns:
        raise KeyError("Final labels file must contain cluster_id.")

    aligned = primary[required_primary].merge(
        labels[[id_column, "cluster_id"]],
        on=id_column,
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) != len(primary) or len(aligned) != len(labels):
        raise ValueError("Institution IDs do not align one-to-one across inputs.")

    X = aligned[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("Primary matrix contains missing or non-numeric values.")
    if float(X.min()) < 0.0 or float(X.max()) > 1.0:
        raise ValueError("Primary matrix must remain within [0,1].")

    institution_ids = aligned[id_column].astype(str).to_numpy()
    reference_labels = aligned["cluster_id"].to_numpy()

    if len(np.unique(reference_labels)) != 2:
        raise ValueError(
            "The frozen reference partition must contain exactly two profiles."
        )

    return aligned, X, reference_labels, features, id_column


def align_labels(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[Any, Any], np.ndarray]:
    predicted_values = list(np.unique(predicted))
    reference_values = list(np.unique(reference))

    if len(predicted_values) != len(reference_values):
        raise ValueError(
            "Predicted and reference partitions do not contain the same number of labels."
        )

    contingency = np.zeros(
        (len(predicted_values), len(reference_values)),
        dtype=int,
    )

    for row_idx, predicted_value in enumerate(predicted_values):
        for col_idx, reference_value in enumerate(reference_values):
            contingency[row_idx, col_idx] = int(
                np.sum((predicted == predicted_value) & (reference == reference_value))
            )

    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {
        predicted_values[row]: reference_values[col]
        for row, col in zip(row_ind, col_ind)
    }

    aligned = np.array([mapping[value] for value in predicted], dtype=reference.dtype)
    return aligned, mapping, contingency


def nearest_centroid_labels(
    Z_all: np.ndarray,
    Z_sample: np.ndarray,
    fitted_labels: np.ndarray,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    centroids: dict[Any, np.ndarray] = {}
    unique_labels = list(np.unique(fitted_labels))

    for label in unique_labels:
        centroids[label] = Z_sample[fitted_labels == label].mean(axis=0)

    centroid_matrix = np.vstack([centroids[label] for label in unique_labels])
    squared_distances = ((Z_all[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(
        axis=2
    )
    nearest_indices = np.argmin(squared_distances, axis=1)
    predicted = np.array([unique_labels[index] for index in nearest_indices])

    return predicted, centroids


def internal_metrics(Z_sample: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        raise ValueError("Cannot compute clustering metrics with one cluster.")

    return {
        "silhouette": float(silhouette_score(Z_sample, labels)),
        "davies_bouldin": float(davies_bouldin_score(Z_sample, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Z_sample, labels)),
    }


def family_target_count(
    family: FamilySpec,
    mode: str,
    cipher_config: dict[str, Any],
    smoke_members_per_family: int,
) -> int:
    if mode == "smoke":
        if smoke_members_per_family < 1:
            raise ValueError("--smoke-members-per-family must be at least 1.")
        return smoke_members_per_family

    frozen_targets = cipher_config["ensemble"]["families"]
    if family.name not in frozen_targets:
        raise KeyError(f"Missing frozen target for family {family.name}.")
    return int(frozen_targets[family.name])


def make_output_directory(mode: str, overwrite: bool) -> Path:
    output_dir = ENSEMBLE_ROOT / mode

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite only when intentionally rerunning the same mode."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_ensemble(
    mode: str,
    output_dir: Path,
    X: np.ndarray,
    institution_ids: np.ndarray,
    reference_labels: np.ndarray,
    features: list[str],
    cipher_config: dict[str, Any],
    smoke_members_per_family: int,
) -> dict[str, Any]:
    base_seed = int(cipher_config["random_seed"])
    n_institutions = X.shape[0]
    sample_fraction = float(cipher_config["ensemble"]["institution_sample_fraction"])
    sample_size = int(math.ceil(sample_fraction * n_institutions))
    feature_sample_count = int(cipher_config["ensemble"]["feature_sample_count"])
    pca_threshold = float(cipher_config["ensemble"]["pca_variance_threshold"])
    minimum_cluster_size = int(
        cipher_config["ensemble"]["minimum_sampled_cluster_size"]
    )
    kmeans_n_init = int(cipher_config["ensemble"]["kmeans_n_init"])

    if sample_size >= n_institutions:
        raise ValueError("Row sample size must leave at least one OOB institution.")
    if feature_sample_count >= len(features):
        raise ValueError("Feature sample count must be smaller than total features.")

    all_member_predictions: dict[str, np.ndarray] = {}
    oob_member_predictions: dict[str, np.ndarray] = {}
    manifests: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    family_summaries: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for family_index, family in enumerate(FAMILIES):
        target = family_target_count(
            family,
            mode,
            cipher_config,
            smoke_members_per_family,
        )

        accepted = 0
        attempts = 0
        max_attempts = max(target * 5, target + 20)

        while accepted < target and attempts < max_attempts:
            attempts += 1
            member_seed = base_seed + (family_index + 1) * 1_000_000 + attempts
            rng = np.random.default_rng(member_seed)

            sample_indices = np.sort(
                rng.choice(
                    n_institutions,
                    size=sample_size,
                    replace=False,
                )
            )
            oob_mask = np.ones(n_institutions, dtype=bool)
            oob_mask[sample_indices] = False
            oob_indices = np.flatnonzero(oob_mask)

            feature_indices = np.sort(
                rng.choice(
                    len(features),
                    size=feature_sample_count,
                    replace=False,
                )
            )
            selected_features = [features[index] for index in feature_indices]

            X_selected = X[:, feature_indices]
            scaler = StandardScaler()
            Z_sample_raw = scaler.fit_transform(X_selected[sample_indices])
            Z_all_raw = scaler.transform(X_selected)

            pca_components = None
            pca_explained_variance = None

            if family.use_pca:
                pca = PCA(
                    n_components=pca_threshold,
                    svd_solver="full",
                    random_state=member_seed,
                )
                Z_sample = pca.fit_transform(Z_sample_raw)
                Z_all = pca.transform(Z_all_raw)
                pca_components = int(pca.n_components_)
                pca_explained_variance = float(pca.explained_variance_ratio_.sum())
            else:
                Z_sample = Z_sample_raw
                Z_all = Z_all_raw

            if family.algorithm == "ward":
                estimator = AgglomerativeClustering(
                    n_clusters=2,
                    linkage="ward",
                )
                fitted_labels = estimator.fit_predict(Z_sample)
                extension_labels_all, _ = nearest_centroid_labels(
                    Z_all,
                    Z_sample,
                    fitted_labels,
                )
                extension_labels_sample = extension_labels_all[sample_indices]
                extension_fidelity = float(
                    np.mean(extension_labels_sample == fitted_labels)
                )
            elif family.algorithm == "kmeans":
                estimator = KMeans(
                    n_clusters=2,
                    n_init=kmeans_n_init,
                    random_state=member_seed,
                )
                fitted_labels = estimator.fit_predict(Z_sample)
                extension_labels_all = estimator.predict(Z_all)
                extension_fidelity = 1.0
            else:
                raise ValueError(f"Unsupported algorithm: {family.algorithm}")

            raw_cluster_sizes = {
                str(to_python_scalar(label)): int(np.sum(fitted_labels == label))
                for label in np.unique(fitted_labels)
            }

            smallest_cluster = min(raw_cluster_sizes.values())
            if smallest_cluster < minimum_cluster_size:
                rejected_rows.append(
                    {
                        "family": family.name,
                        "attempt": attempts,
                        "seed": member_seed,
                        "reason": "MINIMUM_CLUSTER_SIZE",
                        "raw_cluster_sizes_json": json.dumps(raw_cluster_sizes),
                    }
                )
                continue

            reference_sample = reference_labels[sample_indices]
            aligned_sample_labels, mapping, contingency = align_labels(
                fitted_labels,
                reference_sample,
            )

            mapped_extension_all = np.array(
                [mapping[label] for label in extension_labels_all],
                dtype=reference_labels.dtype,
            )

            hybrid_all = mapped_extension_all.copy()
            hybrid_all[sample_indices] = aligned_sample_labels

            oob_only = np.full(n_institutions, np.nan, dtype=object)
            oob_only[oob_indices] = mapped_extension_all[oob_indices]

            accepted += 1
            member_id = f"{family.name}__M{accepted:04d}"

            metric_values = internal_metrics(Z_sample, fitted_labels)
            reference_ari = float(
                adjusted_rand_score(reference_sample, aligned_sample_labels)
            )

            aligned_cluster_sizes = {
                str(to_python_scalar(label)): int(
                    np.sum(aligned_sample_labels == label)
                )
                for label in np.unique(reference_labels)
            }

            all_member_predictions[member_id] = hybrid_all
            oob_member_predictions[member_id] = oob_only

            manifests.append(
                {
                    "member_id": member_id,
                    "family": family.name,
                    "algorithm": family.algorithm,
                    "representation": "PCA85" if family.use_pca else "RAW_STANDARDIZED",
                    "accepted_index_within_family": accepted,
                    "attempt_within_family": attempts,
                    "seed": member_seed,
                    "sample_size": sample_size,
                    "oob_size": int(len(oob_indices)),
                    "feature_count": feature_sample_count,
                    "feature_names_json": json.dumps(selected_features),
                    "sampled_institution_ids_json": json.dumps(
                        institution_ids[sample_indices].tolist()
                    ),
                    "oob_institution_ids_json": json.dumps(
                        institution_ids[oob_indices].tolist()
                    ),
                    "pca_components": pca_components,
                    "pca_explained_variance": pca_explained_variance,
                    "label_mapping_json": json.dumps(
                        {
                            str(to_python_scalar(key)): to_python_scalar(value)
                            for key, value in mapping.items()
                        }
                    ),
                    "alignment_contingency_json": json.dumps(contingency.tolist()),
                }
            )

            metrics_rows.append(
                {
                    "member_id": member_id,
                    "family": family.name,
                    "silhouette": metric_values["silhouette"],
                    "davies_bouldin": metric_values["davies_bouldin"],
                    "calinski_harabasz": metric_values["calinski_harabasz"],
                    "reference_ari_on_sample": reference_ari,
                    "ward_or_native_extension_fidelity_on_sample": extension_fidelity,
                    "minimum_sampled_cluster_size": smallest_cluster,
                    "raw_cluster_sizes_json": json.dumps(raw_cluster_sizes),
                    "aligned_cluster_sizes_json": json.dumps(aligned_cluster_sizes),
                }
            )

        family_summaries[family.name] = {
            "target_accepted_members": target,
            "accepted_members": accepted,
            "attempts": attempts,
            "rejected_attempts": attempts - accepted,
            "acceptance_rate": accepted / attempts if attempts else 0.0,
        }

        if accepted < target:
            raise RuntimeError(
                f"Could not obtain {target} admissible members for {family.name}; "
                f"accepted {accepted} after {attempts} attempts."
            )

    elapsed_seconds = time.perf_counter() - started

    predictions_all = pd.DataFrame({"institution_id": institution_ids})
    predictions_oob = pd.DataFrame({"institution_id": institution_ids})

    for member_id, values in all_member_predictions.items():
        predictions_all[member_id] = values
    for member_id, values in oob_member_predictions.items():
        predictions_oob[member_id] = values

    manifest_df = pd.DataFrame(manifests)
    metrics_df = pd.DataFrame(metrics_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    predictions_all.to_csv(output_dir / "member_predictions_all.csv", index=False)
    predictions_oob.to_csv(output_dir / "member_predictions_oob.csv", index=False)
    manifest_df.to_csv(output_dir / "member_manifest.csv", index=False)
    metrics_df.to_csv(output_dir / "member_metrics.csv", index=False)
    rejected_df.to_csv(output_dir / "rejected_attempts.csv", index=False)

    member_ids = list(all_member_predictions.keys())
    prediction_matrix = np.column_stack(
        [all_member_predictions[member_id] for member_id in member_ids]
    )

    n = n_institutions
    coassignment_sum = np.zeros((n, n), dtype=float)
    for column_index in range(prediction_matrix.shape[1]):
        labels_column = prediction_matrix[:, column_index]
        coassignment_sum += (labels_column[:, None] == labels_column[None, :]).astype(
            float
        )
    coassignment_all = coassignment_sum / prediction_matrix.shape[1]

    oob_matrix = np.column_stack(
        [oob_member_predictions[member_id] for member_id in member_ids]
    )
    oob_coassignment_sum = np.zeros((n, n), dtype=float)
    oob_pair_counts = np.zeros((n, n), dtype=int)

    for column_index in range(oob_matrix.shape[1]):
        labels_column = oob_matrix[:, column_index]
        valid = pd.notna(labels_column)
        valid_pairs = np.outer(valid, valid)
        same = (labels_column[:, None] == labels_column[None, :]) & valid_pairs
        oob_coassignment_sum += same.astype(float)
        oob_pair_counts += valid_pairs.astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        coassignment_oob = np.divide(
            oob_coassignment_sum,
            oob_pair_counts,
            out=np.full_like(oob_coassignment_sum, np.nan, dtype=float),
            where=oob_pair_counts > 0,
        )

    def matrix_frame(matrix: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(
            matrix,
            index=institution_ids,
            columns=institution_ids,
        )
        frame.index.name = "institution_id"
        return frame

    matrix_frame(coassignment_all).to_csv(output_dir / "coassignment_matrix.csv")
    matrix_frame(coassignment_oob).to_csv(output_dir / "coassignment_oob_matrix.csv")
    matrix_frame(oob_pair_counts).to_csv(output_dir / "coassignment_oob_counts.csv")

    family_consensus_rows: list[dict[str, Any]] = []
    reference_values = list(np.unique(reference_labels))

    for family in FAMILIES:
        family_member_ids = manifest_df.loc[
            manifest_df["family"] == family.name,
            "member_id",
        ].tolist()

        for institution_index, institution_id in enumerate(institution_ids):
            values = [
                oob_member_predictions[member_id][institution_index]
                for member_id in family_member_ids
                if pd.notna(oob_member_predictions[member_id][institution_index])
            ]

            row: dict[str, Any] = {
                "institution_id": institution_id,
                "family": family.name,
                "n_oob_predictions": len(values),
            }

            for profile in reference_values:
                row[f"profile_{profile}_probability"] = (
                    float(np.mean(np.array(values) == profile)) if values else np.nan
                )

            family_consensus_rows.append(row)

    family_consensus_df = pd.DataFrame(family_consensus_rows)
    family_consensus_df.to_csv(
        output_dir / "family_consensus.csv",
        index=False,
    )

    oob_prediction_counts = (
        predictions_oob.drop(columns=["institution_id"]).notna().sum(axis=1)
    )

    accepted_total = int(len(member_ids))
    attempts_total = int(sum(item["attempts"] for item in family_summaries.values()))
    acceptance_rate_total = accepted_total / attempts_total if attempts_total else 0.0

    every_family_has_both_profiles = True
    for family in FAMILIES:
        family_ids = manifest_df.loc[
            manifest_df["family"] == family.name,
            "member_id",
        ].tolist()
        family_values = predictions_all[family_ids].to_numpy().ravel()
        if set(np.unique(family_values)) != set(reference_values):
            every_family_has_both_profiles = False

    report = {
        "mode": mode,
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "random_seed": base_seed,
        "n_institutions": n_institutions,
        "sample_fraction": sample_fraction,
        "sample_size_rule": "ceil(fraction * n)",
        "sample_size": sample_size,
        "oob_size_per_member": n_institutions - sample_size,
        "total_features": len(features),
        "sampled_features_per_member": feature_sample_count,
        "reference_profiles": [to_python_scalar(value) for value in reference_values],
        "accepted_members": accepted_total,
        "attempts": attempts_total,
        "overall_acceptance_rate": acceptance_rate_total,
        "family_summaries": family_summaries,
        "oob_predictions_per_institution": {
            "minimum": int(oob_prediction_counts.min()),
            "median": float(oob_prediction_counts.median()),
            "maximum": int(oob_prediction_counts.max()),
        },
        "minimum_oob_pair_count": int(oob_pair_counts[np.triu_indices(n, k=1)].min()),
        "every_family_has_both_profiles": every_family_has_both_profiles,
        "elapsed_seconds": elapsed_seconds,
    }

    if mode == "official":
        gate_checks = {
            "accepted_exact_frozen_target": accepted_total == 1000,
            "overall_acceptance_rate_at_least_0_90": acceptance_rate_total >= 0.90,
            "minimum_oob_predictions_at_least_150": int(oob_prediction_counts.min())
            >= 150,
            "every_family_has_both_profiles": every_family_has_both_profiles,
        }
        report["gate_checks"] = gate_checks
        report["gate_status"] = (
            "PASS_STAGE_1"
            if all(gate_checks.values())
            else "FAIL_STAGE_1_REVIEW_REQUIRED"
        )
    else:
        report["gate_status"] = "SMOKE_COMPLETE_REVIEW_REQUIRED"

    (output_dir / "ensemble_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report


def print_report(report: dict[str, Any], output_dir: Path) -> None:
    print("\n=== CIPHER STAGE 1 ENSEMBLE REPORT ===\n")
    print(f"Mode: {report['mode']}")
    print(f"Accepted members: {report['accepted_members']}")
    print(f"Attempts: {report['attempts']}")
    print(f"Overall acceptance rate: {report['overall_acceptance_rate']:.4f}")
    print(f"Sample size per member: {report['sample_size']}")
    print(f"OOB size per member: {report['oob_size_per_member']}")
    print(
        "OOB predictions per institution: "
        f"min={report['oob_predictions_per_institution']['minimum']}, "
        f"median={report['oob_predictions_per_institution']['median']:.1f}, "
        f"max={report['oob_predictions_per_institution']['maximum']}"
    )
    print("Minimum OOB pair count: " f"{report['minimum_oob_pair_count']}")
    print(
        "Every family contains both profiles: "
        f"{report['every_family_has_both_profiles']}"
    )
    print(f"Elapsed seconds: {report['elapsed_seconds']:.3f}")

    print("\nFamily summaries:")
    for family_name, summary in report["family_summaries"].items():
        print(
            f"- {family_name}: accepted={summary['accepted_members']}, "
            f"attempts={summary['attempts']}, "
            f"acceptance_rate={summary['acceptance_rate']:.4f}"
        )

    print(f"\nOutput directory: {output_dir}")
    print(f"\nGATE STATUS: {report['gate_status']}")


def main() -> None:
    args = parse_args()

    fricta_config, cipher_config, _ = validate_stage0_and_hashes()
    _, X, reference_labels, features, _ = load_aligned_data(fricta_config)

    institution_ids = (
        pd.read_csv(PRIMARY_MATRIX_PATH)[fricta_config["id_column"]]
        .astype(str)
        .to_numpy()
    )

    # Re-align IDs to the merged data order used by load_aligned_data.
    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)
    aligned_ids = (
        primary[[fricta_config["id_column"]]]
        .merge(
            labels[[fricta_config["id_column"]]],
            on=fricta_config["id_column"],
            how="inner",
            validate="one_to_one",
        )[fricta_config["id_column"]]
        .astype(str)
        .to_numpy()
    )

    institution_ids = aligned_ids

    output_dir = make_output_directory(args.mode, args.overwrite)

    try:
        report = build_ensemble(
            mode=args.mode,
            output_dir=output_dir,
            X=X,
            institution_ids=institution_ids,
            reference_labels=reference_labels,
            features=features,
            cipher_config=cipher_config,
            smoke_members_per_family=args.smoke_members_per_family,
        )
    except Exception:
        # Keep the directory for forensic inspection, but mark the run as failed.
        (output_dir / "RUN_FAILED.txt").write_text(
            "The run failed. Inspect the console traceback before rerunning.\n",
            encoding="utf-8",
        )
        raise

    print_report(report, output_dir)


if __name__ == "__main__":
    main()
