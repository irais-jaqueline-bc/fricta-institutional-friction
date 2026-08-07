from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from cipher_03_null_models import severity_cross_validation
from cipher_synthetic_generators import FEATURE_NAMES, generate_scenario

ROOT = Path(__file__).resolve().parents[1]

EVALUATOR_PATH = ROOT / "cipher" / "design" / "stage8_evaluator_freeze_v2.json"
GENERATOR_FREEZE_PATH = (
    ROOT / "cipher" / "design" / "stage8_generator_implementation_freeze_v1.json"
)
STAGE8B_AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8b_generator_smoke_audit.json"
)

OUTPUT_ROOT = ROOT / "cipher" / "outputs" / "synthetic" / "performance_smoke_v2"
AUDIT_PATH = (
    ROOT / "cipher" / "outputs" / "audit" / "stage8d1_core_pipeline_smoke_audit_v2.json"
)

MASTER_SEED = 20260807
SMOKE_REPLICATE = 1
STABILITY_ITERATIONS = 40
STABILITY_SAMPLE_FRACTION = 0.80
ENSEMBLE_MEMBERS_PER_FAMILY = 50
ENSEMBLE_FEATURE_COUNT = 11
ENSEMBLE_SAMPLE_FRACTION = 0.80

SCENARIOS = [
    "S1_CONFIG_TWO_PROFILE",
    "S2_CORE_BOUNDARY",
    "S3_DIRECTIONAL_REACHABILITY",
    "S4_SEVERITY_CONTINUUM",
    "S5_GOVERNANCE_CONFOUNDED",
    "S6_NO_CLUSTER_NULL",
]

REPRESENTATIONS = ["R0_STANDARDIZED", "R1_PCA85"]
ALGORITHMS = ["HAC_WARD", "KMEANS"]
K_VALUES = [2, 3, 4, 5, 6]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def transform_representation(
    X: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, StandardScaler, PCA | None]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    if representation == "R0_STANDARDIZED":
        return Z, scaler, None

    if representation == "R1_PCA85":
        pca = PCA(
            n_components=0.85,
            svd_solver="full",
        )
        Zp = pca.fit_transform(Z)
        return Zp, scaler, pca

    raise KeyError(representation)


def apply_representation(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA | None,
) -> np.ndarray:
    Z = scaler.transform(X)
    if pca is not None:
        Z = pca.transform(Z)
    return Z


def fit_cluster(
    Z: np.ndarray,
    algorithm: str,
    k: int,
    seed: int,
):
    if algorithm == "HAC_WARD":
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )
        labels = model.fit_predict(Z)
        return model, labels

    if algorithm == "KMEANS":
        model = KMeans(
            n_clusters=k,
            n_init=25,
            random_state=seed,
        )
        labels = model.fit_predict(Z)
        return model, labels

    raise KeyError(algorithm)


def internal_metrics(
    Z: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    counts = pd.Series(labels).value_counts()

    return {
        "silhouette": float(
            silhouette_score(
                Z,
                labels,
                metric="euclidean",
            )
        ),
        "davies_bouldin": float(
            davies_bouldin_score(
                Z,
                labels,
            )
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(
                Z,
                labels,
            )
        ),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
    }


def full_candidate_search(
    X: np.ndarray,
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    labels_by_candidate: dict[str, np.ndarray] = {}

    for representation in REPRESENTATIONS:
        Z, _, _ = transform_representation(
            X,
            representation,
        )

        for algorithm in ALGORITHMS:
            for k in K_VALUES:
                candidate_id = f"{representation}__{algorithm}__K{k}"

                _, labels = fit_cluster(
                    Z,
                    algorithm,
                    k,
                    stable_seed(
                        MASTER_SEED,
                        "full",
                        scenario,
                        candidate_id,
                    ),
                )

                metrics = internal_metrics(
                    Z,
                    labels,
                )

                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "representation": representation,
                        "algorithm": algorithm,
                        "k_requested": int(k),
                        **metrics,
                        "eligible_min_cluster_size": bool(
                            metrics["minimum_cluster_size"] >= 5
                        ),
                    }
                )

                labels_by_candidate[candidate_id] = labels.astype(int)

    return (
        pd.DataFrame(rows),
        labels_by_candidate,
    )


def build_shortlist(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    eligible = metrics.loc[metrics["eligible_min_cluster_size"].astype(bool)].copy()

    selections: dict[str, set[str]] = {}

    for (
        representation,
        algorithm,
    ), group in eligible.groupby(
        ["representation", "algorithm"],
        sort=True,
    ):
        rules = {
            "BEST_SILHOUETTE": group.loc[
                group["silhouette"].idxmax(),
                "candidate_id",
            ],
            "BEST_DAVIES_BOULDIN": group.loc[
                group["davies_bouldin"].idxmin(),
                "candidate_id",
            ],
            "BEST_CALINSKI_HARABASZ": group.loc[
                group["calinski_harabasz"].idxmax(),
                "candidate_id",
            ],
        }

        for reason, candidate_id in rules.items():
            selections.setdefault(
                str(candidate_id),
                set(),
            ).add(reason)

    selected_ids = sorted(selections)

    shortlist = eligible.loc[eligible["candidate_id"].isin(selected_ids)].copy()

    shortlist["shortlist_reasons"] = shortlist["candidate_id"].map(
        lambda candidate_id: ";".join(sorted(selections[str(candidate_id)]))
    )

    return shortlist.reset_index(drop=True)


def matched_cluster_jaccards(
    reference: np.ndarray,
    predicted: np.ndarray,
) -> list[float]:
    ref_values = np.unique(reference)
    pred_values = np.unique(predicted)

    contingency = np.zeros(
        (
            len(ref_values),
            len(pred_values),
        ),
        dtype=int,
    )

    for i, ref in enumerate(ref_values):
        for j, pred in enumerate(pred_values):
            contingency[i, j] = int(np.sum((reference == ref) & (predicted == pred)))

    row_ind, col_ind = linear_sum_assignment(-contingency)

    result = []

    for row, col in zip(
        row_ind,
        col_ind,
    ):
        ref = ref_values[row]
        pred = pred_values[col]

        intersection = int(np.sum((reference == ref) & (predicted == pred)))
        union = int(np.sum((reference == ref) | (predicted == pred)))

        result.append(float(intersection / union) if union > 0 else np.nan)

    return result


def stability_for_candidate(
    X: np.ndarray,
    scenario: str,
    candidate_row: pd.Series,
    reference_labels: np.ndarray,
) -> dict[str, Any]:
    candidate_id = str(candidate_row["candidate_id"])
    representation = str(candidate_row["representation"])
    algorithm = str(candidate_row["algorithm"])
    k = int(candidate_row["k_requested"])

    sample_size = int(round(len(X) * STABILITY_SAMPLE_FRACTION))

    ari_values = []
    cluster_jaccards: dict[int, list[float]] = {
        int(cluster): [] for cluster in np.unique(reference_labels)
    }
    min_cluster_sizes = []

    rng = np.random.default_rng(
        stable_seed(
            MASTER_SEED,
            "stability",
            scenario,
            candidate_id,
        )
    )

    for iteration in range(STABILITY_ITERATIONS):
        sample = np.sort(
            rng.choice(
                np.arange(
                    len(X),
                    dtype=int,
                ),
                size=sample_size,
                replace=False,
            )
        )

        Z_sample, _, _ = transform_representation(
            X[sample],
            representation,
        )

        _, labels_sample = fit_cluster(
            Z_sample,
            algorithm,
            k,
            stable_seed(
                MASTER_SEED,
                "stability_fit",
                scenario,
                candidate_id,
                iteration,
            ),
        )

        ref_sample = reference_labels[sample]

        ari_values.append(
            float(
                adjusted_rand_score(
                    ref_sample,
                    labels_sample,
                )
            )
        )

        counts = pd.Series(labels_sample).value_counts()

        min_cluster_sizes.append(int(counts.min()))

        ref_values = np.unique(ref_sample)
        pred_values = np.unique(labels_sample)

        contingency = np.zeros(
            (
                len(ref_values),
                len(pred_values),
            ),
            dtype=int,
        )

        for i, ref in enumerate(ref_values):
            for j, pred in enumerate(pred_values):
                contingency[i, j] = int(
                    np.sum((ref_sample == ref) & (labels_sample == pred))
                )

        row_ind, col_ind = linear_sum_assignment(-contingency)

        for row, col in zip(
            row_ind,
            col_ind,
        ):
            ref = int(ref_values[row])
            pred = pred_values[col]

            intersection = int(np.sum((ref_sample == ref) & (labels_sample == pred)))
            union = int(np.sum((ref_sample == ref) | (labels_sample == pred)))

            if union > 0:
                cluster_jaccards[ref].append(float(intersection / union))

    weakest_cluster_mean = min(
        float(np.mean(values)) for values in cluster_jaccards.values() if values
    )

    return {
        "candidate_id": candidate_id,
        "representation": representation,
        "algorithm": algorithm,
        "k_requested": k,
        "ari_median": float(np.median(ari_values)),
        "ari_p025": float(
            np.quantile(
                ari_values,
                0.025,
            )
        ),
        "ari_p975": float(
            np.quantile(
                ari_values,
                0.975,
            )
        ),
        "clusterwise_jaccard_min_mean": (weakest_cluster_mean),
        "minimum_resample_cluster_size_min": int(min(min_cluster_sizes)),
        "stability_iterations": int(STABILITY_ITERATIONS),
    }


def run_stability(
    X: np.ndarray,
    scenario: str,
    shortlist: pd.DataFrame,
    labels_by_candidate: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []

    for _, candidate in shortlist.iterrows():
        candidate_id = str(candidate["candidate_id"])

        rows.append(
            stability_for_candidate(
                X,
                scenario,
                candidate,
                labels_by_candidate[candidate_id],
            )
        )

    return pd.DataFrame(rows)


def pairwise_partition_equivalent(
    candidate_ids: list[str],
    labels_by_candidate: dict[str, np.ndarray],
    threshold: float = 0.95,
) -> tuple[bool, list[dict[str, Any]]]:
    rows = []

    for i in range(len(candidate_ids)):
        for j in range(
            i + 1,
            len(candidate_ids),
        ):
            left = candidate_ids[i]
            right = candidate_ids[j]

            ari = float(
                adjusted_rand_score(
                    labels_by_candidate[left],
                    labels_by_candidate[right],
                )
            )

            rows.append(
                {
                    "candidate_a": left,
                    "candidate_b": right,
                    "ari": ari,
                    "partition_equivalent_at_0_95": bool(ari >= threshold),
                }
            )

    if len(candidate_ids) <= 1:
        return True, rows

    return all(row["partition_equivalent_at_0_95"] for row in rows), rows


def choose_model(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    labels_by_candidate: dict[str, np.ndarray],
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    diagnostics = stability.merge(
        metrics,
        on=[
            "candidate_id",
            "representation",
            "algorithm",
            "k_requested",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "_stability",
            "_full",
        ),
    )

    diagnostics["resampling_min_cluster_ok"] = (
        diagnostics["minimum_resample_cluster_size_min"] >= 5
    )

    diagnostics["full_data_min_cluster_ok"] = diagnostics["minimum_cluster_size"] >= 5

    diagnostics["selection_eligible"] = (
        diagnostics["resampling_min_cluster_ok"]
        & diagnostics["full_data_min_cluster_ok"]
    )

    eligible = diagnostics.loc[diagnostics["selection_eligible"]].copy()

    if eligible.empty:
        raise RuntimeError(
            "No eligible synthetic candidate " "remains after stability checks."
        )

    best_median = float(eligible["ari_median"].max())

    tied = eligible.loc[
        np.isclose(
            eligible["ari_median"],
            best_median,
            atol=1e-12,
            rtol=0,
        )
    ].copy()

    tied_ids = tied["candidate_id"].astype(str).tolist()

    equivalent, pairwise = pairwise_partition_equivalent(
        tied_ids,
        labels_by_candidate,
        threshold=0.95,
    )

    decision = {
        "maximum_ari_median": best_median,
        "median_tied_candidates": tied_ids,
        "tied_partitions_equivalent": bool(equivalent),
        "tied_pairwise_ari": pairwise,
    }

    if len(tied) == 1:
        selected = tied.iloc[0]
        decision["rule_used"] = "Unique highest median ARI."

    elif equivalent:
        selected = tied.sort_values(
            [
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "k_requested",
            ],
            ascending=[
                False,
                True,
                False,
                True,
            ],
        ).iloc[0]

        decision["rule_used"] = (
            "Median-ARI tie with partition-equivalent "
            "solutions; internal-separation tiebreak."
        )

    else:
        selected = tied.sort_values(
            [
                "ari_p025",
                "clusterwise_jaccard_min_mean",
                "silhouette",
                "k_requested",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        ).iloc[0]

        decision["rule_used"] = (
            "Median-ARI tie with non-equivalent "
            "solutions; lower-tail stability tiebreak."
        )

    return (
        selected,
        decision,
        diagnostics,
    )


def multiclass_severity_cv(
    severity: np.ndarray,
    labels: np.ndarray,
    selected_k: int,
    seed: int,
) -> pd.DataFrame:
    rkf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=20,
        random_state=seed,
    )

    rows = []

    for fold_index, (
        train_idx,
        test_idx,
    ) in enumerate(
        rkf.split(
            severity.reshape(
                -1,
                1,
            ),
            labels,
        ),
        start=1,
    ):
        model = DecisionTreeClassifier(
            max_leaf_nodes=selected_k,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=seed + fold_index,
        )

        model.fit(
            severity[train_idx].reshape(
                -1,
                1,
            ),
            labels[train_idx],
        )

        pred = model.predict(
            severity[test_idx].reshape(
                -1,
                1,
            )
        )

        rows.append(
            {
                "fold_index": fold_index,
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        labels[test_idx],
                        pred,
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        labels[test_idx],
                        pred,
                        average="macro",
                    )
                ),
                "ari": float(
                    adjusted_rand_score(
                        labels[test_idx],
                        pred,
                    )
                ),
                "nmi": float(
                    normalized_mutual_info_score(
                        labels[test_idx],
                        pred,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def severity_audit(
    X: np.ndarray,
    labels: np.ndarray,
    selected_k: int,
    scenario: str,
) -> dict[str, Any]:
    severity = X.mean(axis=1)

    if selected_k == 2:
        observed_binary_labels = set(int(value) for value in np.unique(labels))
        if observed_binary_labels != {1, 2}:
            raise ValueError(
                "Inherited binary severity baseline requires canonical "
                f"profile labels {{1,2}}, got {sorted(observed_binary_labels)}."
            )
        folds, _ = severity_cross_validation(
            severity,
            labels,
            splits=5,
            repeats=20,
            seed=stable_seed(
                MASTER_SEED,
                "severity_binary",
                scenario,
            ),
        )

        ba_median = float(folds["test_balanced_accuracy"].median())
        ari_median = float(folds["test_ari"].median())
        macro_f1_median = float(folds["test_macro_f1"].median())
        auc_median = float(folds["test_roc_auc"].median())
        nmi_median = float(folds["test_nmi"].median())

        method = "INHERITED_BINARY_THRESHOLD"

    else:
        folds = multiclass_severity_cv(
            severity,
            labels,
            selected_k,
            stable_seed(
                MASTER_SEED,
                "severity_multiclass",
                scenario,
            ),
        )

        ba_median = float(folds["balanced_accuracy"].median())
        ari_median = float(folds["ari"].median())
        macro_f1_median = float(folds["macro_f1"].median())
        auc_median = np.nan
        nmi_median = float(folds["nmi"].median())

        method = "PROSPECTIVE_1D_TREE_MULTICLASS"

    matched_pairs = 0

    for i in range(len(severity)):
        for j in range(
            i + 1,
            len(severity),
        ):
            if labels[i] == labels[j]:
                continue

            if abs(severity[i] - severity[j]) <= 0.05:
                matched_pairs += 1

    flag = bool(ba_median >= 0.90 and ari_median >= 0.80)

    return {
        "method": method,
        "balanced_accuracy_median": ba_median,
        "macro_f1_median": macro_f1_median,
        "roc_auc_median": (None if not np.isfinite(auc_median) else auc_median),
        "ari_median": ari_median,
        "nmi_median": nmi_median,
        "matched_severity_opposite_cluster_pairs": int(matched_pairs),
        "severity_nearly_reconstructs_profiles": flag,
    }


def cramers_v_bias_corrected(
    table: np.ndarray,
) -> float:
    chi2, _, _, _ = chi2_contingency(
        table,
        correction=False,
    )

    n = table.sum()
    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    r, k = table.shape

    phi2corr = max(
        0.0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1),
    )

    rcorr = r - ((r - 1) ** 2) / (n - 1)

    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(
        kcorr - 1,
        rcorr - 1,
    )

    if denominator <= 0:
        return 0.0

    return float(np.sqrt(phi2corr / denominator))


def governance_permutation(
    governance: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    table = pd.crosstab(
        pd.Series(
            labels,
            name="cluster",
        ),
        pd.Series(
            governance,
            name="governance",
        ),
    ).to_numpy()

    observed = cramers_v_bias_corrected(table)

    rng = np.random.default_rng(seed)

    null = np.empty(
        permutations,
        dtype=float,
    )

    for i in range(permutations):
        perm = rng.permutation(governance)

        table_perm = pd.crosstab(
            pd.Series(
                labels,
                name="cluster",
            ),
            pd.Series(
                perm,
                name="governance",
            ),
        ).to_numpy()

        null[i] = cramers_v_bias_corrected(table_perm)

    p = float((1 + np.sum(null >= observed)) / (permutations + 1))

    return observed, p


def governance_cv(
    governance: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rkf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=20,
        random_state=seed,
    )

    rows = []

    for fold_index, (
        train_idx,
        test_idx,
    ) in enumerate(
        rkf.split(
            governance.reshape(
                -1,
                1,
            ),
            labels,
        ),
        start=1,
    ):
        mapping = {}

        for value in np.unique(governance[train_idx]):
            mask = governance[train_idx] == value

            mapping[value] = int(
                pd.Series(labels[train_idx][mask]).value_counts().idxmax()
            )

        global_majority = int(pd.Series(labels[train_idx]).value_counts().idxmax())

        pred = np.array(
            [
                mapping.get(
                    value,
                    global_majority,
                )
                for value in governance[test_idx]
            ],
            dtype=int,
        )

        rows.append(
            {
                "fold_index": fold_index,
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        labels[test_idx],
                        pred,
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        labels[test_idx],
                        pred,
                        average="macro",
                    )
                ),
                "ari": float(
                    adjusted_rand_score(
                        labels[test_idx],
                        pred,
                    )
                ),
                "nmi": float(
                    normalized_mutual_info_score(
                        labels[test_idx],
                        pred,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def governance_audit(
    truth: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    scenario: str,
) -> dict[str, Any]:
    if "governance_type" not in truth.columns or truth["governance_type"].isna().all():
        return {
            "status": "NOT_APPLICABLE_NO_GOVERNANCE_VARIABLE",
            "strong_governance_association": False,
            "governance_nearly_reconstructs_profiles": False,
        }

    merged = truth[
        [
            "institution_id",
            "governance_type",
        ]
    ].merge(
        labels_by_id,
        on="institution_id",
        validate="one_to_one",
    )

    governance = merged["governance_type"].astype(str).to_numpy()

    labels = merged["cluster_id"].astype(int).to_numpy()

    observed_v, p = governance_permutation(
        governance,
        labels,
        permutations=2000,
        seed=stable_seed(
            MASTER_SEED,
            "governance_perm",
            scenario,
        ),
    )

    cv = governance_cv(
        governance,
        labels,
        stable_seed(
            MASTER_SEED,
            "governance_cv",
            scenario,
        ),
    )

    ba_median = float(cv["balanced_accuracy"].median())
    ari_median = float(cv["ari"].median())

    return {
        "status": "COMPLETED",
        "bias_corrected_cramers_v": observed_v,
        "permutation_p": p,
        "balanced_accuracy_median": ba_median,
        "ari_median": ari_median,
        "macro_f1_median": float(cv["macro_f1"].median()),
        "nmi_median": float(cv["nmi"].median()),
        "strong_governance_association": bool(observed_v >= 0.50 and p < 0.05),
        "governance_nearly_reconstructs_profiles": bool(
            ba_median >= 0.90 and ari_median >= 0.80
        ),
    }


def align_member_labels(
    native_sample_labels: np.ndarray,
    reference_sample_labels: np.ndarray,
) -> dict[int, int]:
    native_values = np.unique(native_sample_labels)
    reference_values = np.unique(reference_sample_labels)

    contingency = np.zeros(
        (
            len(native_values),
            len(reference_values),
        ),
        dtype=int,
    )

    for i, native in enumerate(native_values):
        for j, reference in enumerate(reference_values):
            contingency[i, j] = int(
                np.sum(
                    (native_sample_labels == native)
                    & (reference_sample_labels == reference)
                )
            )

    row_ind, col_ind = linear_sum_assignment(-contingency)

    return {
        int(native_values[row]): int(reference_values[col])
        for row, col in zip(
            row_ind,
            col_ind,
        )
    }


def ward_centroid_predict(
    Z_train: np.ndarray,
    labels_train: np.ndarray,
    Z_new: np.ndarray,
) -> np.ndarray:
    clusters = np.unique(labels_train)
    centroids = np.vstack(
        [Z_train[labels_train == cluster].mean(axis=0) for cluster in clusters]
    )

    distances = np.sqrt(
        (
            (
                Z_new[
                    :,
                    None,
                    :,
                ]
                - centroids[
                    None,
                    :,
                    :,
                ]
            )
            ** 2
        ).sum(axis=2)
    )

    nearest = distances.argmin(axis=1)

    return clusters[nearest]


def binary_uncertainty_ensemble(
    X: np.ndarray,
    reference_labels: np.ndarray,
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if set(np.unique(reference_labels)) != {1, 2}:
        raise ValueError(
            "Binary uncertainty requires " "reference labels exactly {1,2}."
        )

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

    member_count = 0

    for (
        family,
        representation,
        algorithm,
    ) in family_specs:
        rng = np.random.default_rng(
            stable_seed(
                MASTER_SEED,
                "ensemble_family",
                scenario,
                family,
            )
        )

        for member in range(ENSEMBLE_MEMBERS_PER_FAMILY):
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
                    size=ENSEMBLE_FEATURE_COUNT,
                    replace=False,
                )
            )

            X_sample = X[sample][:, feature_idx]

            X_oob = X[oob][:, feature_idx]

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
                    "ensemble_fit",
                    scenario,
                    family,
                    member,
                ),
            )

            mapping = align_member_labels(
                native_labels,
                reference_labels[sample],
            )

            Z_oob = apply_representation(
                X_oob,
                scaler,
                pca,
            )

            if algorithm == "KMEANS":
                native_oob = model.predict(Z_oob)

            else:
                native_oob = ward_centroid_predict(
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

            member_count += 1

    rows = []

    for i in range(n):
        values = np.array(
            oob_predictions[i],
            dtype=int,
        )

        if len(values) == 0:
            raise RuntimeError(f"No OOB predictions for row {i}.")

        p1 = float(np.mean(values == 1))
        p2 = float(np.mean(values == 2))

        reference = int(reference_labels[i])
        reference_probability = p1 if reference == 1 else p2

        probs_nonzero = np.array(
            [
                p1,
                p2,
            ],
            dtype=float,
        )
        probs_nonzero = probs_nonzero[probs_nonzero > 0]

        entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)) / np.log(2.0))

        margin = float(2.0 * abs(p1 - 0.5))

        family_reference_probs = []

        for (
            family,
            _,
            _,
        ) in family_specs:
            family_values = np.array(
                family_oob_predictions[family][i],
                dtype=int,
            )

            if len(family_values) == 0:
                family_reference_probs.append(np.nan)
                continue

            family_reference_probs.append(float(np.mean(family_values == reference)))

        finite_family = [
            value for value in family_reference_probs if np.isfinite(value)
        ]

        if len(finite_family) != 4:
            raise RuntimeError(
                f"Missing family OOB predictions "
                f"for row {i}: "
                f"{family_reference_probs}"
            )

        family_consistency = float(min(finite_family))

        if reference_probability >= 0.90 and family_consistency >= 0.80:
            certainty_class = "CORE"

        elif reference_probability < 0.75 or family_consistency < 0.60:
            certainty_class = "BOUNDARY"

        else:
            certainty_class = "HALO"

        rows.append(
            {
                "row_index": i,
                "reference_profile": reference,
                "n_oob_predictions": int(len(values)),
                "profile_1_probability": p1,
                "profile_2_probability": p2,
                "reference_profile_probability": reference_probability,
                "normalized_entropy": entropy,
                "membership_margin": margin,
                "family_consistency": family_consistency,
                "certainty_class": certainty_class,
            }
        )

    certainty = pd.DataFrame(rows)

    report = {
        "ensemble_members": int(member_count),
        "members_per_family": int(ENSEMBLE_MEMBERS_PER_FAMILY),
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

    return certainty, report


def remap_selected_labels_to_profiles(
    raw_labels: np.ndarray,
) -> np.ndarray:
    values = sorted(int(value) for value in np.unique(raw_labels))

    if len(values) != 2:
        raise ValueError("Expected exactly two selected clusters.")

    mapping = {
        values[0]: 1,
        values[1]: 2,
    }

    return np.array(
        [mapping[int(value)] for value in raw_labels],
        dtype=int,
    )


def truth_evaluation(
    scenario: str,
    bundle,
    selected_labels: np.ndarray,
    selected_k: int,
    certainty: pd.DataFrame | None,
    stable_partition_claim: bool,
    configurational_claim: bool,
) -> dict[str, Any]:
    truth = bundle.truth.copy()

    result: dict[str, Any] = {
        "scenario": scenario,
        "selected_k": int(selected_k),
    }

    if truth["true_profile"].notna().all():
        true_profile = truth["true_profile"].astype(int).to_numpy()

        result["ari_vs_true_profile"] = float(
            adjusted_rand_score(
                true_profile,
                selected_labels,
            )
        )

        result["nmi_vs_true_profile"] = float(
            normalized_mutual_info_score(
                true_profile,
                selected_labels,
            )
        )

    else:
        result["ari_vs_true_profile"] = None
        result["nmi_vs_true_profile"] = None

    if scenario == "S2_CORE_BOUNDARY" and certainty is not None:
        y_boundary = truth["true_boundary"].astype(bool).astype(int).to_numpy()

        entropy = certainty["normalized_entropy"].to_numpy(dtype=float)

        result["boundary_uncertainty_auc"] = float(
            roc_auc_score(
                y_boundary,
                entropy,
            )
        )

        result["median_entropy_boundary"] = float(np.median(entropy[y_boundary == 1]))

        result["median_entropy_core"] = float(np.median(entropy[y_boundary == 0]))

    else:
        result["boundary_uncertainty_auc"] = None

    if scenario in {
        "S4_SEVERITY_CONTINUUM",
        "S5_GOVERNANCE_CONFOUNDED",
    }:
        result["false_configurational_profile_claim"] = bool(configurational_claim)
    else:
        result["false_configurational_profile_claim"] = None

    if scenario == "S6_NO_CLUSTER_NULL":
        result["false_stable_profile_claim"] = bool(stable_partition_claim)
    else:
        result["false_stable_profile_claim"] = None

    return result


def run_scenario(
    scenario: str,
) -> dict[str, Any]:
    bundle = generate_scenario(
        scenario_id=scenario,
        replicate=SMOKE_REPLICATE,
        master_seed=MASTER_SEED,
    )

    scenario_dir = OUTPUT_ROOT / scenario
    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = bundle.data[FEATURE_NAMES].to_numpy(dtype=float)

    metrics, labels_by_candidate = full_candidate_search(
        X,
        scenario,
    )

    shortlist = build_shortlist(metrics)

    stability = run_stability(
        X,
        scenario,
        shortlist,
        labels_by_candidate,
    )

    selected, decision, diagnostics = choose_model(
        metrics,
        stability,
        labels_by_candidate,
    )

    selected_id = str(selected["candidate_id"])
    selected_k = int(selected["k_requested"])

    raw_selected_labels = labels_by_candidate[selected_id]

    labels_out = pd.DataFrame(
        {
            "institution_id": bundle.data["institution_id"].astype(str),
            "cluster_id": (raw_selected_labels.astype(int)),
        }
    )

    stable_partition_claim = bool(
        selected["selection_eligible"] and float(selected["ari_median"]) >= 0.70
    )

    # IMPORTANT: sklearn clustering emits 0-based labels. The inherited
    # binary severity-null code from the real CIPHER pipeline assumes
    # PROFILE_1=1 and PROFILE_2=2. Canonicalize *before* calling it.
    if selected_k == 2:
        severity_labels = remap_selected_labels_to_profiles(raw_selected_labels)
    else:
        severity_labels = raw_selected_labels.astype(int)

    severity = severity_audit(
        X,
        severity_labels,
        selected_k,
        scenario,
    )

    governance = governance_audit(
        bundle.truth,
        labels_out,
        scenario,
    )

    configurational_claim = bool(
        stable_partition_claim
        and not severity["severity_nearly_reconstructs_profiles"]
        and not governance["strong_governance_association"]
        and not governance["governance_nearly_reconstructs_profiles"]
    )

    certainty = None
    uncertainty_report: dict[str, Any]

    if selected_k == 2:
        reference_profiles = severity_labels.copy()

        certainty, uncertainty_report = binary_uncertainty_ensemble(
            X,
            reference_profiles,
            scenario,
        )

        certainty.insert(
            0,
            "institution_id",
            bundle.data["institution_id"].astype(str).to_numpy(),
        )

        uncertainty_report["status"] = "COMPLETED_K2"

    else:
        uncertainty_report = {
            "status": "NOT_APPLICABLE_SELECTED_K_NOT_2",
            "selected_k": int(selected_k),
        }

    # Save all model/pipeline outputs before truth evaluation.
    metrics.to_csv(
        scenario_dir / "candidate_metrics.csv",
        index=False,
    )
    shortlist.to_csv(
        scenario_dir / "stability_shortlist.csv",
        index=False,
    )
    stability.to_csv(
        scenario_dir / "stability_summary.csv",
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

    if certainty is not None:
        certainty.to_csv(
            scenario_dir / "membership_certainty.csv",
            index=False,
        )

    pretruth_report = {
        "scenario": scenario,
        "smoke_replicate": int(SMOKE_REPLICATE),
        "selected_candidate": selected_id,
        "selected_representation": str(selected["representation"]),
        "selected_algorithm": str(selected["algorithm"]),
        "selected_k": selected_k,
        "selected_silhouette": float(selected["silhouette"]),
        "selected_ari_median": float(selected["ari_median"]),
        "selected_ari_p025": float(selected["ari_p025"]),
        "selected_weakest_cluster_mean_jaccard": float(
            selected["clusterwise_jaccard_min_mean"]
        ),
        "selected_minimum_cluster_size": int(selected["minimum_cluster_size"]),
        "selected_minimum_resample_cluster_size": int(
            selected["minimum_resample_cluster_size_min"]
        ),
        "selection_decision": decision,
        "stable_partition_claim": stable_partition_claim,
        "severity": severity,
        "governance": governance,
        "configurational_profile_claim": configurational_claim,
        "uncertainty": uncertainty_report,
    }

    (scenario_dir / "pipeline_pretruth_report.json").write_text(
        json.dumps(
            pretruth_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    truth_eval = truth_evaluation(
        scenario,
        bundle,
        raw_selected_labels,
        selected_k,
        certainty,
        stable_partition_claim,
        configurational_claim,
    )

    (scenario_dir / "truth_evaluation_smoke_only.json").write_text(
        json.dumps(
            truth_eval,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checks = {
        "candidate_grid_complete": len(metrics) == 20,
        "shortlist_nonempty": len(shortlist) > 0,
        "selected_candidate_exists": selected_id in labels_by_candidate,
        "selected_k_in_frozen_range": selected_k in K_VALUES,
        "binary_severity_labels_canonical_when_k2": (
            selected_k != 2
            or set(int(value) for value in np.unique(severity_labels)) == {1, 2}
        ),
        "selection_eligible": bool(selected["selection_eligible"]),
        "truth_rows_match_data": len(bundle.truth) == len(bundle.data) == 80,
        "uncertainty_semantics_respected": (
            (
                selected_k == 2
                and certainty is not None
                and len(certainty) == 80
                and int(certainty["n_oob_predictions"].min()) >= 20
            )
            or (
                selected_k > 2
                and certainty is None
                and uncertainty_report["status"] == "NOT_APPLICABLE_SELECTED_K_NOT_2"
            )
        ),
        "governance_only_applied_when_present": (
            (
                scenario == "S5_GOVERNANCE_CONFOUNDED"
                and governance["status"] == "COMPLETED"
            )
            or (
                scenario != "S5_GOVERNANCE_CONFOUNDED"
                and governance["status"] == "NOT_APPLICABLE_NO_GOVERNANCE_VARIABLE"
            )
        ),
    }

    return {
        "scenario": scenario,
        "pretruth": pretruth_report,
        "truth": truth_eval,
        "checks": checks,
    }


def main() -> None:
    # These warnings exposed a real label-semantics mismatch in smoke v1.
    # In v2 they are promoted to errors so the bug cannot silently recur.
    warnings.filterwarnings(
        "error",
        message="y_pred contains classes not in y_true",
    )
    warnings.filterwarnings(
        "error",
        message="Only one class is present in y_true.*",
    )

    evaluator = load_json(EVALUATOR_PATH)
    generator = load_json(GENERATOR_FREEZE_PATH)
    stage8b_audit = load_json(STAGE8B_AUDIT_PATH)

    prechecks = {
        "evaluator_v2_passed": (
            evaluator.get("gate_status") == "PASS_STAGE_8C1_MULTICLASS_AMENDMENT"
        ),
        "generator_implementation_frozen": (
            generator.get("gate_status") == "FROZEN_STAGE8_GENERATOR_IMPLEMENTATION_V1"
        ),
        "generator_smoke_passed": (
            stage8b_audit.get("gate_status") == "PASS_STAGE_8B_GENERATOR_SMOKE_AUDIT"
        ),
        "performance_smoke_not_preexisting": (not OUTPUT_ROOT.exists()),
    }

    print("\n=== CIPHER STAGE 8D1 v2 — CORE PIPELINE PERFORMANCE SMOKE ===\n")
    print(
        "NON-OFFICIAL smoke only. "
        "These six replicates do not count toward Stage 8 official results."
    )
    print(
        "This stage tests discovery, stability, falsification, and "
        "binary uncertainty mechanics. Counterfactual reachability is NOT run yet."
    )

    print("\n=== PRECHECKS ===\n")
    for name, passed in prechecks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(prechecks.values()):
        print("\nGATE STATUS: FAIL_STAGE_8D1_PRECHECK")
        raise SystemExit(1)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    results = []

    for scenario in SCENARIOS:
        print(f"\n--- Running {scenario} ---")

        result = run_scenario(scenario)
        results.append(result)

        pre = result["pretruth"]
        truth = result["truth"]

        print(
            "selected:",
            pre["selected_candidate"],
        )
        print(
            "k:",
            pre["selected_k"],
            "| median stability ARI:",
            f"{pre['selected_ari_median']:.4f}",
            "| silhouette:",
            f"{pre['selected_silhouette']:.4f}",
        )
        print(
            "stable claim:",
            pre["stable_partition_claim"],
            "| severity reconstructs:",
            pre["severity"]["severity_nearly_reconstructs_profiles"],
            "| governance strong/reconstructs:",
            pre["governance"]["strong_governance_association"],
            "/",
            pre["governance"]["governance_nearly_reconstructs_profiles"],
            "| configurational claim:",
            pre["configurational_profile_claim"],
        )
        print(
            "uncertainty:",
            pre["uncertainty"]["status"],
        )

        if truth["ari_vs_true_profile"] is not None:
            print(
                "SMOKE truth ARI:",
                f"{truth['ari_vs_true_profile']:.4f}",
            )

        if truth["boundary_uncertainty_auc"] is not None:
            print(
                "SMOKE boundary entropy AUC:",
                f"{truth['boundary_uncertainty_auc']:.4f}",
            )

        if truth["false_configurational_profile_claim"] is not None:
            print(
                "SMOKE false configurational claim:",
                truth["false_configurational_profile_claim"],
            )

        if truth["false_stable_profile_claim"] is not None:
            print(
                "SMOKE false stable claim:",
                truth["false_stable_profile_claim"],
            )

    all_checks = {
        "prechecks_pass": all(prechecks.values()),
        "six_scenarios_completed": len(results) == 6,
        "all_scenario_integrity_checks_pass": all(
            all(result["checks"].values()) for result in results
        ),
        "official_result_directory_untouched": not (
            ROOT / "cipher" / "outputs" / "synthetic" / "official"
        ).exists(),
    }

    summary_rows = []

    for result in results:
        pre = result["pretruth"]
        truth = result["truth"]

        summary_rows.append(
            {
                "scenario": result["scenario"],
                "selected_candidate": pre["selected_candidate"],
                "selected_k": pre["selected_k"],
                "stability_ari_median": pre["selected_ari_median"],
                "silhouette": pre["selected_silhouette"],
                "stable_partition_claim": pre["stable_partition_claim"],
                "severity_reconstructs": pre["severity"][
                    "severity_nearly_reconstructs_profiles"
                ],
                "governance_strong": pre["governance"]["strong_governance_association"],
                "governance_reconstructs": pre["governance"][
                    "governance_nearly_reconstructs_profiles"
                ],
                "configurational_claim": pre["configurational_profile_claim"],
                "uncertainty_status": pre["uncertainty"]["status"],
                "truth_ari_smoke_only": truth["ari_vs_true_profile"],
                "boundary_auc_smoke_only": truth["boundary_uncertainty_auc"],
                "false_configurational_claim_smoke_only": truth[
                    "false_configurational_profile_claim"
                ],
                "false_stable_claim_smoke_only": truth["false_stable_profile_claim"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_ROOT / "stage8d1_v2_smoke_summary.csv",
        index=False,
    )

    report = {
        "status": (
            "PASS_STAGE_8D1_V2_CORE_PIPELINE_SMOKE"
            if all(all_checks.values())
            else "FAIL_STAGE_8D1_V2_CORE_PIPELINE_SMOKE"
        ),
        "scientific_interpretation_allowed": False,
        "smoke_replicate": int(SMOKE_REPLICATE),
        "stability_iterations": int(STABILITY_ITERATIONS),
        "uncertainty_ensemble_members_when_k2": int(4 * ENSEMBLE_MEMBERS_PER_FAMILY),
        "prechecks": prechecks,
        "scenario_checks": {result["scenario"]: result["checks"] for result in results},
        "global_checks": all_checks,
        "summary": summary_rows,
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

    print("\n=== NON-OFFICIAL SMOKE SUMMARY ===\n")
    print(summary.to_string(index=False))

    print("\n=== TECHNICAL GATE CHECKS ===\n")

    for name, passed in all_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    for result in results:
        failed = [
            name
            for (
                name,
                passed,
            ) in result["checks"].items()
            if not passed
        ]

        print(
            f"{result['scenario']}:",
            "PASS" if not failed else f"FAIL {failed}",
        )

    print(f"\nGATE STATUS: {report['status']}")

    if report["status"] == "PASS_STAGE_8D1_V2_CORE_PIPELINE_SMOKE":
        print(
            "Core pipeline mechanics pass. "
            "Do NOT treat smoke ARI/AUC/false-claim values as scientific results."
        )
        print(
            "Next step after review: Stage 8D2 counterfactual/reachability smoke "
            "for the structured k=2-eligible scenarios."
        )
    else:
        print(
            "Do not proceed. Fix only the failed technical mechanic; "
            "do not tune scientific thresholds to improve smoke performance."
        )


if __name__ == "__main__":
    main()
