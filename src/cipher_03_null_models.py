from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRICTA_CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
PRIMARY_MATRIX_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
)
FINAL_LABELS_PATH = (
    PROJECT_ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
CERTAINTY_PATH = (
    PROJECT_ROOT / "cipher" / "outputs" / "certainty" / "institution_certainty.csv"
)
CIPHER_CONFIG_PATH = (
    PROJECT_ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"
)

OUTPUT_DIR = PROJECT_ROOT / "cipher" / "outputs" / "null_models"
AUDIT_DIR = PROJECT_ROOT / "cipher" / "outputs" / "audit"

GOVERNANCE_CANDIDATES = [
    "governance_type",
    "institution_type",
    "management_type",
    "ownership_type",
    "sector_type",
    "governance",
    "type_of_institution",
]

PROFILE_1 = 1
PROFILE_2 = 2


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_governance_column(df: pd.DataFrame) -> str | None:
    normalized = {c.strip().lower(): c for c in df.columns}
    for candidate in GOVERNANCE_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def best_threshold_from_training(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, int, float]:
    unique_scores = np.unique(scores)

    if len(unique_scores) == 1:
        thresholds = np.array([unique_scores[0]], dtype=float)
    else:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        thresholds = np.concatenate(
            [
                [unique_scores[0] - 1e-9],
                midpoints,
                [unique_scores[-1] + 1e-9],
            ]
        )

    best = None

    for direction in (1, -1):
        for threshold in thresholds:
            if direction == 1:
                predicted = np.where(scores >= threshold, PROFILE_2, PROFILE_1)
            else:
                predicted = np.where(scores >= threshold, PROFILE_1, PROFILE_2)

            ba = balanced_accuracy_score(labels, predicted)

            key = (
                ba,
                -abs(threshold - np.median(scores)),
                direction,
            )
            if best is None or key > best[0]:
                best = (
                    key,
                    float(threshold),
                    int(direction),
                    float(ba),
                )

    assert best is not None
    return best[1], best[2], best[3]


def severity_cross_validation(
    severity: np.ndarray,
    labels: np.ndarray,
    splits: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rkf = RepeatedStratifiedKFold(
        n_splits=splits,
        n_repeats=repeats,
        random_state=seed,
    )

    fold_rows = []
    heldout_rows = []

    for fold_index, (train_idx, test_idx) in enumerate(
        rkf.split(severity.reshape(-1, 1), labels),
        start=1,
    ):
        threshold, direction, train_ba = best_threshold_from_training(
            severity[train_idx],
            labels[train_idx],
        )

        if direction == 1:
            test_pred = np.where(
                severity[test_idx] >= threshold,
                PROFILE_2,
                PROFILE_1,
            )
        else:
            test_pred = np.where(
                severity[test_idx] >= threshold,
                PROFILE_1,
                PROFILE_2,
            )

        test_ba = balanced_accuracy_score(labels[test_idx], test_pred)
        test_f1 = f1_score(
            labels[test_idx],
            test_pred,
            average="macro",
        )
        test_ari = adjusted_rand_score(labels[test_idx], test_pred)
        test_nmi = normalized_mutual_info_score(labels[test_idx], test_pred)

        y_binary = (labels[test_idx] == PROFILE_2).astype(int)
        score_direction = severity[test_idx] if direction == 1 else -severity[test_idx]

        try:
            test_auc = roc_auc_score(y_binary, score_direction)
        except ValueError:
            test_auc = np.nan

        fold_rows.append(
            {
                "fold_index": fold_index,
                "threshold": threshold,
                "direction": direction,
                "train_balanced_accuracy": train_ba,
                "test_balanced_accuracy": test_ba,
                "test_macro_f1": test_f1,
                "test_roc_auc": test_auc,
                "test_ari": test_ari,
                "test_nmi": test_nmi,
            }
        )

        for local_pos, idx in enumerate(test_idx):
            heldout_rows.append(
                {
                    "fold_index": fold_index,
                    "row_index": int(idx),
                    "true_profile": int(labels[idx]),
                    "severity_score": float(severity[idx]),
                    "predicted_profile": int(test_pred[local_pos]),
                    "correct": int(test_pred[local_pos] == labels[idx]),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(heldout_rows)


def cramers_v(table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    r, k = table.shape

    if n <= 1 or min(r - 1, k - 1) <= 0:
        return np.nan

    phi2 = chi2 / n
    phi2corr = max(
        0.0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1),
    )
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min(kcorr - 1, rcorr - 1)

    if denominator <= 0:
        return 0.0

    return float(np.sqrt(phi2corr / denominator))


def governance_permutation_test(
    governance: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    observed_table = pd.crosstab(
        pd.Series(labels, name="profile"),
        pd.Series(governance, name="governance"),
    ).to_numpy()

    observed_v = cramers_v(observed_table)

    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)

    for i in range(permutations):
        permuted = rng.permutation(labels)
        table = pd.crosstab(
            pd.Series(permuted, name="profile"),
            pd.Series(governance, name="governance"),
        ).to_numpy()
        null_values[i] = cramers_v(table)

    p_value = (1 + np.sum(null_values >= observed_v)) / (permutations + 1)

    return observed_v, float(p_value)


def governance_only_cv(
    governance: np.ndarray,
    labels: np.ndarray,
    splits: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rkf = RepeatedStratifiedKFold(
        n_splits=splits,
        n_repeats=repeats,
        random_state=seed,
    )

    rows = []

    for fold_index, (train_idx, test_idx) in enumerate(
        rkf.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        train_df = pd.DataFrame(
            {
                "governance": governance[train_idx],
                "label": labels[train_idx],
            }
        )

        mapping = (
            train_df.groupby("governance")["label"]
            .agg(lambda s: int(s.value_counts().idxmax()))
            .to_dict()
        )

        global_majority = int(pd.Series(labels[train_idx]).value_counts().idxmax())

        pred = np.array(
            [mapping.get(value, global_majority) for value in governance[test_idx]],
            dtype=int,
        )

        rows.append(
            {
                "fold_index": fold_index,
                "balanced_accuracy": balanced_accuracy_score(
                    labels[test_idx],
                    pred,
                ),
                "macro_f1": f1_score(
                    labels[test_idx],
                    pred,
                    average="macro",
                ),
                "ari": adjusted_rand_score(
                    labels[test_idx],
                    pred,
                ),
                "nmi": normalized_mutual_info_score(
                    labels[test_idx],
                    pred,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    fricta_config = load_json(FRICTA_CONFIG_PATH)
    cipher_config = load_json(CIPHER_CONFIG_PATH)

    id_column = fricta_config["id_column"]
    features = fricta_config["primary_features"]

    primary = pd.read_csv(PRIMARY_MATRIX_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)[[id_column, "cluster_id"]]
    certainty = pd.read_csv(CERTAINTY_PATH)

    primary[id_column] = primary[id_column].astype(str)
    labels[id_column] = labels[id_column].astype(str)
    certainty["institution_id"] = certainty["institution_id"].astype(str)

    data = (
        primary[[id_column] + features]
        .merge(
            labels,
            on=id_column,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            certainty[
                [
                    "institution_id",
                    "certainty_class",
                    "reference_profile_probability",
                ]
            ],
            left_on=id_column,
            right_on="institution_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 institutions, found {len(data)}.")

    X = data[features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError("Primary feature matrix contains missing values.")

    y = data["cluster_id"].astype(int).to_numpy()
    severity = X.mean(axis=1).to_numpy(dtype=float)

    splits = int(cipher_config["null_models"]["severity_cv_splits"])
    repeats = int(cipher_config["null_models"]["severity_cv_repeats"])
    tolerance = float(cipher_config["null_models"]["matched_severity_tolerance"])
    permutations = int(cipher_config["null_models"]["governance_permutations"])
    seed = int(cipher_config["random_seed"])

    severity_folds, heldout = severity_cross_validation(
        severity,
        y,
        splits=splits,
        repeats=repeats,
        seed=seed,
    )

    severity_folds.to_csv(
        OUTPUT_DIR / "severity_null_cv_folds.csv",
        index=False,
    )
    heldout.to_csv(
        OUTPUT_DIR / "severity_null_heldout_predictions.csv",
        index=False,
    )

    severity_summary = {
        "balanced_accuracy_median": float(
            severity_folds["test_balanced_accuracy"].median()
        ),
        "balanced_accuracy_q025": float(
            severity_folds["test_balanced_accuracy"].quantile(0.025)
        ),
        "balanced_accuracy_q975": float(
            severity_folds["test_balanced_accuracy"].quantile(0.975)
        ),
        "macro_f1_median": float(severity_folds["test_macro_f1"].median()),
        "roc_auc_median": float(severity_folds["test_roc_auc"].median()),
        "ari_median": float(severity_folds["test_ari"].median()),
        "nmi_median": float(severity_folds["test_nmi"].median()),
    }

    matched_pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if y[i] == y[j]:
                continue

            gap = abs(severity[i] - severity[j])
            if gap <= tolerance:
                feature_l1 = float(
                    np.abs(
                        X.iloc[i].to_numpy(dtype=float)
                        - X.iloc[j].to_numpy(dtype=float)
                    ).sum()
                )
                matched_pairs.append(
                    {
                        "institution_a": str(data.iloc[i][id_column]),
                        "profile_a": int(y[i]),
                        "severity_a": float(severity[i]),
                        "institution_b": str(data.iloc[j][id_column]),
                        "profile_b": int(y[j]),
                        "severity_b": float(severity[j]),
                        "severity_gap": float(gap),
                        "feature_l1_distance": feature_l1,
                    }
                )

    matched_pairs_df = pd.DataFrame(matched_pairs)
    if len(matched_pairs_df):
        matched_pairs_df = matched_pairs_df.sort_values(
            ["severity_gap", "feature_l1_distance"],
            ascending=[True, False],
        )

    matched_pairs_df.to_csv(
        OUTPUT_DIR / "matched_severity_opposite_profile_pairs.csv",
        index=False,
    )

    governance_column = find_governance_column(primary)

    governance_status = "NOT_FOUND_IN_PRIMARY_MATRIX"
    governance_summary = {}
    governance_contingency = pd.DataFrame()
    governance_cv = pd.DataFrame()

    if governance_column is not None:
        governance_data = primary[[id_column, governance_column]].merge(
            labels,
            on=id_column,
            how="inner",
            validate="one_to_one",
        )

        governance_data[governance_column] = (
            governance_data[governance_column].astype(str).str.strip()
        )

        governance = governance_data[governance_column].to_numpy()
        gov_y = governance_data["cluster_id"].astype(int).to_numpy()

        governance_contingency = pd.crosstab(
            governance_data["cluster_id"],
            governance_data[governance_column],
        )
        governance_contingency.to_csv(OUTPUT_DIR / "governance_contingency.csv")

        observed_v, permutation_p = governance_permutation_test(
            governance,
            gov_y,
            permutations=permutations,
            seed=seed + 5000,
        )

        governance_cv = governance_only_cv(
            governance,
            gov_y,
            splits=splits,
            repeats=repeats,
            seed=seed + 6000,
        )
        governance_cv.to_csv(
            OUTPUT_DIR / "governance_only_cv_folds.csv",
            index=False,
        )

        governance_summary = {
            "column": governance_column,
            "cramers_v": observed_v,
            "permutation_p": permutation_p,
            "balanced_accuracy_median": float(
                governance_cv["balanced_accuracy"].median()
            ),
            "macro_f1_median": float(governance_cv["macro_f1"].median()),
            "ari_median": float(governance_cv["ari"].median()),
            "nmi_median": float(governance_cv["nmi"].median()),
        }
        governance_status = "COMPLETED"

    report = {
        "severity_null": severity_summary,
        "matched_opposite_profile_pairs_within_tolerance": int(len(matched_pairs_df)),
        "matched_severity_tolerance": tolerance,
        "governance_status": governance_status,
        "governance": governance_summary,
        "interpretation_flags": {
            "severity_nearly_reconstructs_profiles": bool(
                severity_summary["balanced_accuracy_median"] >= 0.90
                and severity_summary["ari_median"] >= 0.80
            ),
            "matched_severity_opposite_profile_pairs_exist": bool(
                len(matched_pairs_df) > 0
            ),
        },
    }

    (OUTPUT_DIR / "null_model_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 3 — SIMPLER-EXPLANATION FALSIFICATION ===\n")

    print("=== SEVERITY NULL ===\n")
    print(
        "Held-out balanced accuracy:",
        f"q025={severity_summary['balanced_accuracy_q025']:.4f},",
        f"median={severity_summary['balanced_accuracy_median']:.4f},",
        f"q975={severity_summary['balanced_accuracy_q975']:.4f}",
    )
    print(
        "Held-out macro-F1 median:",
        f"{severity_summary['macro_f1_median']:.4f}",
    )
    print(
        "Held-out ROC-AUC median:",
        f"{severity_summary['roc_auc_median']:.4f}",
    )
    print(
        "Held-out ARI median:",
        f"{severity_summary['ari_median']:.4f}",
    )
    print(
        "Held-out NMI median:",
        f"{severity_summary['nmi_median']:.4f}",
    )
    print(
        "Opposite-profile pairs with severity gap <=",
        tolerance,
        ":",
        len(matched_pairs_df),
    )

    if len(matched_pairs_df):
        print("\n10 closest opposite-profile severity pairs:\n")
        print(matched_pairs_df.head(10).to_string(index=False))

    print("\n=== GOVERNANCE NULL ===\n")
    print("Governance status:", governance_status)

    if governance_status == "COMPLETED":
        print("Governance column:", governance_summary["column"])
        print(
            "Cramer's V:",
            f"{governance_summary['cramers_v']:.4f}",
        )
        print(
            "Permutation p:",
            f"{governance_summary['permutation_p']:.6f}",
        )
        print(
            "Governance-only held-out balanced accuracy median:",
            f"{governance_summary['balanced_accuracy_median']:.4f}",
        )
        print(
            "Governance-only held-out macro-F1 median:",
            f"{governance_summary['macro_f1_median']:.4f}",
        )
        print(
            "Governance-only held-out ARI median:",
            f"{governance_summary['ari_median']:.4f}",
        )
        print("\nGovernance contingency:\n")
        print(governance_contingency.to_string())
    else:
        print(
            "No governance column was found in X_primary_raw.csv. "
            "Severity results are valid, but governance testing is incomplete."
        )

    print("\n=== INTERPRETIVE FLAGS ===\n")
    for key, value in report["interpretation_flags"].items():
        print(f"{key}: {value}")

    if governance_status == "COMPLETED":
        print("\nGATE STATUS: STAGE_3_COMPUTED_REVIEW_REQUIRED")
    else:
        print("\nGATE STATUS: STAGE_3_PARTIAL_GOVERNANCE_INPUT_REQUIRED")


if __name__ == "__main__":
    main()
