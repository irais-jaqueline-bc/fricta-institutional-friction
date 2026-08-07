from __future__ import annotations

import hashlib
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
)
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]

GOVERNANCE_PATH = ROOT / "icdm" / "outputs" / "features" / "institution_metadata.csv"
FINAL_LABELS_PATH = (
    ROOT / "icdm" / "outputs" / "clustering" / "final_cluster_labels.csv"
)
CIPHER_CONFIG_PATH = ROOT / "cipher" / "design" / "cipher_experiment_config_frozen.json"

OUTPUT_DIR = ROOT / "cipher" / "outputs" / "null_models"
AUDIT_DIR = ROOT / "cipher" / "outputs" / "audit"

EXPECTED_COUNTS = {
    "NGO": 27,
    "PUBLIC": 22,
    "PRIVATE": 17,
    "MIXED": 14,
    "UNKNOWN": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_governance(value: object) -> str:
    text = str(value).strip().lower()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if "asociacion civil" in text or "ong" in text:
        return "NGO"
    if "public" in text:
        return "PUBLIC"
    if "privad" in text:
        return "PRIVATE"
    if "mixt" in text:
        return "MIXED"
    if "no estoy seguro" in text or "no estoy segura" in text:
        return "UNKNOWN"

    raise ValueError(f"Unrecognized governance value: {value!r}")


def cramers_v_bias_corrected(table: np.ndarray) -> float:
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


def permutation_test(
    governance: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    observed_table = pd.crosstab(
        pd.Series(labels, name="profile"),
        pd.Series(governance, name="governance"),
    ).to_numpy()

    observed_v = cramers_v_bias_corrected(observed_table)

    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)

    for i in range(permutations):
        permuted_labels = rng.permutation(labels)

        perm_table = pd.crosstab(
            pd.Series(permuted_labels, name="profile"),
            pd.Series(governance, name="governance"),
        ).to_numpy()

        null_values[i] = cramers_v_bias_corrected(perm_table)

    p_value = (1 + np.sum(null_values >= observed_v)) / (permutations + 1)

    return observed_v, float(p_value), null_values


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
        train = pd.DataFrame(
            {
                "governance": governance[train_idx],
                "profile": labels[train_idx],
            }
        )

        # Simple prespecified governance-only baseline:
        # each governance category predicts its majority profile in training.
        mapping = (
            train.groupby("governance")["profile"]
            .agg(lambda s: int(s.value_counts().idxmax()))
            .to_dict()
        )

        global_majority = int(pd.Series(labels[train_idx]).value_counts().idxmax())

        predicted = np.array(
            [mapping.get(value, global_majority) for value in governance[test_idx]],
            dtype=int,
        )

        rows.append(
            {
                "fold_index": fold_index,
                "balanced_accuracy": balanced_accuracy_score(
                    labels[test_idx],
                    predicted,
                ),
                "macro_f1": f1_score(
                    labels[test_idx],
                    predicted,
                    average="macro",
                ),
                "ari": adjusted_rand_score(
                    labels[test_idx],
                    predicted,
                ),
                "nmi": normalized_mutual_info_score(
                    labels[test_idx],
                    predicted,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if not GOVERNANCE_PATH.exists():
        raise FileNotFoundError(GOVERNANCE_PATH)

    if not FINAL_LABELS_PATH.exists():
        raise FileNotFoundError(FINAL_LABELS_PATH)

    config = load_json(CIPHER_CONFIG_PATH)

    metadata = pd.read_csv(GOVERNANCE_PATH)
    labels = pd.read_csv(FINAL_LABELS_PATH)

    required_metadata = {"institution_id", "institution_type"}
    missing_metadata = required_metadata - set(metadata.columns)

    if missing_metadata:
        raise KeyError(f"Missing metadata columns: {sorted(missing_metadata)}")

    if "institution_id" not in labels.columns or "cluster_id" not in labels.columns:
        raise KeyError(
            "final_cluster_labels.csv must contain institution_id and cluster_id."
        )

    metadata = metadata[["institution_id", "institution_type"]].copy()

    metadata["institution_id"] = metadata["institution_id"].astype(str)
    labels["institution_id"] = labels["institution_id"].astype(str)

    if metadata["institution_id"].duplicated().any():
        raise ValueError("Duplicate institution IDs in governance metadata.")

    if labels["institution_id"].duplicated().any():
        raise ValueError("Duplicate institution IDs in final labels.")

    metadata["governance"] = metadata["institution_type"].map(canonical_governance)

    counts = metadata["governance"].value_counts().to_dict()

    if counts != EXPECTED_COUNTS:
        raise ValueError(
            "Governance counts do not match the verified 81-institution distribution.\n"
            f"Observed: {counts}\n"
            f"Expected: {EXPECTED_COUNTS}"
        )

    data = labels[["institution_id", "cluster_id"]].merge(
        metadata[["institution_id", "institution_type", "governance"]],
        on="institution_id",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != 81:
        raise ValueError(f"Expected 81 aligned institutions; found {len(data)}.")

    governance = data["governance"].to_numpy()
    y = data["cluster_id"].astype(int).to_numpy()

    if set(np.unique(y)) != {1, 2}:
        raise ValueError(f"Expected profiles {{1,2}}, found {set(np.unique(y))}.")

    contingency = pd.crosstab(
        data["cluster_id"],
        data["governance"],
    ).reindex(
        columns=["NGO", "PUBLIC", "PRIVATE", "MIXED", "UNKNOWN"],
        fill_value=0,
    )

    permutations = int(config["null_models"]["governance_permutations"])
    splits = int(config["null_models"]["severity_cv_splits"])
    repeats = int(config["null_models"]["severity_cv_repeats"])
    seed = int(config["random_seed"])

    observed_v, permutation_p, null_values = permutation_test(
        governance,
        y,
        permutations=permutations,
        seed=seed + 5000,
    )

    cv = governance_only_cv(
        governance,
        y,
        splits=splits,
        repeats=repeats,
        seed=seed + 6000,
    )

    contingency.to_csv(OUTPUT_DIR / "governance_contingency.csv")

    cv.to_csv(
        OUTPUT_DIR / "governance_only_cv_folds.csv",
        index=False,
    )

    null_df = pd.DataFrame(
        {
            "permutation_index": np.arange(
                1,
                permutations + 1,
            ),
            "cramers_v": null_values,
        }
    )
    null_df.to_csv(
        OUTPUT_DIR / "governance_permutation_null.csv",
        index=False,
    )

    source_manifest = {
        "source_path": str(GOVERNANCE_PATH.relative_to(ROOT)),
        "source_sha256": sha256_file(GOVERNANCE_PATH),
        "aligned_n": int(len(data)),
        "governance_counts": {key: int(value) for key, value in counts.items()},
        "used_in_clustering": False,
        "used_for_model_selection": False,
        "role": ("post-hoc external metadata null test only"),
    }

    (AUDIT_DIR / "stage3_governance_source_manifest.json").write_text(
        json.dumps(
            source_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "cramers_v_bias_corrected": observed_v,
        "permutation_p": permutation_p,
        "permutations": permutations,
        "cv_splits": splits,
        "cv_repeats": repeats,
        "cv_folds": int(len(cv)),
        "balanced_accuracy": {
            "q025": float(cv["balanced_accuracy"].quantile(0.025)),
            "median": float(cv["balanced_accuracy"].median()),
            "q975": float(cv["balanced_accuracy"].quantile(0.975)),
        },
        "macro_f1_median": float(cv["macro_f1"].median()),
        "ari_median": float(cv["ari"].median()),
        "nmi_median": float(cv["nmi"].median()),
        "interpretive_flags": {
            "strong_governance_association": bool(
                observed_v >= 0.50 and permutation_p < 0.05
            ),
            "governance_nearly_reconstructs_profiles": bool(
                cv["balanced_accuracy"].median() >= 0.90 and cv["ari"].median() >= 0.80
            ),
        },
    }

    (OUTPUT_DIR / "governance_null_report.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 3B — GOVERNANCE NULL ===\n")

    print("Source:")
    print(
        " ",
        GOVERNANCE_PATH.relative_to(ROOT),
    )
    print(
        "SHA-256:",
        source_manifest["source_sha256"],
    )

    print("\nVerified governance counts:")
    print(counts)

    print("\n=== CONTINGENCY TABLE ===\n")
    print(contingency.to_string())

    print("\n=== ASSOCIATION TEST ===\n")
    print(
        "Bias-corrected Cramer's V:",
        f"{observed_v:.4f}",
    )
    print(
        "Permutation p:",
        f"{permutation_p:.6f}",
    )

    print("\n=== GOVERNANCE-ONLY HELD-OUT BASELINE ===\n")
    print(
        "Balanced accuracy:",
        f"q025={summary['balanced_accuracy']['q025']:.4f},",
        f"median={summary['balanced_accuracy']['median']:.4f},",
        f"q975={summary['balanced_accuracy']['q975']:.4f}",
    )
    print(
        "Macro-F1 median:",
        f"{summary['macro_f1_median']:.4f}",
    )
    print(
        "ARI median:",
        f"{summary['ari_median']:.4f}",
    )
    print(
        "NMI median:",
        f"{summary['nmi_median']:.4f}",
    )

    print("\n=== INTERPRETIVE FLAGS ===\n")
    for key, value in summary["interpretive_flags"].items():
        print(f"{key}: {value}")

    print("\nGATE STATUS: STAGE_3B_GOVERNANCE_COMPUTED_REVIEW_REQUIRED")


if __name__ == "__main__":
    main()
