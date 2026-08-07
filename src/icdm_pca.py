from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "icdm" / "design" / "experiment_config.json"
INPUT_PATH = PROJECT_ROOT / "icdm" / "outputs" / "features" / "X_primary_raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "icdm" / "outputs" / "pca"
FIGURE_DIR = PROJECT_ROOT / "icdm" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing primary matrix: {INPUT_PATH}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(INPUT_PATH)

    id_column = config["id_column"]
    features = config["primary_features"]

    missing_columns = [c for c in [id_column] + features if c not in df.columns]
    if missing_columns:
        raise KeyError("Missing columns:\n- " + "\n- ".join(missing_columns))

    if df[id_column].isna().any() or df[id_column].duplicated().any():
        raise ValueError("Institution IDs are missing or duplicated.")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        missing = X.isna().sum()
        raise ValueError(
            "Primary matrix contains missing values:\n"
            + missing[missing > 0].to_string()
        )

    return config, df[[id_column]].copy(), X


def main():
    config, ids, X = load_inputs()

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA(svd_solver="full")
    scores = pca.fit_transform(Xz)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    retained = int(np.searchsorted(cumulative, 0.85) + 1)

    components = [f"PC{i+1}" for i in range(len(explained))]

    variance_table = pd.DataFrame(
        {
            "component": components,
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": explained,
            "explained_variance_percent": explained * 100,
            "cumulative_variance_ratio": cumulative,
            "cumulative_variance_percent": cumulative * 100,
            "retained_at_85_percent": [i < retained for i in range(len(explained))],
        }
    )
    variance_table.to_csv(OUTPUT_DIR / "explained_variance.csv", index=False)

    scores_all = pd.DataFrame(scores, columns=components)
    scores_all.insert(0, config["id_column"], ids.iloc[:, 0].to_numpy())
    scores_all.to_csv(OUTPUT_DIR / "pca_scores_all.csv", index=False)

    retained_columns = [config["id_column"]] + components[:retained]
    scores_all[retained_columns].to_csv(OUTPUT_DIR / "pca_scores.csv", index=False)

    weights = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=components,
    )
    weights.index.name = "feature"
    weights.to_csv(OUTPUT_DIR / "pca_component_weights.csv")

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=X.columns,
        columns=components,
    )
    loadings.index.name = "feature"
    loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv")

    top_rows = []
    for component in components[:retained]:
        ordered = loadings[component].abs().sort_values(ascending=False).head(5).index
        for rank, feature in enumerate(ordered, start=1):
            value = float(loadings.loc[feature, component])
            top_rows.append(
                {
                    "component": component,
                    "rank_by_absolute_loading": rank,
                    "feature": feature,
                    "loading": value,
                    "absolute_loading": abs(value),
                }
            )

    top_loadings = pd.DataFrame(top_rows)
    top_loadings.to_csv(OUTPUT_DIR / "pca_top_loadings.csv", index=False)

    x = np.arange(1, len(components) + 1)
    fig, left = plt.subplots(figsize=(8, 5))
    left.plot(x, explained * 100, marker="o")
    left.set_xlabel("Principal component")
    left.set_ylabel("Individual explained variance (%)")
    left.set_xticks(x)

    right = left.twinx()
    right.plot(x, cumulative * 100, marker="s")
    right.axhline(85, linestyle="--", linewidth=1)
    right.axvline(retained, linestyle=":", linewidth=1)
    right.set_ylabel("Cumulative explained variance (%)")

    left.set_title("PCA explained variance — FRICTA primary indicators")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pca_scree_plot.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "pca_scree_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "PCA_COMPLETE",
        "institutions": int(len(X)),
        "primary_features": int(X.shape[1]),
        "retention_rule": "Smallest number of PCs reaching at least 85% cumulative variance.",
        "retained_components": retained,
        "retained_cumulative_variance_percent": float(cumulative[retained - 1] * 100),
        "important_note": (
            "StandardScaler and PCA must be refitted inside every stability subsample. "
            "Full-data PCA scores are descriptive only."
        ),
        "component_names_not_assigned": True,
    }
    (OUTPUT_DIR / "pca_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== PCA SUMMARY ===\n")
    print(f"Institutions: {len(X)}")
    print(f"Primary features: {X.shape[1]}")
    print(f"Components retained at >=85%: {retained}")
    print(f"Retained cumulative variance: {cumulative[retained - 1] * 100:.2f}%")

    print("\n=== EXPLAINED VARIANCE ===\n")
    print(
        variance_table[
            [
                "component",
                "explained_variance_percent",
                "cumulative_variance_percent",
                "retained_at_85_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "explained_variance_percent": lambda v: f"{v:.2f}",
                "cumulative_variance_percent": lambda v: f"{v:.2f}",
            },
        )
    )

    print("\n=== TOP LOADINGS FOR RETAINED COMPONENTS ===\n")
    print(
        top_loadings[
            ["component", "rank_by_absolute_loading", "feature", "loading"]
        ].to_string(
            index=False,
            formatters={"loading": lambda v: f"{v:.4f}"},
        )
    )

    print(
        "\nGATE STATUS: PCA COMPUTED. Review variance and loadings before clustering.\n"
    )


if __name__ == "__main__":
    main()
