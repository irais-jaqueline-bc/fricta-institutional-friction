# src/figure3_branch_heatmap.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

corr = pd.DataFrame(
    [
        [1.000000, 0.081022, 0.802558, -0.104180],
        [0.081022, 1.000000, 0.335618, 0.790872],
        [0.802558, 0.335618, 1.000000, 0.234659],
        [-0.104180, 0.790872, 0.234659, 1.000000],
    ],
    columns=["ICI", "OCI", "OLI", "HCARI"],
    index=["ICI", "OCI", "OLI", "HCARI"],
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-0.2,
    vmax=1.0,
    square=True,
    linewidths=1,
    cbar_kws={"label": "Pearson Correlation"},
)

plt.title("FRICTA Branch Correlation Structure")
plt.tight_layout()

plt.savefig(
    "paper_figures/figure3_branch_correlation_heatmap.png", dpi=600, bbox_inches="tight"
)

plt.show()
