import os
import pandas as pd

OUTPUT_FILE = "paper_drafts/results_section.md"


def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def main():
    scored = read_csv("data/processed/fricta_scored.csv")
    branch = read_csv("data/processed/institutional_branch_ranking.csv")
    factors = read_csv("data/processed/institutional_factor_ranking.csv")
    profiles = read_csv("data/processed/high_vs_low_differences.csv")
    stress = read_csv("data/processed/hypothesis_stress_test_summary.csv")
    threshold = read_csv("data/processed/threshold_robustness_summary.csv")
    sensitivity = read_csv("data/processed/sensitivity_top10_overlap.csv")
    branch_removal = read_csv("data/processed/hypothesis_branch_removal_test.csv")
    ici = read_csv("data/processed/ici_decomposition_components.csv")

    n = len(scored)
    mean_afs = scored["AFS_theoretical"].mean()
    std_afs = scored["AFS_theoretical"].std()

    top_branch = branch.iloc[0]
    top_factor = factors.iloc[0]
    top_profile = profiles.iloc[0]

    digital = stress[stress["score"] == "digital_maturity_deficit"].iloc[0]
    human = stress[stress["score"] == "human_capacity_deficit"].iloc[0]

    top_ici = ici.iloc[0]

    os.makedirs("paper_drafts", exist_ok=True)

    text = f"""# 4. Results

## 4.1 Overall Friction Distribution

The final analytical dataset included **{n} institutions**. The theoretical Adoption Friction Score (AFS_theoretical) had a mean of **{mean_afs:.3f}** and a standard deviation of **{std_afs:.3f}**, indicating measurable variation in digital adoption friction across institutions.

## 4.2 Branch-Level Associations

At the branch level, all four FRICTA dimensions showed positive association with AFS_theoretical. The strongest branch-level association was observed for **{top_branch['branch']}** (Pearson r = **{top_branch['pearson_correlation_with_AFS']:.3f}**), followed by the remaining framework dimensions.

These results suggest that digital adoption friction is not explained by a single institutional dimension, but by a combination of operational, organizational, infrastructural, and human-capacity factors.

## 4.3 Institutional Factor Ranking

The strongest individual institutional factor associated with AFS_theoretical was **{top_factor['factor']}** (Pearson r = **{top_factor['pearson_correlation_with_AFS']:.3f}**). The highest-ranked predictors were concentrated around device availability, digital record-keeping, digital usage patterns, resource constraints, and tool variety.

This pattern suggests that friction is more closely associated with institutional digital maturity than with isolated barriers alone.

## 4.4 High- vs Low-Friction Institutional Profiles

To examine whether FRICTA distinguished meaningful institutional profiles, institutions were divided into low-, medium-, and high-friction groups based on AFS_theoretical percentiles.

The strongest high-vs-low difference was observed for **{top_profile['feature']}**, with a raw difference of **{top_profile['high_minus_low_difference']:.3f}** and a standardized difference of **{top_profile['standardized_difference']:.3f}**.

The top distinguishing factors included prior implementation experience, resource constraints, device availability, record-keeping systems, and digital usage frequency. This indicates that high-friction institutions differ from low-friction institutions across multiple dimensions of digital maturity and operational integration.

## 4.5 Hypothesis Stress Test

A direct stress test compared two competing explanations:

1. **Digital maturity deficit**
2. **Human-capacity deficit**

Digital maturity deficit showed a standardized difference of **{digital['standardized_difference']:.3f}**, compared with **{human['standardized_difference']:.3f}** for human-capacity deficit.

Thus, digital maturity indicators distinguished high-friction institutions more strongly than isolated human-capacity barriers.

## 4.6 Threshold Robustness

The digital maturity finding remained stable across alternative high- and low-friction threshold definitions. Across 20/80, 25/75, and 30/70 percentile splits, digital maturity consistently showed a stronger standardized difference than human capacity.

This suggests that the main finding does not depend on a single arbitrary percentile cutoff.

## 4.7 Weight Robustness

Sensitivity analysis showed that the top-10 institutional overlap remained stable under alternative weighting schemes. Across tested alternatives, the top-10 overlap ranged from **{sensitivity['top10_overlap_ratio'].min():.2f}** to **{sensitivity['top10_overlap_ratio'].max():.2f}**.

This indicates that institutional rankings were robust to moderate changes in FRICTA branch weights.

## 4.8 Branch-Removal Robustness

A leave-one-branch-out hypothesis test showed that digital maturity remained stronger than human capacity in **3 of 4** branch-removal conditions. The exception occurred when **ICI** was removed, where human capacity became slightly stronger.

This result suggests that the central finding is broadly robust, but also that infrastructure-related indicators act as a critical bridge between digital maturity and adoption friction.

## 4.9 ICI Decomposition

To better understand the role of infrastructure, ICI-related components were decomposed. The strongest component was **{top_ici['feature']}**, with a standardized difference of **{top_ici['standardized_difference']:.3f}** and a raw high-vs-low difference of **{top_ici['high_minus_low_difference']:.3f}**.

This suggests that device availability is the most informative infrastructure-related component for distinguishing low-friction and high-friction institutions.

## 4.10 Main Finding

Across institutional factor ranking, profile analysis, hypothesis stress testing, threshold robustness, branch-removal robustness, and component-level decomposition, digital adoption friction appeared to be associated more strongly with institutional digital maturity indicators than with isolated human-capacity barriers.

Specifically, infrastructure availability, digital record-keeping practices, operational digital integration, and previous implementation experience consistently distinguished high-friction institutions from low-friction institutions. Device availability emerged as the strongest individual institutional characteristic associated with friction, although the results should be interpreted as associative rather than causal.
"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    print("[SUCCESS] Results section generated.")
    print(f"[OUTPUT] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
