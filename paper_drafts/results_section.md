# 4. Results

## 4.1 Overall Friction Distribution

The final analytical dataset included **81 institutions**. The theoretical Adoption Friction Score (AFS_theoretical) had a mean of **0.416** and a standard deviation of **0.151**, indicating measurable variation in digital adoption friction across institutions.


## 4.2 Branch-Level Associations

At the branch level, all four FRICTA dimensions showed positive association with AFS_theoretical. The strongest branch-level association was observed for **OLI** (Pearson r = **0.847**), followed by the remaining framework dimensions.

These results suggest that digital adoption friction is not explained by a single institutional dimension, but by a combination of operational, organizational, infrastructural, and human-capacity factors.

## 4.3 Institutional Factor Ranking

The strongest individual institutional factor associated with AFS_theoretical was **device_constraint** (Pearson r = **0.694**). The highest-ranked predictors were concentrated around device availability, digital record-keeping, digital usage patterns, resource constraints, and tool variety.

This pattern suggests that friction is more closely associated with institutional digital maturity than with isolated barriers alone.

## 4.4 High- vs Low-Friction Institutional Profiles

To examine whether FRICTA distinguished meaningful institutional profiles, institutions were divided into low-, medium-, and high-friction groups based on AFS_theoretical percentiles.

The strongest high-vs-low difference was observed for **previous_implementation_constraint**, with a raw difference of **0.571** and a standardized difference of **1.250**.

The top distinguishing factors included prior implementation experience, resource constraints, device availability, record-keeping systems, and digital usage frequency. This indicates that high-friction institutions differ from low-friction institutions across multiple dimensions of digital maturity and operational integration.

## 4.5 Hypothesis Stress Test

A direct stress test compared two competing explanations:

1. **Digital maturity deficit**
2. **Human-capacity deficit**

Digital maturity deficit showed a standardized difference of **1.535**, compared with **0.899** for human-capacity deficit.

Thus, digital maturity indicators distinguished high-friction institutions more strongly than isolated human-capacity barriers.

## 4.6 Threshold Robustness

The digital maturity finding remained stable across alternative high- and low-friction threshold definitions. Across 20/80, 25/75, and 30/70 percentile splits, digital maturity consistently showed a stronger standardized difference than human capacity.

This suggests that the main finding does not depend on a single arbitrary percentile cutoff.

## 4.7 Weight Robustness

Sensitivity analysis showed that the top-10 institutional overlap remained stable under alternative weighting schemes. Across tested alternatives, the top-10 overlap ranged from **0.80** to **0.90**.

This indicates that institutional rankings were robust to moderate changes in FRICTA branch weights.

## 4.8 Branch-Removal Robustness

A leave-one-branch-out hypothesis test showed that digital maturity remained stronger than human capacity in **3 of 4** branch-removal conditions. The exception occurred when **ICI** was removed, where human capacity became slightly stronger.

This result suggests that the central finding is broadly robust, but also that infrastructure-related indicators act as a critical bridge between digital maturity and adoption friction.

## 4.9 ICI Decomposition

To better understand the role of infrastructure, ICI-related components were decomposed. The strongest component was **device_constraint**, with a standardized difference of **1.595** and a raw high-vs-low difference of **0.556**.

This suggests that device availability is the most informative infrastructure-related component for distinguishing low-friction and high-friction institutions.

## 4.10 Main Finding

Across institutional factor ranking, profile analysis, hypothesis stress testing, threshold robustness, branch-removal robustness, and component-level decomposition, digital adoption friction appeared to be associated more strongly with institutional digital maturity indicators than with isolated human-capacity barriers.

Specifically, infrastructure availability, digital record-keeping practices, operational digital integration, and previous implementation experience consistently distinguished high-friction institutions from low-friction institutions. Device availability emerged as the strongest individual institutional characteristic associated with friction, although the results should be interpreted as associative rather than causal.

## 4.11 Institutional Friction Archetypes

To examine whether high adoption friction appeared as a homogeneous phenomenon, institutions were classified into rule-based friction archetypes according to their dominant FRICTA branch.

The most frequent archetype was Organizationally-Limited, representing 52 of 81 institutions (64.2%). Infrastructure-Limited institutions represented 13 cases (16.1%), Human-Capacity-Limited institutions represented 10 cases (12.4%), while Multi-Constraint and Operationally-Limited institutions each represented 3 cases (3.7%).

This distribution suggests that digital adoption friction is not reducible to a single technological shortage. Instead, the dominant pattern in the sample was organizational limitation, indicating that internal processes, administrative structure, and resistance to system change may play a central role in shaping adoption friction.

These rule-based archetypes provide a preliminary taxonomy of institutional friction profiles that can later be refined through unsupervised learning techniques in larger datasets.