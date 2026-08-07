# Frozen Model-Selection Protocol

## Candidate solutions

Evaluate K-Means, HAC-Ward, and diagonal-covariance GMM for k = 2, 3, 4, 5, 6
on both R0 (standardized primary features) and R1 (PCA retaining at least 85%
cumulative explained variance).

## Exclusion rule

Discard any solution containing a cluster with fewer than 5 institutions.

## Selection rule

1. Compare internal validity using Silhouette, Davies-Bouldin, and
   Calinski-Harabasz; use BIC additionally for GMM.
2. Retain competitive non-degenerate solutions.
3. Evaluate 1,000 80% subsamples without replacement.
4. Prefer the solution with the strongest median stability.
5. If solutions are practically tied, prefer:
   a. higher Silhouette;
   b. fewer clusters;
   c. clearer institutional interpretation.
6. Do not change this rule after inspecting attractive visualizations.

## Correlated-feature policy

High correlation alone does not force feature deletion when survey items measure
different institutional concepts. Every high-correlation feature will be subjected
to a pre-registered leave-one-feature-out sensitivity test.

## FRICTA v1 comparison

Legacy rule-based profiles are used only for theory-data structural concordance.
They are not labels, ground truth, or an external validation target.
