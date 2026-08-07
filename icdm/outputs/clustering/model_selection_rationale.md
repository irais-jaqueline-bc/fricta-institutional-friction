# Final Model-Selection Rationale

## Selected model

`R1_PCA_85__HAC_WARD__K2`

- Representation: `R1_PCA_85`
- Algorithm: `HAC_WARD`
- k: `2`
- Full-data Silhouette: `0.4108`
- Full-data Davies-Bouldin: `0.7760`
- Full-data Calinski-Harabasz: `46.17`
- Median resampling ARI: `1.0000`
- 95% empirical ARI interval: `[0.9053, 1.0000]`
- Weakest cluster mean matched Jaccard: `0.9916`
- Minimum cluster size observed during resampling: `7`
- Mean within-cluster consensus: `0.9961`
- Mean between-cluster consensus: `0.0027`

## Decision rule

Top-stability candidates produced partition-equivalent labels (ARI >= 0.95). Selected the representation with stronger full-data internal separation; retained the alternative as robustness evidence.

The candidates tied at the maximum median ARI were: R1_PCA_85__HAC_WARD__K2, R0_STANDARDIZED__HAC_WARD__K2.

## Interpretation boundary

This result supports a stable empirical partition in the current sample. It does not
establish causal classes, universal institutional types, or external ground truth.
FRICTA v1 archetypes will be compared only after model selection and were not used
to choose the clustering solution.

## Rejected solution families

Candidates producing clusters smaller than five institutions during resampling are
not retained for the primary solution. Lower-stability alternatives may remain as
sensitivity analyses but are not the primary empirical partition.
