# FRICTA — Institutional Digital Adoption Friction

FRICTA is an independent research project investigating digital adoption friction in Mexican residential childcare institutions.

The project combines institutional data collection, computational measurement, unsupervised learning, robustness analysis, and reproducible software to study how organizational, infrastructural, operational, and human-capacity constraints shape technology adoption.

The repository currently contains two related research tracks:

- **FRICTA** — a theory-driven framework for measuring institutional digital adoption friction.
- **CIPHER** — a post-discovery audit framework for evaluating the integrity of claims derived from unsupervised institutional profiles.

---

## FRICTA

**Framework for Institutional Digital Adoption Friction Assessment**

FRICTA models digital adoption friction as a multidimensional institutional phenomenon rather than simply a question of individual willingness to use technology.

The study uses an original dataset of **81 residential childcare institutions** operating across Mexico.

### Core dimensions

FRICTA constructs four principal institutional indices:

- **ICI** — Infrastructure Constraint Index
- **OCI** — Organizational Constraint Index
- **OLI** — Operational Load Index
- **HCARI** — Human Capacity and Readiness Index

These dimensions contribute to an aggregate **Adoption Friction Score (AFS)** and interpretable institutional friction profiles.

### Current empirical findings

The current analysis identifies organizational and structural conditions as major sources of digital adoption friction.

The original rule-based profiling produced the following distribution:

- Organizationally-Limited: **52/81 (64.2%)**
- Infrastructure-Limited: **13/81 (16.0%)**
- Human-Capacity-Limited: **10/81 (12.3%)**
- Multi-Constraint: **3/81 (3.7%)**
- Operationally-Limited: **3/81 (3.7%)**

The framework is implemented as a reproducible Python-based analytical pipeline.

**Status:** Manuscript submitted to **CONAIC 2026** and currently under review.

---

## CIPHER

**Claim Integrity in Profiling under Heterogeneous Ensemble Robustness**

CIPHER extends the project in a different direction.

Rather than assuming that a stable clustering automatically supports every interpretation derived from it, CIPHER evaluates different claims about an unsupervised result separately.

The framework audits:

1. **Partition reproducibility**
2. **Instance-level membership integrity**
3. **Resistance to simpler alternative explanations**
4. **Counterfactual integrity across model perturbations**

### Discovery model

The analysis evaluates the original 13-dimensional representation and PCA-based representations using:

- Ward hierarchical clustering
- K-means
- Gaussian mixture models
- PCA
- Subsample stability analysis

The retained reference model uses **Ward clustering with k = 2** on five principal components explaining **86.82% of variance**.

Key results:

- Silhouette score: **0.411**
- Median resampling ARI: **1.000**
- 95% empirical interval: **0.905–1.000**
- Final partition: **68 / 13 institutions**

### Claim-integrity audit

CIPHER evaluates robustness using a heterogeneous ensemble of **1,000 perturbed models**, varying:

- sampled institutions,
- feature subsets,
- representation,
- and clustering family.

The framework intentionally distinguishes between claims that are **supported, limited, rejected, or not evaluable**, rather than collapsing robustness into a single global validity score.

**Status:** Paper submitted to the **IEEE ICDM 2026 Teen Research Competition** and currently under review.