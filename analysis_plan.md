# FRICTA — Analysis Plan

## 1. Analysis Objectives

The objective of the FRICTA analysis pipeline is to computationally evaluate digital adoption friction patterns across Mexican childcare institutions through multidimensional institutional diagnostics.

The analysis seeks to:

- quantify institutional digital adoption friction,
- identify dominant institutional constraints,
- compare infrastructural, organizational, operational, and human-capacity friction patterns,
- evaluate weighting robustness,
- explore institutional segmentation patterns,
- and evaluate potential predictive relationships between institutional variables and adoption friction outcomes.

The analysis framework prioritizes interpretability, reproducibility, and computational transparency.

---

# 2. Primary Analytical Targets

## Primary Target Variable

The primary analytical target is:

$$ AFS_{theoretical} $$

This score represents the theoretically informed Adoption Friction Score computed through weighted branch aggregation.

The theoretically informed score is treated as the principal institutional friction estimate because it incorporates theory-driven weighting assumptions derived from digital adoption and organizational constraint literature.

---

## Secondary Analytical Targets

Secondary analytical targets include:

- $$ AFS_{baseline} $$
- institutional friction category,
- dominant institutional constraint,
- branch-level scores,
- and exploratory institutional segmentation profiles.

---

# 3. Dataset Preparation

The dataset preparation pipeline includes:

- duplicate response removal,
- incomplete response filtering,
- categorical encoding,
- normalization procedures,
- derived variable construction,
- branch assignment validation,
- and scoring verification.

Each dataset row represents one institution.

The processing pipeline follows:

Raw Survey Responses  
→ Encoding  
→ Cleaning  
→ Normalization  
→ Derived Variable Construction  
→ Branch Aggregation  
→ Global Scoring  
→ Diagnostic Output Generation

---

# 4. Variable Encoding and Normalization

All scoring variables follow a friction-oriented normalization strategy where:

$$ 0 = Low\ Friction $$

$$ 1 = High\ Friction $$

Normalization procedures follow the mathematical rules defined in:

- `mathematical_formalization.md`
- `data_dictionary.md`

Both direct and reverse min-max normalization procedures are used depending on variable directionality.

---

# 5. Derived Variable Construction

Derived variables are computed after normalization and before branch aggregation.

Derived variables include:

- digital_tool_variety,
- digital_tool_constraint,
- staff_per_device_ratio,
- children_per_staff_ratio,
- administrative_digital_burden,
- adoption_readiness_gap,
- operational_capacity_constraint,
- digital_exposure_constraint,
- training_dependency_pressure,
- and implementation_friction_signal.

Derived variables are used for:

- institutional diagnostics,
- exploratory analysis,
- recommendation refinement,
- clustering,
- and future predictive modeling.

Only designated core variables are included in branch aggregation formulas.

---

# 6. Branch-Level Analysis

FRICTA evaluates four institutional diagnostic branches:

| Branch | Meaning |
|---|---|
| ICI | Infrastructure Constraints Index |
| OCI | Organizational Constraints Index |
| OLI | Operational Load Index |
| HCARI | Human Capacity & Adoption Readiness Index |

Branch-level analysis includes:

- score distributions,
- branch comparison,
- branch dominance frequency,
- and branch interaction analysis.

Branch scores range from 0 to 1.

Higher values indicate greater institutional friction intensity.

---

# 7. Global Friction Analysis

The primary global institutional friction estimate is:

$$ AFS_{theoretical} $$

The equal-weight score:

$$ AFS_{baseline} $$

is used as a robustness reference.

Global friction analysis includes:

- institutional score distributions,
- friction category distributions,
- high-friction institution identification,
- and regional friction comparison.

---

# 8. Descriptive Statistical Analysis

Planned descriptive statistical procedures include:

- mean,
- median,
- standard deviation,
- variance,
- minimum and maximum values,
- quartile analysis,
- branch-level distributions,
- and institutional category frequencies.

Descriptive analysis will be performed for:

- normalized variables,
- branch scores,
- global friction scores,
- and auxiliary diagnostic variables.

---

# 9. Correlation Analysis

FRICTA includes exploratory correlation analysis to evaluate relationships between institutional variables.

Planned correlation procedures include:

- Pearson correlation analysis,
- Spearman rank correlation analysis,
- branch interaction analysis,
- and auxiliary-variable association analysis.

Correlation analysis will evaluate whether:

- infrastructural,
- organizational,
- operational,
- or human-readiness variables

show stronger relationships with institutional friction outcomes.

---

# 10. Sensitivity Analysis

FRICTA evaluates weighting robustness through comparative analysis between:

$$ AFS_{baseline} $$

and

$$ AFS_{theoretical} $$

The primary sensitivity metric is:

$$ \Delta AFS = |AFS_{baseline} - AFS_{theoretical}| $$

Sensitivity analysis includes:

- score stability analysis,
- institutional ranking stability,
- dominant constraint variation,
- and score distribution consistency.

The objective is to evaluate whether theoretical weighting assumptions substantially alter institutional diagnostics.

---

# 11. Institutional Classification Analysis

Institutions are classified into four exploratory friction categories:

| AFS Range | Classification |
|---|---|
| 0.00–0.24 | Low Friction |
| 0.25–0.49 | Moderate Friction |
| 0.50–0.74 | High Friction |
| 0.75–1.00 | Severe Friction |

Classification analysis includes:

- category frequency analysis,
- branch dominance within categories,
- and comparative institutional profiles.

---

# 12. Dominant Constraint Analysis

FRICTA identifies dominant institutional constraints through:

$$ dominant\_constraint = \arg\max(ICI, OCI, OLI, HCARI) $$

Dominant constraint analysis includes:

- institutional profile frequency,
- mixed-profile analysis,
- branch dominance comparison,
- and recommendation alignment analysis.

Mixed friction profiles may be assigned when:

$$ |Score_{highest} - Score_{second\ highest}| \leq 0.10 $$

---

# 13. Institutional Segmentation Analysis

FRICTA includes exploratory institutional segmentation analysis.

Planned segmentation approaches include:

- branch-based grouping,
- friction-category grouping,
- dominant-constraint grouping,
- and clustering-based exploratory institutional profiling.

The objective is to identify recurring institutional friction structures.

---

# 14. Clustering Analysis

Exploratory clustering procedures may include:

- K-Means clustering,
- branch-score clustering,
- and institutional profile grouping.

Clustering analysis is intended to evaluate whether institutions naturally organize into recurring friction-pattern categories.

Potential clustering variables include:

- branch scores,
- normalized core variables,
- and auxiliary diagnostic variables.

---

# 15. Predictive Modeling Analysis

FRICTA includes exploratory predictive modeling analysis intended to evaluate whether institutional variables can predict digital adoption friction outcomes.

Potential modeling approaches include:

- linear regression,
- regularized regression,
- decision trees,
- random forest modeling,
- and interpretable comparative modeling.

The primary predictive target is:

$$ AFS_{theoretical} $$

Secondary predictive targets include:

- friction category,
- dominant constraint,
- and branch-level scores.

---

# 16. Feature Importance Analysis

FRICTA includes exploratory feature importance estimation procedures.

Potential approaches include:

- random forest feature importance,
- regression coefficient analysis,
- and branch contribution comparison.

The objective is to identify which institutional variables most strongly contribute to institutional digital adoption friction.

---

# 17. Visualization Plan

Planned visualizations include:

| Visualization | Purpose |
|---|---|
| Histograms | Friction score distributions |
| Boxplots | Branch comparison |
| Correlation heatmaps | Variable interaction analysis |
| Bar charts | Dominant constraint frequency |
| Radar charts | Institutional diagnostic profiles |
| Scatterplots | Branch relationship exploration |
| Cluster visualizations | Institutional segmentation |

Visualizations are intended to improve interpretability and institutional diagnostic clarity.

---

# 18. Dashboard and Diagnostic Layer

FRICTA includes a planned interactive dashboard and API-based diagnostic architecture.

Planned dashboard outputs include:

- institutional friction scores,
- branch-level diagnostics,
- dominant constraint identification,
- recommendation categories,
- regional distributions,
- and institutional profile visualization.

The dashboard is intended to support institutional interpretation rather than automated decision-making.

---

# 19. Pilot Validation Analysis

Pilot institutions will be used to evaluate diagnostic interpretability and institutional usefulness.

Pilot analysis may include:

- diagnostic validation,
- institutional feedback collection,
- recommendation evaluation,
- onboarding analysis,
- and operational interpretation assessment.

The pilot stage is intended to evaluate whether computational diagnostics align with institutional operational realities.

---

# 20. Reproducibility Framework

FRICTA prioritizes reproducible computational analysis.

The analytical architecture is designed to support:

- reproducible scoring,
- transparent normalization,
- modular computational processing,
- and scalable institutional diagnostics.

Planned computational outputs include:

- reproducible notebooks,
- scoring scripts,
- visualization scripts,
- and exportable institutional diagnostics.

---

# 21. Methodological Boundaries

The current analytical stage is exploratory.

The framework does not currently claim:

- causal inference,
- nationally representative estimation,
- or validated psychometric measurement.

Instead, the analysis framework is designed to evaluate whether computational institutional diagnostics can meaningfully model digital adoption friction patterns within childcare institutions.

---

# 22. Computational Research Position

FRICTA is positioned as a computational institutional diagnostics framework.

The analysis architecture integrates:

- multidimensional scoring,
- institutional diagnostics,
- branch analytics,
- recommendation-oriented outputs,
- exploratory predictive modeling,
- and computational interpretation layers.

The project is intended to function as an interpretable institutional analytics system rather than a purely descriptive survey study.