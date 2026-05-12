# FRICTA — Mathematical Formalization

## 1. Purpose

This document defines the mathematical structure of the FRICTA framework.

FRICTA models digital adoption friction as a normalized, multidimensional institutional construct. The objective is to transform survey responses into interpretable institutional diagnostics through:

1. variable normalization,
2. derived variable construction,
3. branch-level index computation,
4. global adoption friction scoring,
5. institutional classification,
6. dominant constraint detection,
7. and recommendation-oriented diagnostics.

All scoring variables follow a friction-oriented scale:

| Value | Interpretation |
|---|---|
| 0 | Low friction |
| 1 | High friction |

This ensures directional consistency across all institutional indicators.

---

## 2. Friction-Oriented Normalization Principle

All variables included in the scoring model are transformed so that higher values always indicate greater digital adoption friction.

For variables where higher raw values already represent greater friction, the following normalization is used:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

For variables where higher raw values represent lower friction, reverse normalization is used:

$$
x' = \frac{x_{max} - x}{x_{max} - x_{min}}
$$

Where:

| Symbol | Meaning |
|---|---|
| $begin:math:text$x$end:math:text$ | raw encoded value |
| $begin:math:text$x\'$end:math:text$ | normalized friction-oriented value |
| $begin:math:text$x\_\{min\}$end:math:text$ | minimum possible encoded value |
| $begin:math:text$x\_\{max\}$end:math:text$ | maximum possible encoded value |

---

## 3. Redundancy Control Principle

Before computing branch-level indices, FRICTA separates variables into two categories:

1. Core scoring variables
2. Auxiliary diagnostic variables

Core scoring variables are used directly to compute institutional indices.

Auxiliary diagnostic variables are used for interpretation, visualization, reporting, recommendation generation, exploratory analysis, and future modeling, but are not included directly in the primary index formulas.

This distinction prevents accidental double weighting when a derived variable reuses normalized variables that are already included in a branch index.

For example:

$$
implementation\_friction\_signal =
\frac{
implementation\_difficulty\_norm +
system\_change\_resistance\_norm
}{2}
$$

Since both `implementation_difficulty_norm` and `system_change_resistance_norm` already appear as core organizational variables, `implementation_friction_signal` is treated as an auxiliary diagnostic variable rather than a core scoring variable.

---

## 4. Core vs Auxiliary Variable Policy

### Core Scoring Variables

Core scoring variables are included directly in branch-level index computation.

### Auxiliary Diagnostic Variables

Auxiliary variables may be used for:

- diagnostic interpretation,
- institutional profile explanation,
- recommendation mapping,
- visualization,
- exploratory analysis,
- and future modeling.

They are not included in the main branch aggregation formulas unless explicitly justified in a later validated model version.

---

## 5. Final Core Variable Assignment

### 5.1 Infrastructure Constraints Index — ICI

Core variables:

| Variable | Meaning |
|---|---|
| `device_constraint` | Limited device availability |
| `internet_constraint` | Internet instability |
| `digital_tool_constraint` | Low diversity of digital tools |
| `resource_constraint_norm` | Perceived infrastructural-operational scarcity |

Auxiliary variables:

| Variable | Status |
|---|---|
| `staff_per_device_ratio` | Auxiliary diagnostic variable |

---

### 5.2 Organizational Constraints Index — OCI

Core variables:

| Variable | Meaning |
|---|---|
| `admin_disorganization` | Low administrative organization |
| `implementation_difficulty_norm` | Perceived implementation difficulty |
| `system_change_resistance_norm` | Resistance to changing current systems |

Auxiliary variables:

| Variable | Status |
|---|---|
| `previous_digital_implementation` | Auxiliary contextual proxy |
| `implementation_friction_signal` | Auxiliary diagnostic variable |
| `administrative_digital_burden` | Auxiliary diagnostic variable |
| `digital_exposure_constraint` | Auxiliary diagnostic variable |
| `adoption_readiness_gap` | Auxiliary diagnostic variable |

---

### 5.3 Operational Load Index — OLI

Core variables:

| Variable | Meaning |
|---|---|
| `admin_time_load_norm` | Administrative time burden |
| `time_constraint_norm` | Lack of available time |
| `staffing_constraint_norm` | Staffing scarcity |

Auxiliary variables:

| Variable | Status |
|---|---|
| `operational_capacity_constraint` | Auxiliary diagnostic variable |
| `children_per_staff_ratio` | Auxiliary contextual approximation |

---

### 5.4 Human Capacity & Adoption Readiness Index — HCARI

Core variables:

| Variable | Meaning |
|---|---|
| `training_deficit_norm` | Lack of digital training |
| `digital_usage_constraint` | Low digital usage frequency |
| `willingness_constraint` | Low willingness to adopt digital tools |

Auxiliary variables:

| Variable | Status |
|---|---|
| `training_dependency_pressure` | Auxiliary diagnostic variable |
| `perceived_digital_utility_norm` | Auxiliary perception indicator |
| `pilot_openness` | Pilot participation metadata |

---

## 6. Mathematical Implication

The primary FRICTA score is computed only from core scoring variables.

Auxiliary variables do not directly affect the primary Adoption Friction Score.

This design keeps the model:

- interpretable,
- reproducible,
- mathematically stable,
- and protected against unintended signal duplication.

Auxiliary variables remain important for institutional diagnostics, but they operate after the main friction scores are computed.

---

## 7. Derived Variable Construction

Derived variables are computational indicators created from existing survey variables. They are not direct survey responses.

They are used to capture institutional patterns that cannot be observed from a single question.

### 7.1 Digital Tool Variety

$$
digital\_tool\_variety = \sum Tool_i
$$

Where $begin:math:text$Tool\_i$end:math:text$ represents each binary tool-use indicator.

Included tools:

- `uses_excel`
- `uses_whatsapp`
- `uses_google_workspace`
- `uses_specialized_software`
- `other_tool`

---

### 7.2 Digital Tool Constraint

$$
digital\_tool\_constraint =
\frac{x_{max}-x}{x_{max}-x_{min}}
$$

Higher values indicate lower digital tool diversity and greater digital integration constraint.

---

### 7.3 Staff per Device Ratio

$$
staff\_per\_device\_ratio =
\frac{staff\_size}{available\_devices + 1}
$$

The $begin:math:text$\+1$end:math:text$ term prevents division by zero when institutions report zero available devices.

This variable is used only as an auxiliary diagnostic indicator.

---

### 7.4 Children per Staff Ratio

$$
children\_per\_staff\_ratio =
\frac{children\_served}{staff\_size}
$$

Because both variables are ordinal ranges, this ratio is treated as an approximate ordinal-derived indicator, not an exact demographic measure.

This variable is used only as an auxiliary contextual indicator.

---

### 7.5 Administrative Digital Burden

$$
administrative\_digital\_burden =
\frac{
admin\_time\_load\_norm +
digital\_usage\_constraint +
administrative\_digitization\_constraint
}{3}
$$

This variable captures administrative burden under low digital support conditions.

It is used as an auxiliary diagnostic variable because it reuses variables that may also appear in other branches.

---

### 7.6 Adoption Readiness Gap

$$
adoption\_readiness\_gap =
\left|
perceived\_digital\_utility\_norm -
willingness\_constraint
\right|
$$

This variable captures mismatch between perceived usefulness and actual willingness to adopt a digital tool.

It is used as an auxiliary diagnostic variable.

---

### 7.7 Operational Capacity Constraint

$$
operational\_capacity\_constraint =
\frac{
time\_constraint\_norm +
staffing\_constraint\_norm
}{2}
$$

This variable captures combined operational pressure from time scarcity and staffing scarcity.

It is used as an auxiliary diagnostic variable because its components already appear inside the Operational Load Index.

---

### 7.8 Digital Exposure Constraint

$$
digital\_exposure\_constraint =
\frac{
previous\_digital\_implementation +
digital\_usage\_constraint
}{2}
$$

This variable captures low digital exposure based on prior implementation and current digital usage frequency.

It is used as an auxiliary diagnostic variable.

---

### 7.9 Training Dependency Pressure

$$
training\_dependency\_pressure =
\frac{
training\_deficit\_norm +
digital\_usage\_constraint
}{2}
$$

This variable captures adoption pressure associated with limited training and low digital usage.

It is used as an auxiliary diagnostic variable because its components already appear inside the Human Capacity & Adoption Readiness Index.

---

### 7.10 Implementation Friction Signal

$$
implementation\_friction\_signal =
\frac{
implementation\_difficulty\_norm +
system\_change\_resistance\_norm
}{2}
$$

This variable captures direct implementation-related friction.

It is used as an auxiliary diagnostic variable because its components already appear inside the Organizational Constraints Index.

---

## 8. Branch Aggregation Functions

Branch indices are computed by averaging only core scoring variables.

Each branch score ranges from 0 to 1.

Higher values indicate stronger constraint intensity in that branch.

---

### 8.1 Infrastructure Constraints Index — ICI

$$
ICI =
\frac{
device\_constraint +
internet\_constraint +
digital\_tool\_constraint +
resource\_constraint\_norm
}{4}
$$

Interpretation:

| Score Direction | Meaning |
|---|---|
| Low ICI | Low infrastructural constraint |
| High ICI | High infrastructural constraint |

---

### 8.2 Organizational Constraints Index — OCI

$$
OCI =
\frac{
admin\_disorganization +
implementation\_difficulty\_norm +
system\_change\_resistance\_norm
}{3}
$$

Interpretation:

| Score Direction | Meaning |
|---|---|
| Low OCI | Low organizational constraint |
| High OCI | High organizational constraint |

Note: `previous_digital_implementation`, `implementation_friction_signal`, `administrative_digital_burden`, `digital_exposure_constraint`, and `adoption_readiness_gap` are excluded from the primary OCI formula to avoid signal duplication.

---

### 8.3 Operational Load Index — OLI

$$
OLI =
\frac{
admin\_time\_load\_norm +
time\_constraint\_norm +
staffing\_constraint\_norm
}{3}
$$

Interpretation:

| Score Direction | Meaning |
|---|---|
| Low OLI | Low operational saturation |
| High OLI | High operational saturation |

---

### 8.4 Human Capacity & Adoption Readiness Index — HCARI

$$
HCARI =
\frac{
training\_deficit\_norm +
digital\_usage\_constraint +
willingness\_constraint
}{3}
$$

Interpretation:

| Score Direction | Meaning |
|---|---|
| Low HCARI | Low human-capacity/readiness constraint |
| High HCARI | High human-capacity/readiness constraint |

---

## 9. Global Adoption Friction Computation

The Adoption Friction Score (AFS) estimates the overall level of institutional digital adoption friction.

FRICTA computes two scoring scenarios:

1. Equal-weight baseline score
2. Theoretically informed weighted score

---

### 9.1 Equal-Weight Baseline Score

$$
AFS_{baseline} =
\frac{
ICI + OCI + OLI + HCARI
}{4}
$$

The equal-weight baseline score provides a neutral aggregation reference independent of theoretical weighting assumptions.

---

### 9.2 Theoretically Informed Weighted Score

$$
AFS_{theoretical} =
0.30(ICI) +
0.30(OCI) +
0.25(OLI) +
0.15(HCARI)
$$

This weighting scenario reflects the theoretical assumption that infrastructural and organizational constraints operate as higher-order institutional bottlenecks, while operational load and human adoption readiness modulate implementation feasibility.

These weights are not interpreted as empirically validated causal coefficients. They represent a theory-driven scoring scenario for institutional diagnostics.

---

## 10. Sensitivity Analysis Procedure

FRICTA evaluates score robustness through sensitivity analysis between equal-weight and theoretically informed weighting scenarios.

The basic score difference is computed as:

$$
\Delta AFS =
\left|
AFS_{baseline} -
AFS_{theoretical}
\right|
$$

Where:

| Symbol | Meaning |
|---|---|
| $begin:math:text$\\Delta AFS$end:math:text$ | Difference between weighting scenarios |
| $begin:math:text$AFS\_\{baseline\}$end:math:text$ | Equal-weight Adoption Friction Score |
| $begin:math:text$AFS\_\{theoretical\}$end:math:text$ | Theoretically informed Adoption Friction Score |

The sensitivity analysis evaluates:

- score stability,
- institutional ranking stability,
- dominant constraint variation,
- and score distribution consistency.

If $begin:math:text$\\Delta AFS$end:math:text$ remains low across institutions, the model is considered relatively stable across weighting assumptions.

If $begin:math:text$\\Delta AFS$end:math:text$ is high, the diagnostic output is interpreted as sensitive to theoretical weighting assumptions and should be reviewed carefully.

---

## 11. Institutional Friction Classification

FRICTA classifies institutions according to their Adoption Friction Score.

The initial classification system is exploratory and may be refined after empirical validation.

| AFS Range | Classification | Interpretation |
|---|---|---|
| 0.00–0.24 | Low Friction | Low apparent institutional friction |
| 0.25–0.49 | Moderate Friction | Manageable but visible adoption barriers |
| 0.50–0.74 | High Friction | Strong institutional constraints |
| 0.75–1.00 | Severe Friction | Critical adoption barriers |

This classification is used for interpretability and institutional reporting, not as a validated diagnostic threshold system at this stage.

---

## 12. Dominant Constraint Detection

FRICTA identifies the dominant institutional constraint by selecting the highest branch score.

$$
dominant\_constraint =
\arg\max(ICI, OCI, OLI, HCARI)
$$

The dominant constraint indicates the branch with the strongest friction signal.

| Highest Branch Score | Institutional Profile |
|---|---|
| ICI | Infrastructure-limited institution |
| OCI | Organizational rigidity profile |
| OLI | Operational saturation profile |
| HCARI | Human-capacity limitation profile |

If two or more branches have similar high scores, the institution may be classified as having a mixed friction profile.

A mixed profile may be assigned when the difference between the two highest branch scores is less than or equal to 0.10.

$$
MixedProfile =
\left|
Score_{highest} -
Score_{second\ highest}
\right|
\leq 0.10
$$

---

## 13. Recommendation Layer Logic

FRICTA connects dominant constraint profiles to recommendation categories.

The recommendation layer does not prescribe a final technological intervention. Instead, it generates an institutional priority direction.

| Dominant Constraint | Recommendation Type | Strategic Focus |
|---|---|---|
| ICI | Infrastructure support | Stabilize connectivity, devices, or basic digital resources |
| OCI | Administrative restructuring | Simplify workflows, clarify processes, reduce organizational rigidity |
| OLI | Operational load reduction | Reduce administrative burden before adding new systems |
| HCARI | Training and adoption support | Prioritize staff training, onboarding, and digital confidence |

---

### 13.1 Recommendation Rule

$$
Recommendation =
f(dominant\_constraint, AFS, AuxiliarySignals)
$$

Where:

| Component | Role |
|---|---|
| `dominant_constraint` | Identifies the primary institutional bottleneck |
| `AFS` | Indicates overall friction severity |
| `AuxiliarySignals` | Provide interpretive context for tailored recommendations |

Auxiliary variables may refine recommendations but do not alter the primary score.

For example:

- `staff_per_device_ratio` may refine infrastructure recommendations.
- `administrative_digital_burden` may refine organizational or operational recommendations.
- `training_dependency_pressure` may refine training-related recommendations.
- `adoption_readiness_gap` may refine readiness recommendations.

---

## 14. Computational Pipeline Formalization

The complete FRICTA computational pipeline is defined as:

```text
Survey Responses
→ Variable Encoding
→ Variable Normalization
→ Derived Variable Construction
→ Core/Auxiliary Variable Separation
→ Branch Aggregation
→ Global Adoption Friction Computation
→ Sensitivity Analysis
→ Institutional Classification
→ Dominant Constraint Detection
→ Recommendation Layer
```

---

## 15. Planned Statistical Validation

The FRICTA architecture is designed to support statistical validation before final empirical interpretation.

Planned validation procedures include:

- internal consistency analysis,
- correlation structure analysis,
- sensitivity analysis,
- exploratory factor analysis if sample size permits,
- predictive modeling analysis,
- and feature importance estimation.

The predictive objective of the project is:

> Which institutional factors most strongly predict digital adoption friction in Mexican childcare institutions?

---

## 16. Predictive Modeling Layer

After computing branch indices and Adoption Friction Scores, FRICTA may evaluate which institutional variables most strongly predict digital adoption friction.

Potential modeling approaches include:

- linear regression,
- regularized regression,
- decision trees,
- random forest feature importance,
- and interpretable model comparison.

The target variable may be:

| Target | Description |
|---|---|
| $begin:math:text$AFS\_\{baseline\}$end:math:text$ | Equal-weight friction score |
| $begin:math:text$AFS\_\{theoretical\}$end:math:text$ | Theoretically weighted friction score |
| Friction category | Low / moderate / high / severe friction classification |

Candidate predictors may include:

- core normalized variables,
- auxiliary diagnostic variables,
- institutional metadata,
- and branch-level indices.

The predictive modeling layer is intended to answer whether organizational, operational, infrastructural, or human-readiness variables show stronger association with digital adoption friction.

---

## 17. Methodological Boundaries

FRICTA is an exploratory computational diagnostic framework.

The mathematical structure is designed to support reproducible institutional scoring, diagnostic interpretation, and future predictive validation.

At this stage, the framework does not treat theoretical weights as causal coefficients. Instead, weights are used as transparent analytical assumptions subject to sensitivity analysis.

The framework is designed to support future statistical validation, including reliability analysis, factor structure analysis, predictive modeling, and pilot-based diagnostic evaluation.

---

## 18. Summary

FRICTA transforms institutional survey responses into a structured computational diagnostic system.

The complete mathematical architecture follows this sequence:

1. Normalize variables into a friction-oriented 0–1 scale.
2. Construct auxiliary diagnostic variables.
3. Separate core scoring variables from auxiliary indicators.
4. Compute branch-level institutional indices.
5. Compute global Adoption Friction Scores.
6. Compare baseline and theoretically informed weighting scenarios.
7. Classify institutional friction severity.
8. Detect dominant institutional constraints.
9. Generate recommendation-oriented diagnostic outputs.
10. Prepare the system for statistical validation and predictive modeling.

This mathematical structure allows FRICTA to function not only as a survey analysis project, but as an interpretable institutional analytics framework for digital adoption friction in childcare institutions.