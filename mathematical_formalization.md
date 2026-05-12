# FRICTA — Mathematical Formalization

## 1. Purpose

This document defines the mathematical structure of the FRICTA framework.

FRICTA models digital adoption friction as a normalized, multidimensional institutional construct. The objective is to transform survey responses into interpretable institutional diagnostics through:

1. variable normalization,
2. derived variable construction,
3. branch-level index computation,
4. global adoption friction scoring,
5. institutional classification,
6. and recommendation-oriented diagnostics.

All scoring variables follow a friction-oriented scale:

| Value | Interpretation |
|---|---|
| 0 | Low friction |
| 1 | High friction |

This ensures directional consistency across all institutional indicators.

---

## 2. Redundancy Control Principle

Before computing branch-level indices, FRICTA separates variables into two categories:

1. **Core scoring variables**
2. **Auxiliary diagnostic variables**

Core scoring variables are used directly to compute institutional indices.

Auxiliary diagnostic variables are used for interpretation, visualization, reporting, and recommendation generation, but are not included directly in the primary index formulas.

This distinction prevents accidental double weighting when a derived variable reuses normalized variables that are already included in a branch index.

For example:

\[
implementation\_friction\_signal =
\frac{
implementation\_difficulty\_norm +
system\_change\_resistance\_norm
}{2}
\]

Since both `implementation_difficulty_norm` and `system_change_resistance_norm` already appear as core organizational variables, `implementation_friction_signal` is treated as an auxiliary diagnostic variable rather than a core scoring variable.

---

## 3. Core vs Auxiliary Variable Policy

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

## 4. Final Core Variable Assignment

### 4.1 Infrastructure Constraints Index — ICI

Core variables:

| Variable | Meaning |
|---|---|
| device_constraint | Limited device availability |
| internet_constraint | Internet instability |
| digital_tool_constraint | Low diversity of digital tools |
| resource_constraint_norm | Perceived infrastructural-operational scarcity |

Auxiliary variables:

| Variable | Status |
|---|---|
| staff_per_device_ratio | Auxiliary diagnostic variable |

---

### 4.2 Organizational Constraints Index — OCI

Core variables:

| Variable | Meaning |
|---|---|
| admin_disorganization | Low administrative organization |
| implementation_difficulty_norm | Perceived implementation difficulty |
| system_change_resistance_norm | Resistance to changing current systems |

Auxiliary variables:

| Variable | Status |
|---|---|
| previous_digital_implementation | Auxiliary contextual proxy |
| implementation_friction_signal | Auxiliary diagnostic variable |
| administrative_digital_burden | Auxiliary diagnostic variable |
| digital_exposure_constraint | Auxiliary diagnostic variable |
| adoption_readiness_gap | Auxiliary diagnostic variable |

---

### 4.3 Operational Load Index — OLI

Core variables:

| Variable | Meaning |
|---|---|
| admin_time_load_norm | Administrative time burden |
| time_constraint_norm | Lack of available time |
| staffing_constraint_norm | Staffing scarcity |

Auxiliary variables:

| Variable | Status |
|---|---|
| operational_capacity_constraint | Auxiliary diagnostic variable |
| children_per_staff_ratio | Auxiliary contextual approximation |

---

### 4.4 Human Capacity & Adoption Readiness Index — HCARI

Core variables:

| Variable | Meaning |
|---|---|
| training_deficit_norm | Lack of digital training |
| digital_usage_constraint | Low digital usage frequency |
| willingness_constraint | Low willingness to adopt digital tools |

Auxiliary variables:

| Variable | Status |
|---|---|
| training_dependency_pressure | Auxiliary diagnostic variable |
| perceived_digital_utility_norm | Auxiliary perception indicator |
| pilot_openness | Pilot participation metadata |

---

## 5. Mathematical Implication

The primary FRICTA score is computed only from core scoring variables.

Auxiliary variables do not directly affect the primary Adoption Friction Score.

This design keeps the model:

- interpretable,
- reproducible,
- mathematically stable,
- and protected against unintended signal duplication.

Auxiliary variables remain important for institutional diagnostics, but they operate after the main friction scores are computed.