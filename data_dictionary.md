# FRICTA — Data Dictionary

## Overview

This document defines all variables used within the FRICTA framework, including:

- raw survey variables,
- normalized variables,
- derived variables,
- metadata variables,
- branch assignments,
- and computational roles.

The dictionary follows the official mathematical formalization and normalization procedures defined in the FRICTA framework.

All normalized scoring variables follow a friction-oriented scale:

| Value | Interpretation |
|---|---|
| 0 | Low friction |
| 1 | High friction |

---

# SECTION 1 — Metadata Variables

These variables are used for contextual interpretation, segmentation, visualization, and exploratory analysis.

They are not directly included in branch aggregation formulas.

| Variable | Type | Description | Role |
|---|---|---|---|
| `state` | categorical_nominal | Mexican state where the institution operates | metadata |
| `institution_type` | categorical_nominal | Institutional classification | metadata |
| `children_served` | ordinal | Approximate number of children served | metadata / derived-variable input |
| `staff_size` | ordinal | Approximate institutional staff size | metadata / derived-variable input |
| `pilot_openness` | binary | Indicates willingness to participate in pilot testing | metadata |

---

# SECTION 2 — Infrastructure Variables

## Q5 — available_devices

| Field | Value |
|---|---|
| Raw Variable | `available_devices` |
| Normalized Variable | `device_constraint` |
| Branch | ICI |
| Role | core |
| Direction | More devices = lower friction |
| Normalization | reverse min-max |
| Formula | `device_constraint = (x_max - x) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 0 | No devices | 1.00 |
| 1 | Limited devices | 0.67 |
| 2 | Moderate devices | 0.33 |
| 3 | Sufficient devices | 0.00 |

---

## Q6 — internet_stability

| Field | Value |
|---|---|
| Raw Variable | `internet_stability` |
| Normalized Variable | `internet_constraint` |
| Branch | ICI |
| Role | core |
| Direction | More stability = lower friction |
| Normalization | reverse min-max |
| Formula | `internet_constraint = (x_max - x) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Very unstable / no access | 1.00 |
| 2 | Very unstable | 0.75 |
| 3 | Unstable / moderate | 0.50 |
| 4 | Stable | 0.25 |
| 5 | Very stable | 0.00 |

---

## Q7 — digital_tool_variety

| Field | Value |
|---|---|
| Derived Variable | `digital_tool_variety` |
| Normalized Variable | `digital_tool_constraint` |
| Branch | ICI |
| Role | core |
| Direction | More tools = lower friction |
| Formula | `digital_tool_variety = sum(Tool_i)` |

### Tool Indicators

- `uses_excel`
- `uses_whatsapp`
- `uses_google_workspace`
- `uses_specialized_software`
- `other_tool`

### Constraint Normalization

| Tool Count | Normalized |
|---|---|
| 0 | 1.00 |
| 1 | 0.75 |
| 2 | 0.50 |
| 3 | 0.25 |
| 4+ | 0.00 |

---

## Q14D — resource_constraint

| Field | Value |
|---|---|
| Raw Variable | `resource_constraint` |
| Normalized Variable | `resource_constraint_norm` |
| Branch | ICI |
| Role | core |
| Direction | More scarcity = more friction |
| Normalization | direct min-max |
| Formula | `resource_constraint_norm = (x - x_min) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Very low scarcity | 0.00 |
| 2 | Low scarcity | 0.25 |
| 3 | Moderate scarcity | 0.50 |
| 4 | High scarcity | 0.75 |
| 5 | Severe scarcity | 1.00 |

---

# SECTION 3 — Organizational Variables

## Q8 — registration_system_type

| Field | Value |
|---|---|
| Raw Variable | `registration_system_type` |
| Normalized Variable | `administrative_digitization_constraint` |
| Branch | auxiliary |
| Role | auxiliary |
| Direction | More digital registration = lower friction |
| Normalization | reverse min-max |
| Formula | `administrative_digitization_constraint = (x_max - x) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Fully manual | 1.00 |
| 2 | Mostly manual / Excel | 0.67 |
| 3 | Mixed | 0.33 |
| 4 | Mostly digital / software | 0.00 |

---

## Q9 — admin_time_load

| Field | Value |
|---|---|
| Raw Variable | `admin_time_load` |
| Normalized Variable | `admin_time_load_norm` |
| Branch | OLI |
| Role | core |
| Direction | More administrative time = more friction |
| Normalization | direct min-max |
| Formula | `admin_time_load_norm = (x - x_min) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Less administrative time | 0.00 |
| 2 | Low-moderate administrative time | 0.33 |
| 3 | Moderate-high administrative time | 0.67 |
| 4 | Highest administrative time | 1.00 |

---

## Q10 — administrative_organization

| Field | Value |
|---|---|
| Raw Variable | `administrative_organization` |
| Normalized Variable | `admin_disorganization` |
| Branch | OCI |
| Role | core |
| Direction | Better organization = lower friction |
| Normalization | reverse min-max |
| Formula | `admin_disorganization = (x_max - x) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Very low organization | 1.00 |
| 2 | Low organization | 0.75 |
| 3 | Medium organization | 0.50 |
| 4 | High organization | 0.25 |
| 5 | Very high organization | 0.00 |

---

# SECTION 4 — Digital Adoption Variables

## Q11 — digital_usage_frequency

| Field | Value |
|---|---|
| Raw Variable | `digital_usage_frequency` |
| Normalized Variable | `digital_usage_constraint` |
| Branch | HCARI |
| Role | core |
| Direction | More frequent digital usage = lower friction |
| Normalization | reverse min-max |
| Formula | `digital_usage_constraint = (x_max - x) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Never | 1.00 |
| 2 | Rarely | 0.67 |
| 3 | Weekly | 0.33 |
| 4 | Daily | 0.00 |

---

## Q12 — previous_digital_implementation

| Field | Value |
|---|---|
| Raw Variable | `previous_digital_implementation` |
| Normalized Variable | `previous_digital_implementation` |
| Branch | auxiliary |
| Role | auxiliary contextual proxy |
| Direction | Previous implementation = lower uncertainty/friction |
| Normalization | binary contextual encoding |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 0 | No previous implementation | 0.50 |
| 1 | Previous implementation exists | 0.00 |

---

## Q13 — implementation_difficulty

| Field | Value |
|---|---|
| Raw Variable | `implementation_difficulty` |
| Normalized Variable | `implementation_difficulty_norm` |
| Branch | OCI |
| Role | core |
| Direction | More implementation difficulty = more friction |
| Normalization | direct min-max |
| Formula | `implementation_difficulty_norm = (x - x_min) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Very easy | 0.00 |
| 2 | Easy | 0.25 |
| 3 | Moderate | 0.50 |
| 4 | Difficult | 0.75 |
| 5 | Very difficult | 1.00 |

---

# SECTION 5 — Friction Variables

## Q14A — time_constraint

| Field | Value |
|---|---|
| Raw Variable | `time_constraint` |
| Normalized Variable | `time_constraint_norm` |
| Branch | OLI |
| Role | core |
| Direction | More time scarcity = more friction |
| Normalization | direct min-max |

---

## Q14B — staffing_constraint

| Field | Value |
|---|---|
| Raw Variable | `staffing_constraint` |
| Normalized Variable | `staffing_constraint_norm` |
| Branch | OLI |
| Role | core |
| Direction | More staffing scarcity = more friction |
| Normalization | direct min-max |

---

## Q14C — training_deficit

| Field | Value |
|---|---|
| Raw Variable | `training_deficit` |
| Normalized Variable | `training_deficit_norm` |
| Branch | HCARI |
| Role | core |
| Direction | More training deficit = more friction |
| Normalization | direct min-max |

---

## Shared Encoding for Q14 Variables

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | None | 0.00 |
| 2 | Low | 0.25 |
| 3 | Medium | 0.50 |
| 4 | High | 0.75 |
| 5 | Very high | 1.00 |

---

## Q15 — system_change_resistance

| Field | Value |
|---|---|
| Raw Variable | `system_change_resistance` |
| Normalized Variable | `system_change_resistance_norm` |
| Branch | OCI |
| Role | core |
| Direction | More resistance = more friction |
| Normalization | direct min-max |
| Formula | `system_change_resistance_norm = (x - x_min) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Very easy to change | 0.00 |
| 2 | Easy to change | 0.25 |
| 3 | Neutral | 0.50 |
| 4 | Difficult to change | 0.75 |
| 5 | Very difficult to change | 1.00 |

---

## Q16 — perceived_digital_utility

| Field | Value |
|---|---|
| Raw Variable | `perceived_digital_utility` |
| Normalized Variable | `perceived_digital_utility_norm` |
| Branch | auxiliary |
| Role | auxiliary perception indicator |
| Direction | Higher perceived utility = lower friction |
| Normalization | reverse min-max |
| Formula | `perceived_digital_utility_norm = (x_max - x) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | No perceived utility | 1.00 |
| 2 | Low perceived utility | 0.67 |
| 3 | Some perceived utility | 0.33 |
| 4 | High perceived utility | 0.00 |

---

## Q17 — tool_adoption_willingness

| Field | Value |
|---|---|
| Raw Variable | `tool_adoption_willingness` |
| Normalized Variable | `willingness_constraint` |
| Branch | HCARI |
| Role | core |
| Direction | More willingness = lower friction |
| Normalization | reverse min-max |
| Formula | `willingness_constraint = (x_max - x) / (x_max - x_min)` |

### Encoding

| Raw Value | Meaning | Normalized |
|---|---|---|
| 1 | Not willing | 1.00 |
| 2 | Low willingness | 0.75 |
| 3 | Neutral | 0.50 |
| 4 | Willing | 0.25 |
| 5 | Very willing | 0.00 |

---

# SECTION 6 — Derived Variables

| Variable | Formula | Role |
|---|---|---|
| `digital_tool_variety` | `sum(Tool_i)` | derived/core |
| `digital_tool_constraint` | reverse normalized tool variety | derived/core |
| `staff_per_device_ratio` | `staff_size / (available_devices + 1)` | auxiliary |
| `children_per_staff_ratio` | `children_served / staff_size` | auxiliary |
| `administrative_digital_burden` | `average(admin_time_load_norm, digital_usage_constraint, administrative_digitization_constraint)` | auxiliary |
| `adoption_readiness_gap` | `abs(perceived_digital_utility_norm - willingness_constraint)` | auxiliary |
| `operational_capacity_constraint` | `average(time_constraint_norm, staffing_constraint_norm)` | auxiliary |
| `digital_exposure_constraint` | `average(previous_digital_implementation, digital_usage_constraint)` | auxiliary |
| `training_dependency_pressure` | `average(training_deficit_norm, digital_usage_constraint)` | auxiliary |
| `implementation_friction_signal` | `average(implementation_difficulty_norm, system_change_resistance_norm)` | auxiliary |

---

# SECTION 7 — Branch Assignment Summary

| Branch | Variables |
|---|---|
| ICI | `device_constraint`, `internet_constraint`, `digital_tool_constraint`, `resource_constraint_norm` |
| OCI | `admin_disorganization`, `implementation_difficulty_norm`, `system_change_resistance_norm` |
| OLI | `admin_time_load_norm`, `time_constraint_norm`, `staffing_constraint_norm` |
| HCARI | `training_deficit_norm`, `digital_usage_constraint`, `willingness_constraint` |

---

# SECTION 8 — Variable Role Summary

| Role | Meaning |
|---|---|
| core | Included directly in branch aggregation |
| auxiliary | Used for diagnostics, interpretation, recommendations, or modeling |
| metadata | Contextual or descriptive variable |