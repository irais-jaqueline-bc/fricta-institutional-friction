# FRICTA — Institutional Measurement Framework

## Overview

FRICTA operationalizes digital adoption friction through multiple institutional measurement branches representing infrastructural, organizational, operational, and human-level conditions associated with digital adoption in Mexican childcare institutions.

All normalized variables are directionally aligned under a friction-oriented scale ranging from 0 to 1, where:

- 0 = low institutional friction
- 1 = high institutional friction

The framework separates institutional friction into multiple conceptual dimensions in order to improve interpretability, reproducibility, and computational analysis.

---

# 1. Infrastructure Constraints

## Construct Definition

Infrastructure Constraints refer to technological and material limitations that restrict the implementation, integration, or sustained use of digital operational systems within childcare institutions.

---

## Core Variables

| Variable | Description |
|---|---|
| device_constraint | limited device availability |
| internet_constraint | unstable internet connectivity |
| digital_tool_constraint | low diversity of digital operational tools |
| resource_constraint_norm | perceived infrastructural and operational scarcity |

---

## Auxiliary Computational Variables

| Variable | Purpose |
|---|---|
| staff_per_device_ratio | technological pressure approximation |

---

## Interpretation

Higher values indicate:

- reduced access to technological devices,
- unstable internet connectivity,
- low digital tool diversity,
- and stronger infrastructural scarcity.

---

## Conceptual Scope

This branch captures infrastructural technological limitations associated with operational digital adoption capacity.

---

# 2. Organizational Constraints

## Construct Definition

Organizational Constraints refer to institutional rigidity, administrative barriers, and structural resistance associated with digital implementation processes.

---

## Core Variables

| Variable | Description |
|---|---|
| admin_disorganization | low administrative organization |
| implementation_difficulty_norm | perceived implementation complexity |
| system_change_resistance_norm | resistance to operational transition |
| previous_digital_implementation | proxy indicator of prior institutional digital exposure |

---

## Core Derived Variables

| Variable | Description |
|---|---|
| implementation_friction_signal | combined implementation-related friction signal |
| administrative_digital_burden | combined administrative burden under low digital support |

---

## Auxiliary Computational Variables

| Variable | Purpose |
|---|---|
| digital_exposure_constraint | low institutional digital exposure signal |
| adoption_readiness_gap | mismatch between perceived utility and willingness |

---

## Interpretation

Higher values indicate:

- lower organizational structure,
- stronger resistance to operational transition,
- higher implementation complexity,
- and lower prior institutional digital exposure.

---

## Methodological Note

The variable `previous_digital_implementation` is treated as a proxy indicator of institutional digital exposure rather than a direct measure of organizational resistance.

---

## Conceptual Scope

This branch captures structural organizational barriers associated with digital transition processes.

---

# 3. Operational Load

## Construct Definition

Operational Load refers to workload-related institutional conditions that reduce operational capacity available for digital transition and system adoption.

---

## Core Variables

| Variable | Description |
|---|---|
| admin_time_load_norm | administrative workload |
| time_constraint_norm | perceived lack of time |
| staffing_constraint_norm | personnel scarcity |

---

## Core Derived Variables

| Variable | Description |
|---|---|
| operational_capacity_constraint | combined operational saturation signal |

---

## Auxiliary Computational Variables

| Variable | Purpose |
|---|---|
| children_per_staff_ratio | approximate workload pressure estimation |

---

## Interpretation

Higher values indicate:

- elevated administrative workload,
- insufficient time availability,
- and staffing limitations.

---

## Conceptual Scope

This branch captures operational saturation associated with reduced implementation capacity.

---

# 4. Human Capacity & Adoption Readiness

## Construct Definition

Human Capacity & Adoption Readiness refers to institutional human preparedness, digital familiarity, and openness toward digital operational adoption.

---

## Core Variables

| Variable | Description |
|---|---|
| training_deficit_norm | insufficient digital training |
| digital_usage_constraint | low digital usage frequency |
| willingness_constraint | low willingness toward digital adoption |

---

## Core Derived Variables

| Variable | Description |
|---|---|
| training_dependency_pressure | training-related adoption pressure signal |

---

## Interpretation

Higher values indicate:

- insufficient digital training,
- lower operational digital familiarity,
- and reduced willingness toward technological adoption.

---

## Conceptual Scope

This branch captures institutional human-level readiness conditions associated with sustainable digital adoption.

---

# Auxiliary Variables

The following variables are treated as contextual or exploratory indicators and are not currently considered primary structural dimensions.

| Variable | Purpose |
|---|---|
| perceived_digital_utility_norm | auxiliary perception indicator |
| pilot_openness | pilot participation metadata |
| institution_type | institutional classification |
| state | geographic metadata |
| children_served | contextual demographic metadata |
| staff_size | contextual staffing metadata |

---

# Framework Structure Summary

| Branch | Core Variables | Core Derived Variables |
|---|---|---|
| Infrastructure Constraints | 4 | 1 |
| Organizational Constraints | 4 | 2 |
| Operational Load | 3 | 1 |
| Human Capacity & Adoption Readiness | 3 | 1 |

---

# Methodological Position

At this stage, the framework prioritizes:

- conceptual coherence,
- directional normalization consistency,
- computational interpretability,
- and exploratory institutional analytics.

The framework does not yet assume validated latent constructs or final weighted scoring structures. Further statistical validation and consistency analysis will be conducted during later stages of the project.