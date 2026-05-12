# FRICTA — OFFICIAL DERIVED VARIABLES

## Purpose

Derived variables are computational indicators created from existing survey variables. They are not direct survey responses. They are used to capture institutional patterns that cannot be observed from a single question.

At this stage, derived variables are exploratory and will be used for descriptive analysis, correlation analysis, segmentation, and later modeling.

---

# 1. digital_tool_variety

## Definition

Measures the number of digital tools currently used by the institution.

## Source Variables

- uses_excel
- uses_whatsapp
- uses_google_workspace
- uses_specialized_software
- other_tool

## Formula

\[
digital\_tool\_variety = \sum Tool_i
\]

## Interpretation

Higher values indicate greater diversity of existing digital tool usage.

---

# 2. digital_tool_constraint

## Definition

Friction-oriented version of digital tool variety.

## Source Variable

- digital_tool_variety

## Formula

\[
digital\_tool\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

## Interpretation

Higher values indicate lower digital tool diversity and higher digital integration constraint.

---

# 3. staff_per_device_ratio

## Definition

Estimates technological pressure by comparing staff size to available devices.

## Source Variables

- staff_size
- available_devices

## Formula

\[
staff\_per\_device\_ratio = \frac{staff\_size}{available\_devices + 1}
\]

## Methodological Note

The +1 prevents division by zero when institutions report zero available devices.

## Interpretation

Higher values indicate more staff pressure per available device.

---

# 4. children_per_staff_ratio

## Definition

Estimates institutional workload pressure by comparing children served to staff size.

## Source Variables

- children_served
- staff_size

## Formula

\[
children\_per\_staff\_ratio = \frac{children\_served}{staff\_size}
\]

## Interpretation

Higher values indicate greater operational load per staff category.

## Methodological Note

Because both variables are ordinal ranges, this ratio is treated as an approximate ordinal-derived indicator, not an exact demographic measure.

---

# 5. administrative_digital_burden

## Definition

Captures administrative burden under low digital support conditions.

## Source Variables

- admin_time_load_norm
- digital_usage_constraint
- administrative_digitization_constraint

## Formula

\[
administrative\_digital\_burden =
\frac{admin\_time\_load\_norm + digital\_usage\_constraint + administrative\_digitization\_constraint}{3}
\]

## Interpretation

Higher values indicate higher administrative workload combined with lower digital integration.

---

# 6. adoption_readiness_gap

## Definition

Captures the gap between perceived usefulness and actual willingness to adopt a digital tool.

## Source Variables

- perceived_digital_utility_norm
- willingness_constraint

## Formula

\[
adoption\_readiness\_gap =
|perceived\_digital\_utility\_norm - willingness\_constraint|
\]

## Interpretation

Higher values indicate a mismatch between perceived utility and institutional willingness.

---

# 7. operational_capacity_constraint

## Definition

Captures combined operational pressure from time scarcity and staffing scarcity.

## Source Variables

- time_constraint_norm
- staffing_constraint_norm

## Formula

\[
operational\_capacity\_constraint =
\frac{time\_constraint\_norm + staffing\_constraint\_norm}{2}
\]

## Interpretation

Higher values indicate stronger operational capacity limitations.

---

# 8. digital_exposure_constraint

## Definition

Captures low digital exposure based on prior implementation and current digital usage frequency.

## Source Variables

- previous_digital_implementation
- digital_usage_constraint

## Formula

\[
digital\_exposure\_constraint =
\frac{previous\_digital\_implementation + digital\_usage\_constraint}{2}
\]

## Methodological Note

The previous implementation variable is treated as a proxy for prior digital exposure, not as a direct measure of resistance.

## Interpretation

Higher values indicate lower prior digital exposure and lower current digital usage.

---

# 9. training_dependency_pressure

## Definition

Captures adoption pressure associated with limited training and low digital usage.

## Source Variables

- training_deficit_norm
- digital_usage_constraint

## Formula

\[
training\_dependency\_pressure =
\frac{training\_deficit\_norm + digital\_usage\_constraint}{2}
\]

## Interpretation

Higher values indicate stronger training-related adoption pressure.

---

# 10. implementation_friction_signal

## Definition

Captures direct implementation-related friction.

## Source Variables

- implementation_difficulty_norm
- system_change_resistance_norm

## Formula

\[
implementation\_friction\_signal =
\frac{implementation\_difficulty\_norm + system\_change\_resistance\_norm}{2}
\]

## Interpretation

Higher values indicate stronger perceived difficulty in implementation and system transition.