# FRICTA — OFFICIAL NORMALIZED VARIABLES TABLE

## SECTION 1 — GENERAL INFORMATION

| Variable | Type | Normalization | Normalized Variable | Notes |
|---|---|---|---|---|
| state | categorical_nominal | none | none | metadata only |
| institution_type | categorical_nominal | none | none | categorical descriptor |
| children_served | ordinal | optional later | children_served_norm | contextual metadata |
| staff_size | ordinal | optional later | staff_size_norm | used for derived variables |

---

# SECTION 2 — DIGITAL INFRASTRUCTURE

## Q5 — available_devices

### Direction

More devices = LESS friction

### Normalization

\[
device\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 0 | 1.00 |
| 1 | 0.67 |
| 2 | 0.33 |
| 3 | 0.00 |

---

## Q6 — internet_stability

### Direction

More stability = LESS friction

### Normalization

\[
internet\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 5 | 0.00 |
| 4 | 0.25 |
| 3 | 0.50 |
| 2 | 0.75 |
| 1 | 1.00 |

---

## Q7 — digital_tool_variety

### Direction

More digital tools = LESS friction

### Formula

\[
DigitalToolVariety = \sum Tool_i
\]

### Normalization

\[
digital\_tool\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Tool Count | Normalized |
|---|---|
| 0 | 1.00 |
| 1 | 0.75 |
| 2 | 0.50 |
| 3 | 0.25 |
| 4+ | 0.00 |

---

## Q14D — resource_constraint

### Direction

More resource scarcity = MORE friction

### Normalization

\[
resource\_constraint\_norm = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 1 | 0.00 |
| 2 | 0.25 |
| 3 | 0.50 |
| 4 | 0.75 |
| 5 | 1.00 |

---

# SECTION 3 — ADMINISTRATIVE PROCESSES

## Q8 — registration_system_type

### Direction

More digital registration = LESS friction

### Normalization

\[
registration\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 1 | 1.00 |
| 2 | 0.67 |
| 3 | 0.33 |
| 4 | 0.00 |

---

## Q9 — admin_time_load

### Direction

More admin time = MORE friction

### Normalization

\[
admin\_time\_load\_norm = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 1 | 0.00 |
| 2 | 0.33 |
| 3 | 0.67 |
| 4 | 1.00 |

---

## Q10 — administrative_organization

### Direction

Better organization = LESS friction

### Normalization

\[
admin\_disorganization = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 5 | 0.00 |
| 4 | 0.25 |
| 3 | 0.50 |
| 2 | 0.75 |
| 1 | 1.00 |

---

# SECTION 4 — DIGITAL ADOPTION

## Q11 — digital_usage_frequency

### Direction

More usage = LESS friction

### Normalization

\[
digital\_usage\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 4 | 0.00 |
| 3 | 0.33 |
| 2 | 0.67 |
| 1 | 1.00 |

---

## Q12 — previous_digital_implementation

### Direction

Previous implementation = LESS uncertainty/friction

### Binary Encoding

| Raw Value | Normalized |
|---|---|
| 1 | 0.00 |
| 0 | 1.00 |

---

## Q13 — implementation_difficulty

### Direction

More difficulty = MORE friction

### Normalization

\[
implementation\_difficulty\_norm = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 1 | 0.00 |
| 2 | 0.25 |
| 3 | 0.50 |
| 4 | 0.75 |
| 5 | 1.00 |

---

# SECTION 5 — FRICTION

## Q14A — time_constraint

### Direction

More time scarcity = MORE friction

### Normalization

\[
time\_constraint\_norm = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

---

## Q14B — staffing_constraint

### Direction

More staffing scarcity = MORE friction

### Normalization

\[
staffing\_constraint\_norm = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

---

## Q14C — training_deficit

### Direction

More training deficit = MORE friction

### Normalization

\[
training\_deficit\_norm = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

---

### Shared Normalization Table for Q14

| Raw Value | Normalized |
|---|---|
| 1 | 0.00 |
| 2 | 0.25 |
| 3 | 0.50 |
| 4 | 0.75 |
| 5 | 1.00 |

---

## Q15 — system_change_resistance

### Direction

More resistance = MORE friction

### Normalization

\[
system\_change\_resistance\_norm = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 1 | 0.00 |
| 2 | 0.25 |
| 3 | 0.50 |
| 4 | 0.75 |
| 5 | 1.00 |

---

## Q16 — perceived_digital_utility

### Direction

Higher utility perception = LESS friction

### Normalization

\[
utility\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 4 | 0.00 |
| 3 | 0.33 |
| 2 | 0.67 |
| 1 | 1.00 |

---

## Q17 — tool_adoption_willingness

### Direction

More willingness = LESS friction

### Normalization

\[
willingness\_constraint = \frac{x_{max}-x}{x_{max}-x_{min}}
\]

| Raw Value | Normalized |
|---|---|
| 5 | 0.00 |
| 4 | 0.25 |
| 3 | 0.50 |
| 2 | 0.75 |
| 1 | 1.00 |

---

## Q18 — pilot_openness

| Variable | Normalization |
|---|---|
| pilot_openness | none |

Metadata only.