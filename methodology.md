# Methodology

## Research Design

FRICTA adopts an exploratory mixed-methods computational research design focused on identifying, modeling, and diagnosing digital adoption friction within Mexican childcare institutions.

The project combines:

- institutional survey data,
- computational scoring methodologies,
- composite indicator construction,
- and pilot-based institutional diagnostics.

Rather than treating digital adoption as a binary condition, FRICTA conceptualizes adoption friction as a multidimensional operational phenomenon shaped by infrastructural constraints, organizational capacity, administrative workload, and institutional readiness.

The methodological architecture of FRICTA is structured in four sequential layers:

1. Data Collection  
2. Friction Modeling  
3. Institutional Diagnostics  
4. Pilot Intervention Framework

This structure allows the project to move beyond descriptive statistics and toward applied institutional analytics capable of generating operational recommendations for participating institutions.

---

# 1. Data Collection Layer

## Survey Design

Data collection is conducted through a structured institutional survey designed specifically for childcare and social assistance organizations operating in Mexico.

The survey focuses on capturing variables associated with:

- technological infrastructure,
- organizational structure,
- operational workload,
- institutional readiness,
- and perceived implementation barriers.

The instrument was intentionally designed to remain short and operationally accessible in order to reduce participation friction among institutions with limited administrative capacity.

The survey primarily contains:

- categorical variables,
- ordinal scales,
- operational frequency indicators,
- and perception-based adoption measures.

No sensitive child-level personal data is collected.

---

## Institutional Outreach

Institutions are contacted through publicly available communication channels including:

- email,
- institutional directories,
- nonprofit registries,
- and social assistance networks.

The outreach process is documented through an outreach tracking system containing:

- institution identifier,
- contact date,
- communication channel,
- response status,
- and follow-up status.

This process supports methodological transparency and response traceability.

---

## Variable Categories

Collected variables are grouped into five primary analytical categories:

| Category | Examples |
|---|---|
| Institutional Context | institution_type, state, children_count |
| Infrastructure | computer_access, internet_stability |
| Organizational Capacity | process_organization_level, admin_time_load |
| Friction & Constraints | lack_of_time, lack_of_training |
| Adoption Readiness | perceived_usefulness, adoption_openness |

---

# 2. Friction Modeling Layer

## Composite Index Construction

FRICTA models digital adoption friction through the construction of composite institutional indices.

The framework computes four primary analytical indices:

| Index | Purpose |
|---|---|
| Infrastructure Constraints Index (ICI) | Measures infrastructural limitations |
| Organizational Constraints Index (OCI) | Measures organizational and operational barriers |
| Operational Load Index (OLI) | Measures administrative saturation and workload pressure |
| Human Capacity & Adoption Readiness Index (HCARI) | Measures institutional openness and human readiness toward digital adoption |

These indices are later aggregated into the global Adoption Friction Score (AFS).

---

## Data Normalization

Variables are normalized onto comparable scales prior to aggregation.

Depending on variable structure, normalization methods may include:

- min-max normalization,
- ordinal scaling conversion,
- or binary transformation.

Directional consistency is preserved across all variables such that:

Higher scores consistently represent higher friction or stronger constraints.

---

## Weighting Methodology

FRICTA computes two scoring scenarios:

### Equal-weight baseline model

All analytical dimensions contribute equally to the final score.

$$
AFS_{baseline}=\frac{ICI+OCI+OLI+HCARI}{4}
$$

This baseline scenario is used to avoid premature assumptions regarding causal dominance among dimensions.

---

### Theoretically informed weighting model

FRICTA additionally computes a theoretically informed weighted scenario derived from:

- TOE,
- UTAUT,
- organizational inertia theory,
- cognitive load theory,
- and binding constraints methodology.

$$
AFS_{theoretical}=0.30(ICI)+0.30(OCI)+0.25(OLI)+0.15(HCARI)
$$

The weighted structure reflects the theoretical assumption that infrastructural and organizational constraints act as higher-order institutional bottlenecks conditioning adoption feasibility.

These weights are not presented as empirically validated causal coefficients, but as theory-driven analytical scenarios.

---

## Sensitivity Analysis

To reduce arbitrariness in composite indicator construction, FRICTA compares both weighting schemes through sensitivity analysis.

This process evaluates whether institutional rankings and diagnostic outputs remain stable across different weighting assumptions.

Sensitivity analysis follows OECD composite indicator construction principles emphasizing:

- transparency,
- robustness,
- and methodological reproducibility.

---

# 3. Institutional Diagnostics Layer

## Friction Profiling

FRICTA does not only generate a global score.

The framework additionally classifies institutions according to dominant friction patterns.

Examples include:

- infrastructure-constrained institutions,
- organization-constrained institutions,
- operational-overload institutions,
- readiness-constrained institutions.

This diagnostic logic transforms raw survey responses into interpretable institutional profiles.

---

## Constraint-Oriented Interpretation

FRICTA integrates the logic of binding constraints diagnostics.

The framework assumes that not all barriers contribute equally to institutional stagnation.

Instead, the model seeks to identify which constraint appears most operationally restrictive within each institution.

This allows the framework to move from descriptive measurement toward actionable prioritization.

---

# 4. Pilot Intervention Framework

## Pilot Objective

The pilot phase is not designed as a full technological deployment.

Instead, pilots focus on validating whether FRICTA can generate operationally meaningful institutional diagnostics and implementation recommendations.

Participating institutions receive:

- institutional friction analysis,
- adoption readiness interpretation,
- constraint identification,
- and tailored strategic recommendations.

FRICTA therefore provides:

the diagnosis and implementation roadmap,
but not the technological intervention itself.

---

## Pilot Validation Logic

Pilot validation evaluates whether:

- institutional profiles are coherent with operational realities,
- recommendations are perceived as useful,
- and diagnostic outputs align with observed institutional conditions.

This phase acts as an applied validation layer connecting computational modeling with real institutional environments.

---

# Analytical Framework

The analytical structure of FRICTA combines:

- computational social research,
- institutional analytics,
- digital transformation research,
- and applied organizational diagnostics.

The framework conceptualizes digital adoption not as a purely technological process, but as the interaction between:

- infrastructure,
- organizational structure,
- operational capacity,
- and human adaptability.

This multidimensional perspective allows FRICTA to analyze why institutions with similar technological resources may experience radically different implementation outcomes.

---

# Methodological Position

FRICTA is positioned as an exploratory institutional analytics framework operating under constrained-sample conditions.

The project does not claim causal certainty.

Instead, it seeks to:

- identify friction patterns,
- estimate institutional constraints,
- model adoption feasibility,
- and generate reproducible diagnostic methodologies for future large-scale validation.