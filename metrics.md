# FRICTA — Weighting Methodology

## Methodological Position

FRICTA computes two scoring scenarios:

1. Equal-weight baseline score
2. Theoretically informed weighted score

The theoretically informed weighting scenario is derived from digital adoption, organizational behavior, cognitive load, and institutional constraint literature, primarily based on the TOE framework, UTAUT, organizational readiness theory, cognitive load theory, and binding constraints methodology.

---

## Baseline Weighting Scenario

| Branch | Weight |
|---|---|
| Infrastructure Constraints Index (ICI) | 0.25 |
| Organizational Constraints Index (OCI) | 0.25 |
| Operational Load Index (OLI) | 0.25 |
| Human Capacity & Adoption Readiness Index (HCARI) | 0.25 |

---

## Theoretically Informed Weighting Scenario

| Branch | Weight | Theoretical Justification |
|---|---|---|
| Infrastructure Constraints Index (ICI) | 0.30 | TOE technological context; infrastructure operates as an enabling condition for digital adoption feasibility. |
| Organizational Constraints Index (OCI) | 0.30 | TOE organizational context; institutional structure, governance, staffing, and process maturity condition implementation sustainability. |
| Operational Load Index (OLI) | 0.25 | Cognitive load theory and administrative burden literature suggest operational saturation reduces institutional adoption capacity. |
| Human Capacity & Adoption Readiness Index (HCARI) | 0.15 | UTAUT-based adoption factors remain important but are conditioned by organizational and infrastructural constraints. |

---

## Baseline Formula

\[
AFS_{baseline} = \frac{ICI + OCI + OLI + HCARI}{4}
\]

---

## Theoretically Informed Formula

\[
AFS_{theoretical} =
0.30(ICI) +
0.30(OCI) +
0.25(OLI) +
0.15(HCARI)
\]

---

## Symbol Definitions

| Symbol | Meaning |
|---|---|
| ICI | Infrastructure Constraints Index |
| OCI | Organizational Constraints Index |
| OLI | Operational Load Index |
| HCARI | Human Capacity & Adoption Readiness Index |
| AFS | Adoption Friction Score |

---

## Methodological Note

The theoretically informed weighting scenario does not represent empirically validated causal weights. Instead, it represents a theory-driven weighting structure derived from institutional digital adoption literature.

The equal-weight baseline scenario is included to provide a neutral aggregation reference independent of theoretical assumptions.

Final score robustness will later be evaluated through sensitivity analysis between equal-weight and theoretically informed weighting schemes.

This approach follows transparency recommendations from composite indicator methodology literature, including OECD/JRC guidelines for indicator construction and weighting robustness assessment.