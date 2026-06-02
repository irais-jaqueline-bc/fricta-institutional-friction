# FRICTA — Methodology

## 1. Research Objective

The objective of FRICTA is to identify which institutional factors are most strongly associated with digital adoption friction in Mexican childcare institutions through a computational institutional diagnostics framework.

FRICTA seeks to transform institutional survey responses into interpretable computational diagnostics capable of identifying infrastructural, organizational, operational, and human-capacity constraints associated with digital adoption processes.

The framework is designed not only for descriptive analysis, but also for reproducible institutional scoring, exploratory predictive modeling, and recommendation-oriented diagnostics.

---

# 2. Study Design

FRICTA follows an exploratory cross-sectional observational design based on institutional survey responses collected from Mexican childcare institutions.

The project adopts a computational institutional analytics approach in which survey responses are transformed into normalized multidimensional institutional friction indicators through formal mathematical aggregation procedures.

The study is non-interventional during the current exploratory phase.

No institutional operational processes are modified during data collection.

---

# 3. Target Population

The target population includes Mexican childcare institutions involved in residential child protection, shelter, or institutional childcare services.

Target institutions include:

- casas hogar,
- orphanages,
- nonprofit residential childcare organizations,
- DIF-affiliated childcare institutions,
- NGOs,
- charitable childcare organizations,
- and related institutional childcare centers operating within Mexico.

---

# 4. Sampling Strategy

FRICTA uses a non-probabilistic voluntary-response sampling strategy.

Institutions were contacted through:

- direct email outreach,
- WhatsApp outreach,
- institutional directories,
- manual institutional searches,
- DIF-related directories,
- and publicly available organizational contact information.

Approximately 700 institutions were contacted across multiple Mexican states between May 4 and May 10.

The exploratory target sample for the first research phase is approximately 81 institutional responses distributed across multiple Mexican states, with the objective of achieving at least one institutional response per state when possible.

Due to institutional access limitations, response availability, and heterogeneous digital accessibility conditions, the sample should be interpreted as exploratory rather than nationally representative.

---

# 5. Inclusion Criteria

Responses are considered eligible when they meet the following conditions:

- the institution operates as a childcare-related organization,
- the institution provides residential, shelter, or institutional child support services,
- the response corresponds to a real institution,
- the survey response is sufficiently complete for computational scoring,
- and the institution operates within Mexico.

---

# 6. Exclusion Criteria

Responses may be excluded if they meet one or more of the following conditions:

- duplicate submissions,
- incomplete survey responses,
- test responses,
- invalid institutional responses,
- non-childcare organizations,
- or responses lacking sufficient scoring information.

---

# 7. Survey Platform

Data collection is conducted through Google Forms.

Survey URL:

https://docs.google.com/forms/d/e/1FAIpQLScC-h8pH4d6xmhTas_gXx5XRGd8aNggweSTWloQFzlFDACb0Q/viewform

The survey was designed to minimize institutional burden while collecting sufficient operational information for computational diagnostic analysis.

---

# 8. Survey Structure

The survey is divided into multiple institutional dimensions:

| Section | Purpose |
|---|---|
| General Information | Institutional metadata |
| Digital Infrastructure | Technological and infrastructural constraints |
| Administrative Processes | Organizational and workflow-related conditions |
| Digital Adoption | Institutional digital readiness |
| Friction Factors | Operational and implementation barriers |

The survey includes both direct institutional variables and variables later transformed into normalized computational indicators.

---

# 9. Dataset Structure

The dataset follows an institution-level structure where:

- each row represents one institution,
- each column represents one institutional variable,
- responses are anonymous by default,
- no personally identifiable child data is collected,
- no sensitive child-level information is stored,
- and institutional contact information is optional.

The dataset stores institutional geographic state information for regional analysis purposes.

Timestamp collection is not used during the current exploratory phase.

---

# 10. Ethical and Privacy Considerations

FRICTA prioritizes institutional anonymity and minimal-risk data collection.

The project does not collect:

- child names,
- personal child records,
- medical information,
- legal case information,
- or sensitive child-level data.

Participation is voluntary.

Institutional contact information is optional and is used only for future pilot participation or institutional follow-up.

The framework operates exclusively at the institutional level rather than the individual child level.

---

# 11. Pilot Structure

The project includes an exploratory pilot-validation stage.

At the current stage, approximately 7–9 institutions have shown potential interest in future pilot participation.

Within FRICTA, a pilot refers to institutional diagnostic validation through:

- diagnostic report generation,
- institutional friction scoring,
- recommendation delivery,
- onboarding procedures,
- exportable institutional diagnostics,
- and future testing of operational digital support tools.

The pilot stage is intended to evaluate whether institutional computational diagnostics can support real operational decision-making environments.

Future pilot stages may also include deployment of digital operational tools informed by FRICTA diagnostics.

---

# 12. Data Processing Pipeline

The computational processing pipeline is structured as follows:

Google Forms  
→ CSV Export  
→ Python Processing  
→ Pandas Data Cleaning  
→ Variable Normalization  
→ Derived Variable Construction  
→ scoring.py  
→ Branch Aggregation  
→ Adoption Friction Computation  
→ Diagnostic Classification  
→ Dashboard/API Integration

This architecture was designed to maximize reproducibility, transparency, and modular computational analysis.

---

# 13. Computational Stack

FRICTA uses a computational workflow primarily based on Python.

Planned and current tools include:

| Tool | Purpose |
|---|---|
| Python | Core computational environment |
| Pandas | Data cleaning and transformation |
| NumPy | Numerical computation |
| Scikit-learn | Exploratory predictive modeling |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| FastAPI | API architecture |
| Streamlit | Interactive dashboard development |

The computational architecture is intended to support reproducible institutional diagnostics and future scaling.

---

# 14. Statistical Analysis Plan

FRICTA includes both descriptive and exploratory computational analysis procedures.

Planned statistical procedures include:

- descriptive statistics,
- branch score distribution analysis,
- correlation matrix analysis,
- branch-level comparative analysis,
- sensitivity analysis,
- exploratory predictive modeling,
- feature importance estimation,
- and exploratory classification analysis.

If sample size permits, additional analyses may include:

- exploratory factor analysis,
- internal consistency analysis,
- and robustness analysis.

The framework prioritizes interpretability and computational transparency during the current exploratory phase.

---

# 15. Predictive Modeling Strategy

FRICTA plans to evaluate whether institutional variables can predict digital adoption friction outcomes.

Potential exploratory modeling approaches include:

- linear regression,
- regularized regression,
- decision trees,
- random forest models,
- feature importance estimation,
- and interpretable comparative modeling approaches.

Potential target variables include:

- Adoption Friction Score (baseline),
- Adoption Friction Score (theoretical weighting),
- branch-level scores,
- and institutional friction classification categories.

The predictive objective of the framework is:

"Which institutional factors most strongly predict digital adoption friction in Mexican childcare institutions?"

---

# 16. Methodological Position

FRICTA should be interpreted as an exploratory computational institutional diagnostics framework rather than a finalized validated psychometric instrument.

The framework prioritizes:

- interpretability,
- reproducibility,
- computational transparency,
- multidimensional institutional analysis,
- and recommendation-oriented diagnostics.

The current weighting structure represents theory-driven assumptions rather than empirically validated causal coefficients.

Further validation stages are planned through statistical analysis, pilot validation, and future longitudinal refinement.

---

# 17. Methodological Limitations

The current exploratory phase includes several limitations:

- non-probabilistic sampling,
- voluntary-response bias,
- possible regional imbalance,
- self-reported institutional data,
- limited sample size,
- and exploratory weighting assumptions.

The framework does not currently claim national representativeness.

Instead, the objective of the exploratory phase is to evaluate whether institutional computational diagnostics can meaningfully capture digital adoption friction patterns within childcare institutions.

---

# 18. Reproducibility and Future Development

FRICTA is designed as a reproducible computational research framework.

Future development stages may include:

- expanded institutional datasets,
- longitudinal validation,
- larger pilot implementations,
- predictive benchmarking,
- recommendation optimization,
- and operational digital tool deployment informed by institutional diagnostics.

The framework is intended to evolve from an exploratory institutional analytics system into a scalable computational diagnostics architecture for digital adoption analysis in childcare environments.