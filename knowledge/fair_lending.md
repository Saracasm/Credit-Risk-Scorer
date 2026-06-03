# Northwind Bank — Fair Lending & Adverse Action (Sample)

> Fictional sample summary for demonstrating retrieval-augmented advising.
> Not legal advice.

## Equal Credit Opportunity (Fair Lending Principles)

Credit decisions at Northwind must be based solely on creditworthiness factors.
The advisor must **never** base risk explanations or recommendations on protected
characteristics such as race, color, religion, national origin, sex, marital
status, or age (except where age is used in a permitted, empirically derived
scoring context). If a user asks the advisor to weigh any protected
characteristic, decline and redirect to legitimate credit factors.

## Adverse Action Reasons

When an application is declined or approved on worse terms, the applicant is
entitled to the **specific principal reasons** for that decision. Good adverse
action reasons map directly to the model's top risk drivers, for example:

- "Proportion of credit limits being used is too high" (high utilization).
- "Serious delinquency on file" (90+ day late payments).
- "Level of existing debt relative to income is too high" (high DTI).

Reasons must be specific and accurate — never vague ("did not meet our
standards") and never a protected characteristic.

## Explainability Expectations

Because Northwind uses a machine-learning model, every risk explanation should
be traceable to model evidence (SHAP factor contributions). The advisor should
translate those factors into plain language the applicant can act on, and should
avoid implying the decision is final or automated — a human underwriter owns the
final decision.

## Transparency to Applicants

When asked "why," give the real top factors and their direction. When asked
"how to improve," give concrete, lawful, actionable steps. Do not speculate
beyond what the model evidence and policy support.
