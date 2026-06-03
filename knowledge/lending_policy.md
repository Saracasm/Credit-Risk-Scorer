# Northwind Bank — Consumer Lending Policy (Sample)

> This is a fictional sample policy used to demonstrate retrieval-augmented
> advising. It does not reflect any real institution's rules.

## Risk Tiers and Pricing

Northwind Bank classifies every applicant into a risk tier based on the model's
estimated probability of default (PD):

- **Prime** — PD below 20%. Approved with standard terms. APR 5.5%–7.5%.
- **Near-Prime** — PD 20%–40%. Approved with conditions. APR 8%–12%.
- **Subprime** — PD 40%–60%. Conditional approval; collateral or co-signer
  may be required. APR 13%–18%.
- **Deep Subprime** — PD above 60%. Manual review required; most applications
  in this band are declined or referred to secured-credit products. APR 20%+.

## Core Eligibility Criteria

To qualify for **Near-Prime or better** without manual review, an applicant
should generally meet all of the following:

- Credit utilization (revolving balances ÷ limits) **at or below 30%**.
- Debt-to-income ratio **at or below 0.40**.
- **No** payments 90 or more days past due in the file.
- No more than **one** 30–59 day late payment in the recent history.

To qualify for the **Prime** tier, stricter thresholds apply:

- Credit utilization **at or below 15%**.
- Debt-to-income ratio **at or below 0.30**.
- A clean payment history with **no** late payments of any severity.

## Loan Sizing

Maximum monthly payment is capped at **28% of estimated disposable income**
(monthly income net of existing debt). Maximum loan amount is derived from the
allowed payment and the tier's term: 60 months for Prime, 48 for Near-Prime,
36 for Subprime, and 24 for Deep Subprime.

## Conditions by Tier

- **Prime:** standard income documentation.
- **Near-Prime:** proof of income and a six-month employment check.
- **Subprime:** co-signer recommended; collateral may be required.
- **Deep Subprime:** co-signer required, collateral required; declining is the
  default recommendation unless compensating factors are documented.

## Decisioning Rules

The model and these thresholds support — but never replace — a human
underwriter. No final approval or denial is issued by the advisory system; it
provides analysis and eligibility guidance only. All adverse decisions must be
accompanied by the specific principal reasons (see the Fair Lending guide).
