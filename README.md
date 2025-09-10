# Checkout Optimizer — A/B Experimentation at Scale

**A portfolio-ready, production-style A/B testing project** featuring:
- Proper randomization & stratification
- CUPED variance reduction
- Sequential monitoring with alpha-spending (O’Brien–Fleming)
- Cluster-robust SEs
- Bayesian lift estimates with a ROPE
- Uplift modeling (X-learner)
- Guardrails and decision framework
- Full simulation to validate Type I error and power
- Streamlit dashboard

## Quickstart

### 1) Install
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Generate synthetic data
```bash
python scripts/generate_data.py --n_users 10000 --days 14 --seed 42
```

This will create CSVs under `data/`:
- `assignments.csv`, `sessions.csv`, `events.csv`, `orders.csv`, `perf.csv`, `users.csv`

### 3) Run analysis (CUPED + frequentist + Bayesian + sequential)
```bash
python scripts/run_analysis.py --cuped --sequential --bayes --report data/report.json
```

### 4) Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

## Repo layout

```
ab-checkout-optimizer/
  README.md
  requirements.txt
  data/
    *.csv                  # synthetic data
  notebooks/
    01_power_and_cuped.ipynb
    02_sequential_analysis.ipynb
    03_heterogeneity_uplift.ipynb
    04_bandits_vs_ab.ipynb
  src/
    simulate.py            # user + funnel + treatment effects + drift
    assign.py              # hashing, stratification
    metrics.py             # GPV & guardrails
    analyze.py             # CUPED, diff-in-means, CRSE
    sequential.py          # alpha spending
    bayes.py               # posterior + ROPE
    uplift.py              # X-learner
  app/
    streamlit_app.py
  sql/
    create_tables.sql
    quality_checks.sql
  scripts/
    generate_data.py
    run_analysis.py
  slides/
    results_deck.pdf       # placeholder
```

## Decision rubric
- **Primary metric:** Gross Profit per Visitor (GPV) per unique visitor-day.
- **Guardrails:** refund rate, support tickets / 1k orders, checkout p95 latency, D7 retention.
- **Decision example:** Ship if GPV lift > 0 and P(worse than -0.5%) < 5% and no guardrail breaches.

## Notes
- Data are **synthetic** with configurable heterogeneity, novelty, drift, and logging loss.
- Intended for education/portfolio.


## Bandits (Multi‑Armed, scalable online allocation)

Run online policies on a continuous user stream with contextual, heterogenous effects:

```bash
python scripts/run_bandits.py --horizon 50000 --n_arms 3 --seed 123
```

Outputs:
- `data/bandits_report.json` — summary of **PCS**, **cumulative regret**, means, and allocation share
- `data/bandit_trace_*.csv` — per‑step traces for each policy

Policies included:
- `Thompson (GPV)` — maximizes continuous reward (GPV)
- `Thompson (Conversion)` — maximizes conversion probability
- `UCB1` — exploration via uncertainty bonus
- `Epsilon‑Greedy` — simple baseline
