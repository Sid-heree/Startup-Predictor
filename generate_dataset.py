import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 620

os.makedirs("data", exist_ok=True)

industries = ["SaaS", "FinTech", "HealthTech", "EdTech", "E-Commerce",
              "AI/ML", "CleanTech", "BioTech", "Logistics", "Cybersecurity"]

funding_stage = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C"]

funding_amount   = np.random.exponential(scale=3.5, size=n).clip(0.1, 50.0).round(2)  # in $M
team_size        = np.random.randint(2, 120, size=n)
years_active     = np.random.uniform(0.5, 12.0, size=n).round(1)
market_size      = np.random.exponential(scale=800, size=n).clip(10, 5000).round(1)    # $M TAM
num_competitors  = np.random.randint(1, 80, size=n)
revenue_year1    = (funding_amount * np.random.uniform(0.05, 0.4, size=n)).round(2)
burn_rate        = (funding_amount * np.random.uniform(0.1, 0.6, size=n) / 12).round(3)
customer_count   = (team_size * np.random.uniform(5, 80, size=n)).astype(int)
product_score    = np.random.randint(1, 11, size=n)   # 1-10 survey score
marketing_spend  = (funding_amount * np.random.uniform(0.05, 0.25, size=n)).round(2)
industry_col     = np.random.choice(industries, size=n)
stage_col        = np.random.choice(funding_stage, size=n)

# --- Future Revenue (regression target) ---
# Realistic formula with noise
future_revenue = (
    revenue_year1 * 2.5
    + funding_amount * 0.3
    + (team_size / 10) * 0.5
    + (market_size / 500) * 1.2
    + product_score * 0.4
    - burn_rate * 2
    + np.random.normal(0, 1.5, size=n)
).clip(0.01).round(2)

# --- Success label (classification target) ---
# Weighted score
success_score = (
    (funding_amount / 50) * 25
    + (team_size / 120) * 15
    + (years_active / 12) * 10
    + (market_size / 5000) * 20
    + (product_score / 10) * 20
    + (customer_count / (team_size * 80)) * 10
    - (num_competitors / 80) * 10
    - (burn_rate / funding_amount.clip(0.1)) * 5
    + np.random.normal(0, 8, size=n)
)
success = (success_score >= success_score.mean()).astype(int)

df = pd.DataFrame({
    "funding_amount_m":  funding_amount,
    "team_size":         team_size,
    "years_active":      years_active,
    "market_size_m":     market_size,
    "num_competitors":   num_competitors,
    "revenue_year1_m":   revenue_year1,
    "burn_rate_m":       burn_rate,
    "customer_count":    customer_count,
    "product_score":     product_score,
    "marketing_spend_m": marketing_spend,
    "industry":          industry_col,
    "funding_stage":     stage_col,
    "future_revenue_m":  future_revenue,
    "success":           success,          # 1 = Success, 0 = Fail
})

df.to_csv("data/startup_data.csv", index=False)
print(f"✅ Dataset created: {len(df)} rows, {df.columns.tolist()}")
print(f"   Success rate: {df['success'].mean():.1%}")
print(f"   Avg future revenue: ${df['future_revenue_m'].mean():.2f}M")