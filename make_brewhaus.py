"""Generate the fictional BrewHaus practice files used in the book.

Creates two CSVs in the data/ folder next to this script:
  brewhaus_customers.csv  5,000 deliberately messy customer rows (12 columns)
  brewhaus_sales.csv      daily revenue by store for two years

The random seed is fixed, so every run produces identical files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
SNAPSHOT = pd.Timestamp("2026-06-30")   # the day the export was taken
N_UNIQUE = 4950                          # real customers
N_DUPES = 50                             # exact duplicate rows -> 5,000 total

STORES = ["Downtown", "Riverside", "Airport", "University", "Mall"]
DRINKS = ["Latte", "Cappuccino", "Americano", "Cold Brew", "Mocha", "Tea", "Espresso"]

rng = np.random.default_rng(SEED)
out_dir = Path(__file__).parent / "data"
out_dir.mkdir(exist_ok=True)


def messy_date(d):
    """Write a date in one of three formats, as exports from different systems do."""
    if pd.isna(d):
        return np.nan
    style = rng.choice(3, p=[0.5, 0.3, 0.2])
    if style == 0:
        return d.strftime("%Y-%m-%d")
    if style == 1:
        return d.strftime("%m/%d/%Y")
    return d.strftime("%d %b %Y")


# ---- Clean "truth" for each customer ----
n = N_UNIQUE
store = rng.choice(STORES, n, p=[0.28, 0.20, 0.17, 0.20, 0.15])
loyal = rng.random(n) < 0.45
age = np.clip(rng.normal(36, 12, n), 16, 82).round()
drink = rng.choice(DRINKS, n, p=[0.24, 0.16, 0.16, 0.14, 0.10, 0.10, 0.10])

signup = SNAPSHOT - pd.to_timedelta(rng.integers(120, 1600, n), unit="D")
tenure_days = (SNAPSHOT - signup).days.values

# Churn risk: lower for loyalty members and long-standing customers,
# higher at the Airport (travellers) and for the youngest customers.
logit = (-0.7 - 1.3 * loyal + 0.9 * (store == "Airport")
         + 0.4 * (store == "University") - 0.0006 * tenure_days
         + 0.5 * (age < 22))
churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

# Visits in the last 90 days: churned customers stopped coming, so they
# only have visits from the start of that window.
base_rate = np.where(loyal, 13, 7) * rng.lognormal(0, 0.35, n)
visits = np.where(churned == 1, rng.poisson(1.6, n), rng.poisson(base_rate))

last_gap = np.where(churned == 1, rng.integers(61, 240, n), rng.integers(0, 60, n))
last_visit = SNAPSHOT - pd.to_timedelta(last_gap, unit="D")

drink_price = {"Latte": 5.4, "Cappuccino": 5.1, "Americano": 4.2, "Cold Brew": 5.6,
               "Mocha": 6.0, "Tea": 3.6, "Espresso": 3.4}
aov = (np.array([drink_price[d] for d in drink])
       + 1.3 * (store == "Airport") + 0.4 * (store == "Downtown")
       + 0.012 * (age - 36) + rng.normal(0, 0.55, n))
aov = np.clip(aov, 2.5, None).round(2)

# Average monthly spend over the customer's lifetime (not just recent visits).
lifetime_visits_per_month = base_rate / 3 * rng.lognormal(0, 0.25, n)
lifetime_visits_per_month *= np.where(churned == 1, 0.75, 1.0)
spend = (lifetime_visits_per_month * aov).round(2)
corporate = rng.choice(n, 5, replace=False)          # five office accounts
spend[corporate] = rng.uniform(520, 1400, 5).round(2)

# ---- Make it messy ----
city_true = rng.choice(["New York", "Boston", "Hoboken", "Jersey City"], n,
                       p=[0.62, 0.14, 0.12, 0.12])
ny_variants = ["New York", "new york", "NYC", "New York City", " New York", "New York "]
bos_variants = ["Boston", "boston", " Boston", "BOSTON"]
city = []
for c in city_true:
    if c == "New York":
        city.append(rng.choice(ny_variants, p=[0.55, 0.12, 0.13, 0.1, 0.05, 0.05]))
    elif c == "Boston":
        city.append(rng.choice(bos_variants, p=[0.7, 0.15, 0.1, 0.05]))
    elif c == "Jersey City":
        city.append(rng.choice(["Jersey City", "jersey city"], p=[0.85, 0.15]))
    else:
        city.append(c)

loyal_text = np.where(
    loyal,
    rng.choice(["Yes", "Y", "yes"], n, p=[0.75, 0.15, 0.10]),
    rng.choice(["No", "N", "no"], n, p=[0.75, 0.15, 0.10]),
).astype(object)
loyal_text[rng.random(n) < 0.01] = np.nan

age_messy = age.astype(float)
age_messy[rng.random(n) < 0.16] = np.nan
bad = rng.choice(n, 24, replace=False)
age_messy[bad[:12]] = 0
age_messy[bad[12:]] = 150

aov_text = []
for v in aov:
    r = rng.random()
    if r < 0.25:
        aov_text.append(f"${v:.2f}")
    elif r < 0.30:
        aov_text.append(f" {v:.2f}")
    elif r < 0.31:
        aov_text.append(np.nan)
    else:
        aov_text.append(f"{v:.2f}")

signup_text = [messy_date(d) for d in signup]
last_text = [messy_date(d) for d in last_visit]
for i in rng.choice(n, 8, replace=False):
    signup_text[i] = "unknown"

customers = pd.DataFrame({
    "customer_id": [f"C{10001 + i}" for i in range(n)],
    "signup_date": signup_text,
    "city": city,
    "store": store,
    "age": age_messy,
    "loyalty_member": loyal_text,
    "visits_90d": visits,
    "avg_order_value": aov_text,
    "monthly_spend": spend,
    "favorite_drink": drink,
    "last_visit_date": last_text,
    "churned": churned,
})

dupes = customers.sample(N_DUPES, random_state=SEED)
customers = pd.concat([customers, dupes]).sample(frac=1, random_state=SEED)
customers.to_csv(out_dir / "brewhaus_customers.csv", index=False)

# ---- Daily sales by store, two years ----
days = pd.date_range("2024-07-01", SNAPSHOT, freq="D")
store_base = {"Downtown": 3100, "Riverside": 2300, "Airport": 3400,
              "University": 2000, "Mall": 2200}
store_growth = {"Downtown": 0.05, "Riverside": 0.08, "Airport": 0.02,
                "University": 0.19, "Mall": -0.03}
rows = []
for s in STORES:
    t = (days - days[0]).days.values / 365
    month = days.month.values
    season = np.select([np.isin(month, [6, 7, 8]), np.isin(month, [9, 10, 11]),
                        np.isin(month, [12, 1, 2])], [0.85, 1.12, 1.05], 1.0)
    if s == "University":   # students leave for the summer
        season = np.where(np.isin(month, [6, 7, 8]), 0.6, season)
    weekend = np.where(days.dayofweek.values >= 5,
                       0.8 if s in ("Downtown", "University") else 1.15, 1.0)
    revenue = (store_base[s] * (1 + store_growth[s]) ** t * season * weekend
               * rng.normal(1, 0.07, len(days)))
    ticket = rng.normal(6.2 if s == "Airport" else 5.3, 0.15, len(days))
    rows.append(pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "store": s,
        "transactions": (revenue / ticket).round().astype(int),
        "revenue": revenue.round(2),
    }))
sales = pd.concat(rows).sort_values(["date", "store"])
sales.to_csv(out_dir / "brewhaus_sales.csv", index=False)

print(f"customers: {len(customers)} rows, sales: {len(sales)} rows -> {out_dir}")
