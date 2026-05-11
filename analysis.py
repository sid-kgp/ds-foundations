import pandas as pd
import matplotlib.pyplot as plt

# ── 1. LOAD ──────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasets/population/main/data/population.csv"
df = pd.read_csv(url)

# ── 2. EXPLORE ───────────────────────────────────────────
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# ── 3. FILTER to latest year ─────────────────────────────
latest_year = df["Year"].max()
print(f"\nLatest year in data: {latest_year}")
df_latest = df[df["Year"] == latest_year].copy()

# ── 4. CLEAN ─────────────────────────────────────────────
# Remove missing and zero values
df_latest = df_latest.dropna(subset=["Value"])
df_latest = df_latest[df_latest["Value"] > 0]

# SMARTER FIX: World Bank uses exactly 3-LETTER codes for real countries
# Aggregates/groups have codes like "IBT", "LMY", "EAP" — also 3 letters
# BUT they always contain keywords like "income", "total", "dividend" in name
# Easiest reliable filter: exclude any name containing these words
exclude_keywords = [
    "income", "total", "dividend", "region", "world",
    "africa", "asia", "europe", "pacific", "caribbean",
    "america", "middle east", "north africa", "saharan",
    "oecd", "euro", "ibrd", "ida ", "idb", "blend",
    "demographic", "fragile", "heavily", "debt",
    "developing", "developed", "small state", "situation"
]

# Convert to lowercase for comparison, check if any keyword appears
mask = df_latest["Country Name"].str.lower().apply(
    lambda name: not any(kw in name for kw in exclude_keywords)
)
df_countries = df_latest[mask].copy()

print(f"\nRows after cleaning: {len(df_countries)}")
print("\nSample of countries kept:")
print(df_countries["Country Name"].head(10).tolist())

# ── 5. SUMMARISE ─────────────────────────────────────────
top10 = df_countries.sort_values("Value", ascending=False).head(10)
print("\nTop 10 most populous countries:")
print(top10[["Country Name", "Country Code", "Value"]].to_string(index=False))

# ── 6. CHART ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
bars = plt.barh(top10["Country Name"], top10["Value"] / 1_000_000,
                color="steelblue")

# Add value labels inside each bar
for bar, val in zip(bars, top10["Value"] / 1_000_000):
    plt.text(bar.get_width() - 20, bar.get_y() + bar.get_height()/2,
             f"{val:.0f}M", va="center", ha="right",
             color="white", fontweight="bold", fontsize=9)

plt.xlabel("Population (millions)")
plt.title(f"Top 10 Most Populous Countries ({latest_year})")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top10_population.png", dpi=150)
plt.show()
print("\nChart saved as top10_population.png ✓")