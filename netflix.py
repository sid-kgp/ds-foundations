import pandas as pd
import matplotlib.pyplot as plt

# ── 1. LOAD ──────────────────────────────────────────────
url = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/netflix_titles.csv"
df = pd.read_csv(url)

# ── 2. MISSING VALUES AUDIT ──────────────────────────────
print("=" * 55)
print("MISSING VALUES:")
missing = pd.DataFrame({
    "Count": df.isnull().sum(),
    "%": (df.isnull().sum() / len(df) * 100).round(1)
})
print(missing[missing["Count"] > 0])

# ── 3. MOVIES VS TV SHOWS ────────────────────────────────
# value_counts() — counts how many of each category exists
print("\n" + "=" * 55)
print("MOVIES VS TV SHOWS:")
type_counts = df["type"].value_counts()
print(type_counts)
print(f"\nMovies are {type_counts['Movie']/len(df)*100:.1f}% of all content")

# ── 4. TOP 10 COUNTRIES ──────────────────────────────────
# Some rows have multiple countries — we take the first one
print("\n" + "=" * 55)
print("TOP 10 CONTENT-PRODUCING COUNTRIES:")
df["primary_country"] = df["country"].str.split(",").str[0].str.strip()
top_countries = df["primary_country"].value_counts().head(10)
print(top_countries)

# ── 5. RATINGS BREAKDOWN ─────────────────────────────────
print("\n" + "=" * 55)
print("CONTENT RATINGS:")
print(df["rating"].value_counts())

# ── 6. CONTENT ADDED PER YEAR ────────────────────────────
print("\n" + "=" * 55)
print("CONTENT ADDED PER YEAR:")
# date_added looks like "January 1, 2020" — extract the year
df["year_added"] = pd.to_datetime(df["date_added"].str.strip(),
                                   format="mixed",
                                   errors="coerce").dt.year
yearly = df["year_added"].value_counts().sort_index()
print(yearly)

# ── 7. TOP 10 DIRECTORS ──────────────────────────────────
print("\n" + "=" * 55)
print("TOP 10 DIRECTORS (excluding missing):")
top_directors = df["director"].dropna().value_counts().head(10)
print(top_directors)

# ── 8. MOVIE DURATION ANALYSIS ───────────────────────────
print("\n" + "=" * 55)
print("MOVIE DURATION ANALYSIS:")
movies = df[df["type"] == "Movie"].copy()
# duration looks like "90 min" — extract just the number
movies["minutes"] = movies["duration"].str.replace(" min", "").astype(float)
print(f"Average movie length : {movies['minutes'].mean():.0f} minutes")
print(f"Shortest movie       : {movies['minutes'].min():.0f} minutes")
print(f"Longest movie        : {movies['minutes'].max():.0f} minutes")

# Longest and shortest titles
longest = movies.loc[movies["minutes"].idxmax(), ["title", "minutes", "country"]]
shortest = movies.loc[movies["minutes"].idxmin(), ["title", "minutes", "country"]]
print(f"\nLongest : {longest['title']} ({longest['minutes']:.0f} min)")
print(f"Shortest: {shortest['title']} ({shortest['minutes']:.0f} min)")

# ── 9. FILTERING — INDIAN CONTENT ONLY ───────────────────
print("\n" + "=" * 55)
print("INDIAN CONTENT ANALYSIS:")
india = df[df["primary_country"] == "India"].copy()
print(f"Total Indian titles : {len(india)}")
print(f"Movies              : {len(india[india['type']=='Movie'])}")
print(f"TV Shows            : {len(india[india['type']=='TV Show'])}")
print("\nTop genres in India:")
print(india["listed_in"].value_counts().head(5))

# ── 10. VISUALISE ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Netflix Content Analysis", fontsize=16, fontweight="bold")

# Chart 1 — Movies vs TV Shows
type_counts.plot(kind="bar", ax=axes[0,0], color=["#E50914", "steelblue"],
                 edgecolor="white")
axes[0,0].set_title("Movies vs TV Shows")
axes[0,0].set_ylabel("Count")
axes[0,0].set_xticklabels(["Movies", "TV Shows"], rotation=0)

# Chart 2 — Top 10 Countries
top_countries.plot(kind="barh", ax=axes[0,1], color="#E50914")
axes[0,1].set_title("Top 10 Content-Producing Countries")
axes[0,1].set_xlabel("Number of Titles")
axes[0,1].invert_yaxis()

# Chart 3 — Content added per year
yearly.plot(kind="bar", ax=axes[1,0], color="steelblue", edgecolor="white")
axes[1,0].set_title("Content Added to Netflix Per Year")
axes[1,0].set_ylabel("Titles Added")
axes[1,0].tick_params(axis="x", rotation=45)

# Chart 4 — Top 10 Directors
top_directors.plot(kind="barh", ax=axes[1,1], color="#E50914")
axes[1,1].set_title("Top 10 Most Featured Directors")
axes[1,1].set_xlabel("Number of Titles")
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.savefig("netflix_analysis.png", dpi=150)
plt.show()
print("\nChart saved as netflix_analysis.png ✓")
# ── 1. LOAD ──────────────────────────────────────────────
url = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/netflix_titles.csv"
df = pd.read_csv(url)

# ── 2. FIRST LOOK ────────────────────────────────────────
print("=" * 55)
print("SHAPE:", df.shape)
print("\nCOLUMNS:", df.columns.tolist())
print("\nFIRST 3 ROWS:")
print(df.head(3))
print("\nMISSING VALUES:")
missing = pd.DataFrame({
    "Count": df.isnull().sum(),
    "%": (df.isnull().sum() / len(df) * 100).round(1)
})
print(missing[missing["Count"] > 0])