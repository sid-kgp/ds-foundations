import pandas as pd
import numpy as np

# ── 1. CREATE A DELIBERATELY MESSY DATASET ───────────────
# This is a fake employee dataset with real-world data problems
data = {
    "name":       ["Alice", "BOB", "charlie", "  Diana  ", "Eve", "Frank", "Grace", "Henry", "Alice", "Bob"],
    "age":        [25, 30, None, 28, 999, 35, 22, None, 25, 30],
    "salary":     [50000, 60000, 55000, None, 70000, 80000, 45000, 62000, 50000, 60000],
    "department": ["HR", "hr", "HR", "Engineering", "engineering", "Finance", "Finance", None, "HR", "hr"],
    "join_date":  ["2020-01-15", "2019-03-22", "2021-07-01", "2020/05/10",
                   "2018-11-30", "2022-02-14", "2023-01-01", "2021-08-15",
                   "2020-01-15", "2019-03-22"],
    "email":      ["alice@co.com", "bob@co.com", "charlie@co.com", "diana@co.com",
                   "eve@co.com", "frank@co.com", "grace@co.com", "henry@co.com",
                   "alice@co.com", "bob@co.com"]
}

df = pd.DataFrame(data)

print("=" * 55)
print("RAW MESSY DATA:")
print(df.to_string())

print("\n" + "=" * 55)
print("STARTING DATA CLEANING...")

# ── 2. FIX NAMES ─────────────────────────────────────────
# Strip whitespace + title case (First Letter Capital)
df["name"] = df["name"].str.strip().str.title()
print("\n✓ Names fixed:", df["name"].tolist())

# ── 3. FIX DEPARTMENT CASING ─────────────────────────────
df["department"] = df["department"].str.strip().str.title()
print("✓ Departments fixed:", df["department"].tolist())

# ── 4. HANDLE OUTLIER IN AGE ─────────────────────────────
# 999 is clearly wrong — replace with NaN first
df["age"] = df["age"].replace(999, np.nan)
print(f"\n✓ Age outlier (999) replaced with NaN")
print(f"  Missing ages now: {df['age'].isnull().sum()}")

# ── 5. FILL MISSING AGE WITH MEDIAN ──────────────────────
# Why median not mean? Median is resistant to outliers
median_age = df["age"].median()
df["age"] = df["age"].fillna(median_age)
print(f"✓ Missing ages filled with median: {median_age}")

# ── 6. FILL MISSING SALARY WITH MEDIAN ───────────────────
median_salary = df["salary"].median()
df["salary"] = df["salary"].fillna(median_salary)
print(f"✓ Missing salary filled with median: {median_salary}")

# ── 7. FILL MISSING DEPARTMENT ───────────────────────────
# Can't guess department — use "Unknown"
df["department"] = df["department"].fillna("Unknown")
print(f"✓ Missing department filled with 'Unknown'")

# ── 8. FIX DATE FORMAT ───────────────────────────────────
# pd.to_datetime handles mixed formats automatically
df["join_date"] = pd.to_datetime(df["join_date"],
                                  format="mixed",
                                  errors="coerce")
print(f"✓ Dates standardised to datetime format")

# ── 9. REMOVE DUPLICATES ─────────────────────────────────
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\n✓ Duplicates removed: {before - after} rows dropped")
print(f"  Rows before: {before} → after: {after}")

# ── 10. FINAL RESULT ─────────────────────────────────────
print("\n" + "=" * 55)
print("CLEANED DATA:")
print(df.to_string())

print("\n" + "=" * 55)
print("FINAL MISSING VALUES CHECK:")
print(df.isnull().sum())

# ── 11. SAVE CLEANED DATA ────────────────────────────────
df.to_csv("employees_clean.csv", index=False)
print("\n✓ Cleaned data saved as employees_clean.csv")