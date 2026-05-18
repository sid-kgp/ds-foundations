import pandas as pd

# ── 1. LOAD ──────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ── 2. Dataset Size ────────────────────────────────────
print(f"Dataset shape: {df.shape}")
# Rows and columns separately
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# ── 3. COLUMN NAMES ─────────────────────────────────────
print("\nColumns:")
print(df.columns)

# ── 4. DATA TYPES ───────────────────────────────────────
print("\nData types:")
print(df.dtypes)

# ── 5. MISSING VALUES ───────────────────────────────────
missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
})

# Show only columns with missing values
missing = missing[missing["Missing Count"] > 0]

print(missing)

#Section-2
#What was the overall survival rate?
print(df['Survived'].unique())
survival_rate = df['Survived'].mean() * 100
print(f"Overall Survival Rate: {survival_rate:.2f}%")

#How did gender affect survival? (use groupby)
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
print("\nSurvival Rate by Gender:")
print(gender_survival)

#How did passenger class affect survival? (use groupby)
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
print('Survival Rate by Class:')
print(survival_by_class.round(2))

#Did age matter? Compare average age of survivors vs non-survivors
average_age_survived = df[df['Survived'] == 1]['Age'].mean()
average_age_not_survived = df[df['Survived'] == 0]['Age'].mean()
print(f"\nAverage Age of Survivors: {average_age_survived:.2f}")
print(f"Average Age of Non-Survivors: {average_age_not_survived:.2f}")