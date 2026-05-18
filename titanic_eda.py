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

#Section 3
#Fill missing ages with the median. Drop rows where Embarked is missing. Drop the Cabin column (too many missing values — 77%)   

#Section 3
# ── 1. Fill missing ages with the median. Drop rows where Embarked is missing. Drop the Cabin column (too many missing values — 77%) ──────────────────────
# ── 2. STORE SHAPE BEFORE CLEANING ──────────────────────
before_shape = df.shape

# ── 3. CLEANING ──────────────────────────────────────────

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Drop rows where Embarked is missing
df.dropna(subset=['Embarked'], inplace=True)

# Drop Cabin column completely
df.drop(columns=['Cabin'], inplace=True)

# ── 4. STORE SHAPE AFTER CLEANING ───────────────────────
after_shape = df.shape

# ── 5. PRINT RESULTS ────────────────────────────────────
print("Dataset Shape Comparison")
print(f"Before Cleaning: {before_shape}")
print(f"After Cleaning:  {after_shape}")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#Section 4
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 10))
plt.suptitle('Titanic Survival Analysis', fontsize=16, y=1.02)

# Subplot 1: Survival Count (Bar Chart)
plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df, palette='viridis')
plt.title('Survival Count (0=No, 1=Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])

# Subplot 2: Survival by Gender (Bar Chart)
plt.subplot(2, 2, 2)
survival_by_gender = df.groupby('Sex')['Survived'].mean().reset_index()
sns.barplot(x='Sex', y='Survived', data=survival_by_gender, palette='plasma')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.ylim(0, 1) # Set y-axis limit for proportion

# Subplot 3: Survival by Class (Bar Chart)
plt.subplot(2, 2, 3)
survival_by_class = df.groupby('Pclass')['Survived'].mean().reset_index()
sns.barplot(x='Pclass', y='Survived', data=survival_by_class, palette='coolwarm')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.ylim(0, 1) # Set y-axis limit for proportion

# Subplot 4: Age Distribution of Survivors vs Non-Survivors (Histogram)
plt.subplot(2, 2, 4)
sns.histplot(df[df['Survived'] == 1]['Age'].dropna(), color='skyblue', label='Survived', kde=True)
sns.histplot(df[df['Survived'] == 0]['Age'].dropna(), color='orange', label='Non-Survived', kde=True)
plt.title('Age Distribution: Survivors vs Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
plt.savefig("titanic_eda_charts.png")
plt.show()

# ── SECTION 5: KEY FINDINGS ──────────────────────────────
print("\n" + "=" * 55)
print("KEY FINDINGS SUMMARY")
print("=" * 55)
print(f"""
- Overall survival rate was only 38.4% — less than 4 in 10 survived

- Gender was the strongest predictor: 74% of women survived
  vs only 19% of men — "women and children first" was real

- Wealth determined survival: 1st class (63%) had 2.6x better
  odds than 3rd class (24%)

- Survivors were slightly younger (avg 28.3) vs non-survivors
  (avg 30.6) — children were prioritised in lifeboats

- 77% of Cabin data was missing — likely because most 3rd class
  passengers had no assigned cabin, linking missingness to class
""")