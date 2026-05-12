import pandas as pd

# ── 1. LOAD ──────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ── 2. MISSING VALUES AUDIT ───────────────────────────────
# First thing after loading — find ALL missing data
print("=" * 50)
print("MISSING VALUES PER COLUMN:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
})
print(missing_report[missing_report["Missing Count"] > 0])

# ── 3. SURVIVAL OVERVIEW ─────────────────────────────────
print("\n" + "=" * 50)
print("HOW MANY SURVIVED?")
print(df["Survived"].value_counts())
print("\nIn plain English:")
survived = df["Survived"].sum()
total = len(df)
print(f"  {survived} out of {total} survived ({survived/total*100:.1f}%)")

# ── 4. SURVIVAL BY GENDER ────────────────────────────────
print("\n" + "=" * 50)
print("SURVIVAL RATE BY GENDER:")
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print(gender_survival.round(1).to_string())
print("\n(% who survived within each gender)")

# ── 5. SURVIVAL BY CLASS ─────────────────────────────────
print("\n" + "=" * 50)
print("SURVIVAL RATE BY PASSENGER CLASS:")
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print(class_survival.round(1).to_string())
print("\n(1 = First class, 2 = Second, 3 = Third)")

# ── 6. AGE ANALYSIS ──────────────────────────────────────
print("\n" + "=" * 50)
print("AGE ANALYSIS:")
print(f"  Average age : {df['Age'].mean():.1f} years")
print(f"  Youngest    : {df['Age'].min():.1f} years")
print(f"  Oldest      : {df['Age'].max():.1f} years")
print(f"  Missing ages: {df['Age'].isnull().sum()} passengers")

# Who was the oldest and youngest?
oldest = df.loc[df["Age"].idxmax(), ["Name", "Age", "Pclass", "Survived"]]
youngest = df.loc[df["Age"].idxmin(), ["Name", "Age", "Pclass", "Survived"]]
print(f"\n  Oldest  — {oldest['Name']}, Age {oldest['Age']}, Class {oldest['Pclass']}, Survived: {oldest['Survived']}")
print(f"  Youngest— {youngest['Name']}, Age {youngest['Age']}, Class {youngest['Pclass']}, Survived: {youngest['Survived']}")

# ── 7. FARE ANALYSIS ─────────────────────────────────────
print("\n" + "=" * 50)
print("FARE ANALYSIS BY CLASS:")
fare_by_class = df.groupby("Pclass")["Fare"].mean()
print(fare_by_class.round(2).to_string())

# ── 8. FAMILY ON BOARD ───────────────────────────────────
print("\n" + "=" * 50)
print("TRAVELLING ALONE VS WITH FAMILY:")
df["FamilySize"] = df["SibSp"] + df["Parch"]
df["Alone"] = df["FamilySize"] == 0
print(df["Alone"].value_counts().rename({True: "Alone", False: "With family"}))
alone_survival = df.groupby("Alone")["Survived"].mean() * 100
print("\nSurvival rate:")
print(alone_survival.rename({True: "Alone", False: "With family"}).round(1).to_string())

import matplotlib.pyplot as plt

# ── 9. VISUALISE KEY FINDINGS ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Titanic Survival Analysis", fontsize=14, fontweight="bold")

# Chart 1 — Survival by Gender
gender_survival.plot(kind="bar", ax=axes[0], color=["coral", "steelblue"],
                     edgecolor="white")
axes[0].set_title("Survival Rate by Gender")
axes[0].set_ylabel("Survival Rate (%)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(["Female", "Male"], rotation=0)
axes[0].set_ylim(0, 100)

# Chart 2 — Survival by Class
class_survival.plot(kind="bar", ax=axes[1], color=["gold", "silver", "#cd7f32"],
                    edgecolor="white")
axes[1].set_title("Survival Rate by Class")
axes[1].set_ylabel("Survival Rate (%)")
axes[1].set_xlabel("Passenger Class")
axes[1].set_xticklabels(["1st", "2nd", "3rd"], rotation=0)
axes[1].set_ylim(0, 100)

# Chart 3 — Alone vs Family
alone_survival.rename({True: "Alone", False: "With Family"}).plot(
    kind="bar", ax=axes[2], color=["tomato", "mediumseagreen"], edgecolor="white")
axes[2].set_title("Survival Rate: Alone vs Family")
axes[2].set_ylabel("Survival Rate (%)")
axes[2].set_xlabel("")
axes[2].set_xticklabels(["Alone", "With Family"], rotation=0)
axes[2].set_ylim(0, 100)

plt.tight_layout()
plt.savefig("titanic_analysis.png", dpi=150)
plt.show()
print("\nChart saved as titanic_analysis.png ✓")