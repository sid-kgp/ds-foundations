import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 0. SETUP ─────────────────────────────────────────────
# Seaborn styling makes matplotlib charts look professional
sns.set_theme(style="whitegrid", palette="muted")

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Create a figure with all 5 charts in a grid
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Titanic Dataset — 5 Chart Types", fontsize=16, fontweight="bold", y=1.01)

# ── 1. BAR CHART — survival count by class ───────────────
ax1 = fig.add_subplot(2, 3, 1)
survival_by_class = df.groupby("Pclass")["Survived"].sum()
bars = ax1.bar(["1st Class", "2nd Class", "3rd Class"],
               survival_by_class.values,
               color=["gold", "silver", "#cd7f32"])
ax1.set_title("Bar Chart\nSurvivors by Class")
ax1.set_ylabel("Number of Survivors")
# Add value labels on top of each bar
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             str(int(bar.get_height())),
             ha="center", fontweight="bold")

# ── 2. LINE CHART — not enough time data in Titanic ──────
# Let's simulate monthly ticket purchases
ax2 = fig.add_subplot(2, 3, 2)
# Group by embarkation port and class — use as a trend proxy
fare_by_class = df.groupby("Pclass")["Fare"].mean()
ax2.plot([1, 2, 3], fare_by_class.values,
         marker="o", linewidth=2, markersize=8, color="steelblue")
ax2.set_title("Line Chart\nAvg Fare by Class")
ax2.set_ylabel("Average Fare (£)")
ax2.set_xlabel("Passenger Class")
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(["1st", "2nd", "3rd"])
for x, y in zip([1, 2, 3], fare_by_class.values):
    ax2.annotate(f"£{y:.0f}", (x, y),
                 textcoords="offset points", xytext=(0, 10), ha="center")

# ── 3. SCATTER PLOT — age vs fare ────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
survived = df[df["Survived"] == 1]
died = df[df["Survived"] == 0]
ax3.scatter(died["Age"], died["Fare"],
            alpha=0.4, color="tomato", label="Died", s=20)
ax3.scatter(survived["Age"], survived["Fare"],
            alpha=0.6, color="steelblue", label="Survived", s=20)
ax3.set_title("Scatter Plot\nAge vs Fare (coloured by survival)")
ax3.set_xlabel("Age")
ax3.set_ylabel("Fare (£)")
ax3.legend()
ax3.set_ylim(0, 300)

# ── 4. HISTOGRAM — age distribution ─────────────────────
ax4 = fig.add_subplot(2, 3, 4)
ax4.hist(df["Age"].dropna(), bins=30,
         color="steelblue", edgecolor="white", alpha=0.8)
ax4.axvline(df["Age"].mean(), color="red",
            linestyle="--", linewidth=1.5, label=f"Mean: {df['Age'].mean():.1f}")
ax4.axvline(df["Age"].median(), color="orange",
            linestyle="--", linewidth=1.5, label=f"Median: {df['Age'].median():.1f}")
ax4.set_title("Histogram\nAge Distribution")
ax4.set_xlabel("Age")
ax4.set_ylabel("Number of Passengers")
ax4.legend()

# ── 5. HEATMAP — correlation matrix ──────────────────────
ax5 = fig.add_subplot(2, 3, 5)
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax5, square=True,
            cbar_kws={"shrink": 0.8})
ax5.set_title("Heatmap\nCorrelation Matrix")

# ── 6. BONUS — seaborn boxplot ───────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
sns.boxplot(data=df, x="Pclass", y="Age",
            palette=["gold", "silver", "#cd7f32"], ax=ax6)
ax6.set_title("Box Plot\nAge Distribution by Class")
ax6.set_xlabel("Passenger Class")
ax6.set_ylabel("Age")
ax6.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])

plt.tight_layout()
plt.savefig("titanic_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("All 6 charts saved as titanic_charts.png ✓")