import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)

sns.set_theme(style="whitegrid")

# ── 1. LOAD & PREPARE ────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.dropna(subset=["Embarked"])
df = df.drop(columns=["Cabin"])
df["Sex_encoded"] = df["Sex"].map({"female": 1, "male": 0})

features = ["Pclass", "Sex_encoded", "Age", "SibSp", "Parch", "Fare"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 2. TRAIN ─────────────────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of surviving

# ── 3. ALL METRICS ───────────────────────────────────────
print("=" * 55)
print("CLASSIFICATION METRICS:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"  Precision : {precision_score(y_test, y_pred)*100:.1f}%")
print(f"  Recall    : {recall_score(y_test, y_pred)*100:.1f}%")
print(f"  F1 Score  : {f1_score(y_test, y_pred):.3f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.3f}")

# ── 4. CONFUSION MATRIX ──────────────────────────────────
print("\n" + "=" * 55)
print("CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(f"\n  True Negatives  (predicted died,     actually died)    : {cm[0][0]}")
print(f"  False Positives (predicted survived, actually died)    : {cm[0][1]}")
print(f"  False Negatives (predicted died,     actually survived): {cm[1][0]}")
print(f"  True Positives  (predicted survived, actually survived): {cm[1][1]}")

# ── 5. FULL REPORT ───────────────────────────────────────
print("\n" + "=" * 55)
print("FULL CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred,
      target_names=["Died", "Survived"]))

# ── 6. THRESHOLD EXPERIMENT ──────────────────────────────
# Default threshold is 0.5 — predict survived if prob > 50%
# What if we lower it to 0.3? More people predicted survived
# Recall goes up, precision goes down
print("=" * 55)
print("THRESHOLD EXPERIMENT:")
print("(How changing the decision threshold affects metrics)\n")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob >= threshold).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"  Threshold {threshold:.1f} → Precision: {p:.2f}  Recall: {r:.2f}  F1: {f:.2f}")

# ── 7. VISUALISE ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Classification Metrics — Titanic", fontweight="bold")

# Chart 1 — Confusion Matrix heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"],
            yticklabels=["Died", "Survived"],
            ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# Chart 2 — ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color="steelblue", linewidth=2,
             label=f"ROC Curve (AUC = {auc:.3f})")
axes[1].plot([0, 1], [0, 1], "r--", linewidth=1, label="Random guess")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("ml-basics/classification_metrics.png", dpi=150)
plt.show()
print("\nChart saved ✓")