import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Muat Dataset
df = pd.read_csv("dataset_banjir_semarang.csv")

feature_cols = ['elevation', 'slope', 'twi', 'ndvi', 'precipitation']
X = df[feature_cols]
y = df['flood_label']

# 2. Pembagian Data Latih dan Uji (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Inisialisasi Model (XGBoost & Random Forest)
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric='logloss',
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# 4. Pelatihan Model
print("Sedang melatih model...")
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# 5. Evaluasi Metrik
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_test, xgb_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"\nSkor AUC XGBoost       : {xgb_auc:.4f}")
print(f"Skor AUC Random Forest : {rf_auc:.4f}")

print("\n--- Evaluasi Klasifikasi XGBoost ---")
print(classification_report(y_test, xgb_model.predict(X_test)))

# 6. Plot Kurva ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

plt.figure(figsize=(7, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.2f})', color='blue', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.2f})', color='orange', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC Kerentanan Banjir Semarang')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_semarang.png', dpi=300)
print("\nKurva evaluasi disimpan ke 'roc_curve_semarang.png'")

# 7. Tampilkan Kepentingan Fitur (Feature Importance)
importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTingkat Kepentingan Fitur (XGBoost):")
print(importance)