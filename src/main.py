import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    # classifiacation_report,
    confusion_matrix,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")

def main():
    print("PREDIKSI ONLINE SHOPPERS PURCHASING INTENTION")
    print("Dataset: UCI Machine Learning Repository")

# Prepare path untuk directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "dataset", "online_shoppers_intention.csv")
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

print("\nLoading dataset dan checking dataset...")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di path: {data_path}")

df = pd.read_csv(data_path)
total_awal = len(df)
print(f"Dimensi Dataset Mentah: {total_awal} baris x {df.shape[1]} kolom")

# check for missing values
null_counts = df.isnull().sum().sum()
print(f"Total missing values: {null_counts}")

# cleaning
jml_duplikat = df.duplicated().sum()
print(f"Data Duplikat Ditemukan: {jml_duplikat} baris ({jml_duplikat/total_awal*100:.2f}%)")
print("Membersihkan baris duplikat untuk mencegah data leakage dan model bias...")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dimensi Dataset Setelah De-duplikasi: {len(df)} baris")

df['Revenue_Num'] = df['Revenue'].astype(int)

num_cols = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
print(f"Total Fitur Input: {len(num_cols)} Numerik + {len(cat_cols)} Categorical = {len(num_cols)+len(cat_cols)} Fitur")

# visualisasi
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(6.5, 5))
counts = df['Revenue'].value_counts()
colors = ['#0390fc', "#ff7700"]
labels = ['False', 'True']
bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor='black')
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 150, f"{yval:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')
ax.set_title("Distribusi Kelas Target: Keputusan Pembelian", fontsize=12, fontweight='bold', pad=15)
ax.set_ylabel("Jumlah Sesi Kunjungan", fontsize=10)
ax.set_ylim(0, 12500)
plt.tight_layout()
v1_path = os.path.join(output_dir, "01_distribusi_target_imbalance.png")
plt.savefig(v1_path, dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df, x='Revenue', y='PageValues', hue='Revenue', palette=['#0390fc', "#ff7700"], legend=False, ax=ax, showfliers=False)
ax.set_title("Distribusi Nilai Halaman vs Keputusan Pembelian", fontsize=12, fontweight='bold', pad=15)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Tidak Membeli', 'Membeli'])
ax.set_ylabel("Rata-rata Page Values", fontsize=10)
plt.tight_layout()
v2_path = os.path.join(output_dir, "02_pagevalues_vs_revenue.png")
plt.savefig(v2_path, dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8,6))
corr_cols = ['BounceRates', 'ExitRates', 'PageValues', 'ProductRelated_Duration', 'Administrative_Duration', 'Revenue_Num']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f", ax=ax, cbar=True)
ax.set_title("Heatmap korelasi fitur perilaku pengunjung vs Revenue", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
v3_path = os.path.join(output_dir, "03_heatmap_correlation.png")
plt.savefig(v3_path, dpi=300)
plt.close()

# preprocessing & modeling
print("Preprocessing data dan membangun model...")
df_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
X = pd.concat([df[num_cols], df_cat], axis=1)
y = df['Revenue_Num']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Total Data Latih: {X_train.shape[0]} baris | Data Uji: {X_test.shape[0]} baris")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

print("Training model Logistic Regression...")
model_lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
y_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

print("Training model Random Forest Classifier...")
model_rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]

print("\nEvaluasi Metrik & Analisis Hasil...")
results = [
    {
        "Model": "Baseline Logistic Regression",
        "Akurasi": accuracy_score(y_test, y_pred_lr),
        "Presisi": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1-Score": f1_score(y_test, y_pred_lr),
        "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
    },
    {
        "Model": "Ensemble Random Forest",
        "Akurasi": accuracy_score(y_test, y_pred_rf),
        "Presisi": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1-Score": f1_score(y_test, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    }
]
df_eval = pd.DataFrame(results)
print("\nTabel perbandingan performa model pada data uji:")
print(df_eval.to_string(index=False))

# lanjut visualisasi
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Prediksi Tidak Membeli', 'Prediksi Membeli'], yticklabels=['Aktual Tidak Membeli', 'Aktual Membeli'])
axes[0].set_title(f"Baseline: : Logistic Regression\nRecall: {recall_score(y_test, y_pred_lr):.3f} | Akurasi: {accuracy_score(y_test, y_pred_lr):.3f}", fontweight='bold')
axes[0].set_ylabel("Kelas Aktual")

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False,
            xticklabels=['Prediksi Tidak Membeli', 'Prediksi Membeli'], yticklabels=['Aktual Tidak Membeli', 'Aktual Membeli'])
axes[1].set_title(f"Ensemble: Random Forest\nRecall: {recall_score(y_test, y_pred_rf):.3f} | Akurasi: {accuracy_score(y_test, y_pred_rf):.3f}", fontweight='bold')
axes[1].set_ylabel("Kelas Aktual")

plt.suptitle("Perbandingan Confusion Matrix", fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
v4_path = os.path.join(output_dir, "04_confusion_matrix_perbandingan.png")
plt.savefig(v4_path, dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})", color='#0390fc', lw=2)
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})", color='#ff7700', lw=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.5)')
ax.set_title("ROC Curve: Perbandingan Kinerja Klasifikasi", fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.legend(loc='lower right')
plt.tight_layout()
v5_path = os.path.join(output_dir, "05_kurva_roc_auc.png")
plt.savefig(v5_path, dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8.5, 5.5))
imp = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
imp.plot(kind='barh', color="#04ff00", edgecolor='black', ax=ax)
ax.set_title("Top 10 Fitur Paling Berpengaruh Terhadap Pembelian", fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel("Tingkat Kepentingan Relatif", fontsize=10)
plt.tight_layout()
v6_path = os.path.join(output_dir, "06_feature_importance.png")
plt.savefig(v6_path, dpi=300)
plt.close()
print(f"\n\nSemua visualisasi tersimpan:\n{v1_path}\n{v2_path}\n{v3_path}\n{v4_path}\n{v5_path}\n{v6_path}")

print("\nSemua proses selesai.")

if __name__ == "__main__":
    main()