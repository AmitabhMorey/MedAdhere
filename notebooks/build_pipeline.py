"""
Complete Pipeline Runner and Notebook Generator
Medication Adherence Analysis (MedAdhere AI)
=================================================
Executes full academic ML pipeline:
- Data quality & cleaning
- Feature engineering
- Model training & evaluation (Dummy, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Hyperparameter tuning via GridSearchCV
- Exporting pipeline & metadata
- Generating figures and reproducible presentation notebook
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Set styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'medication_adherence.csv')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'medication_adherence_processed.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("--- Step 1: Loading Raw Data ---")
df_raw = pd.read_csv(DATA_RAW)
print(f"Loaded shape: {df_raw.shape}")

print("--- Step 2: Data Cleaning & Preprocessing ---")
# 1. Remove duplicates
dups_count = df_raw.duplicated().sum()
df_clean = df_raw.drop_duplicates().copy()
print(f"Removed {dups_count} duplicate records. Clean shape: {df_clean.shape}")

# 2. Inspect missing values
missing = df_clean.isnull().sum()
print("Missing values per column:\n", missing[missing > 0])

# 3. Feature Engineering
# Create clinically grounded features
# A. Age Group
df_clean['Age_Group'] = pd.cut(
    df_clean['Age'],
    bins=[17, 35, 50, 65, 100],
    labels=['Young Adult (18-35)', 'Middle-Aged (36-50)', 'Older Adult (51-65)', 'Senior (65+)']
)

# B. Medication Burden Index = Frequency * Number_of_Medications
df_clean['Medication_Burden_Index'] = df_clean['Medication_Frequency'] * df_clean['Number_of_Medications']

# C. High Missed Dose Flag: >= 3 missed doses in past month
df_clean['High_Missed_Doses_Flag'] = (df_clean['Previous_Missed_Doses'] >= 3).astype(int)

# D. Polypharmacy Flag (>= 5 active medications)
df_clean['Polypharmacy_Flag'] = (df_clean['Number_of_Medications'] >= 5).astype(int)

# Save processed dataset
df_clean.to_csv(DATA_PROCESSED, index=False)
print(f"Saved processed dataset to: {DATA_PROCESSED}")

print("--- Step 3: Generating Visualizations for Reports ---")
# 1. Target Distribution
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(x='Medication_Adherence', data=df_clean, palette=['#e74c3c', '#2ecc71'])
plt.title('Target Variable Distribution: Medication Adherence', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Adherence Status (0 = Non-Adherent, 1 = Adherent)', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df_clean)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'target_distribution.png'), dpi=300)
plt.close()

# 2. Age Distribution
plt.figure(figsize=(8, 4.5))
sns.histplot(df_clean['Age'], kde=True, bins=25, color='#3498db')
plt.title('Age Distribution of Patient Cohort', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Age (Years)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'age_distribution.png'), dpi=300)
plt.close()

# 3. Adherence by Age Group
plt.figure(figsize=(9, 4.5))
sns.barplot(x='Age_Group', y='Medication_Adherence', data=df_clean, ci=None, palette='Blues_d')
plt.title('Adherence Rate by Patient Age Category', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Age Group', fontsize=11)
plt.ylabel('Mean Adherence Rate', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'adherence_by_age_group.png'), dpi=300)
plt.close()

# 4. Adherence by Medication Frequency
plt.figure(figsize=(8, 4.5))
sns.barplot(x='Medication_Frequency', y='Medication_Adherence', data=df_clean, ci=None, palette='Reds_r')
plt.title('Medication Adherence Rate by Daily Dosing Frequency', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Daily Medication Frequency (Doses/Day)', fontsize=11)
plt.ylabel('Mean Adherence Rate', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'adherence_by_medication_frequency.png'), dpi=300)
plt.close()

# 5. Adherence vs Reminder Usage
plt.figure(figsize=(7, 4.5))
sns.barplot(x='Medication_Reminder_Usage', y='Medication_Adherence', data=df_clean, ci=None, palette=['#27ae60', '#e67e22'])
plt.title('Impact of Digital/Manual Medication Reminders on Adherence', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Uses Medication Reminder System', fontsize=11)
plt.ylabel('Mean Adherence Rate', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'adherence_vs_reminder_usage.png'), dpi=300)
plt.close()

# 6. Adherence vs Previous Missed Doses
plt.figure(figsize=(9, 4.5))
sns.boxplot(x='Medication_Adherence', y='Previous_Missed_Doses', data=df_clean, palette=['#e74c3c', '#2ecc71'])
plt.title('Previous Missed Doses (Past 30 Days) by Adherence Status', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Medication Adherence (0 = Non-Adherent, 1 = Adherent)', fontsize=11)
plt.ylabel('Previous Missed Doses (Count)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'adherence_vs_previous_missed_doses.png'), dpi=300)
plt.close()

# 7. Adherence vs Side Effects
plt.figure(figsize=(8, 4.5))
sns.barplot(x='Side_Effects', y='Medication_Adherence', data=df_clean, order=['None', 'Mild', 'Moderate', 'Severe'], ci=None, palette='Purples_r')
plt.title('Medication Adherence Across Reported Side-Effect Severity', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Reported Side Effects Severity', fontsize=11)
plt.ylabel('Mean Adherence Rate', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'adherence_vs_side_effects.png'), dpi=300)
plt.close()

# 8. Adherence vs Medication Cost Burden
plt.figure(figsize=(8, 4.5))
sns.barplot(x='Medication_Cost_Burden', y='Medication_Adherence', data=df_clean, order=['Low', 'Moderate', 'High'], ci=None, palette='Oranges_r')
plt.title('Medication Adherence by Financial Cost Burden', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Medication Cost Burden', fontsize=11)
plt.ylabel('Mean Adherence Rate', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'adherence_vs_cost_burden.png'), dpi=300)
plt.close()

# 9. Correlation Matrix
num_cols_corr = ['Age', 'Medication_Frequency', 'Number_of_Medications', 'Treatment_Duration_Months',
                 'Previous_Missed_Doses', 'Forgetfulness_Score', 'Routine_Consistency',
                 'Treatment_Satisfaction', 'Distance_to_Pharmacy_Km', 'Doctor_Communication',
                 'Medication_Burden_Index', 'Medication_Adherence']
plt.figure(figsize=(10, 8))
corr_matrix = df_clean[num_cols_corr].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.6, vmax=0.6, linewidths=0.5)
plt.title('Pearson Correlation Heatmap of Key Numerical Features', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'correlation_matrix.png'), dpi=300)
plt.close()

print("Figures successfully generated!")

print("--- Step 4: Train/Test Split & Preprocessing Pipeline ---")
# Drop Patient_ID (non-predictive identifier)
feature_df = df_clean.drop(columns=['Patient_ID', 'Medication_Adherence'])
target = df_clean['Medication_Adherence']

X_train, X_test, y_train, y_test = train_test_split(
    feature_df, target, test_size=0.20, stratify=target, random_state=42
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Identify column types
num_features = [
    'Age', 'Medication_Frequency', 'Number_of_Medications', 'Treatment_Duration_Months',
    'Previous_Missed_Doses', 'Forgetfulness_Score', 'Routine_Consistency',
    'Treatment_Satisfaction', 'Distance_to_Pharmacy_Km', 'Doctor_Communication',
    'Medication_Burden_Index', 'High_Missed_Doses_Flag', 'Polypharmacy_Flag'
]

cat_features = [
    'Gender', 'Education_Level', 'Employment_Status', 'Medication_Type',
    'Dosage_Complexity', 'Medication_Reminder_Usage', 'Chronic_Condition',
    'Side_Effects', 'Perceived_Effectiveness', 'Medication_Cost_Burden',
    'Access_to_Pharmacy', 'Insurance_Coverage', 'Family_Support',
    'Healthcare_Followup', 'Age_Group'
]

# Build Preprocessing Transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

print("--- Step 5: Model Training & Comparison ---")
models = {
    'Dummy Baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
}

results = []
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5
    
    # Specific recall for the non-adherent class (class 0)
    rec_non_adherent = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall (Adherent)': round(rec, 4),
        'Recall (Non-Adherent)': round(rec_non_adherent, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4)
    })

results_df = pd.DataFrame(results)
print("Model Comparison Results:\n", results_df.to_string(index=False))
results_df.to_csv(os.path.join(REPORTS_DIR, 'model_results.csv'), index=False)

# Plot Model Comparison
plt.figure(figsize=(10, 5))
metrics_to_plot = ['Accuracy', 'F1-Score', 'ROC-AUC', 'Recall (Non-Adherent)']
df_plot = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='Model', data=df_plot, palette='Set2')
plt.title('Cross-Model Performance Comparison on Test Set', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('Score (0.0 - 1.0)', fontsize=11)
plt.ylim(0, 1.05)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'model_comparison_metrics.png'), dpi=300)
plt.close()

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, pipe in trained_pipelines.items():
    if name == 'Dummy Baseline':
        continue
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_score:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=12)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_curves_comparison.png'), dpi=300)
plt.close()

# Plot Precision-Recall Curves
plt.figure(figsize=(8, 6))
for name, pipe in trained_pipelines.items():
    if name == 'Dummy Baseline':
        continue
    y_proba = pipe.predict_proba(X_test)[:, 1]
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_vals, prec_vals)
    plt.plot(rec_vals, prec_vals, lw=2, label=f'{name} (PR-AUC = {pr_auc:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Adherent Class)', fontsize=11)
plt.ylabel('Precision (Adherent Class)', fontsize=11)
plt.title('Precision-Recall Curves Across Machine Learning Models', fontsize=13, fontweight='bold', pad=12)
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'pr_curves_comparison.png'), dpi=300)
plt.close()

print("--- Step 6: Hyperparameter Tuning (Random Forest) ---")
# Tuning Random Forest via 5-Fold Stratified CV
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [6, 10, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [2, 4],
    'classifier__class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=rf_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
print("Best Hyperparameters:", grid_search.best_params_)

# Final Evaluation
y_pred_final = best_pipeline.predict(X_test)
y_proba_final = best_pipeline.predict_proba(X_test)[:, 1]

acc_final = accuracy_score(y_test, y_pred_final)
prec_final = precision_score(y_test, y_pred_final)
rec_final = recall_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)
roc_auc_final = roc_auc_score(y_test, y_proba_final)

print("\nFinal Tuned Random Forest Test Performance:")
print(f"Accuracy:  {acc_final:.4f}")
print(f"Precision: {prec_final:.4f}")
print(f"Recall:    {rec_final:.4f}")
print(f"F1-Score:  {f1_final:.4f}")
print(f"ROC-AUC:   {roc_auc_final:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_final, target_names=['Non-Adherent (0)', 'Adherent (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Adherent (0)', 'Adherent (1)'],
            yticklabels=['Non-Adherent (0)', 'Adherent (1)'],
            annot_kws={"size": 14, "fontweight": "bold"})
plt.title('Final Tuned Random Forest: Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual True Label', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix_best_model.png'), dpi=300)
plt.close()

# Feature Importance
fitted_preprocessor = best_pipeline.named_steps['preprocessor']
fitted_cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_encoded_feature_names = list(fitted_cat_encoder.get_feature_names_out(cat_features))
all_feature_names = num_features + cat_encoded_feature_names

rf_model = best_pipeline.named_steps['classifier']
importances = rf_model.feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 7))
top_15_feats = feat_imp_df.head(15)
sns.barplot(x='Importance', y='Feature', data=top_15_feats, palette='crest_r')
plt.title('Top 15 Factors Associated With Medication Adherence Prediction', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Random Forest Relative Gini Importance', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=300)
plt.close()

print("--- Step 7: Fairness & Subgroup Analysis ---")
# Evaluate performance across gender and age groups
test_eval_df = X_test.copy()
test_eval_df['True_Adherence'] = y_test
test_eval_df['Predicted_Adherence'] = y_pred_final

subgroup_results = []
for grp_col in ['Gender', 'Age_Group']:
    for grp_val, sub_df in test_eval_df.groupby(grp_col):
        if len(sub_df) > 10:
            sub_acc = accuracy_score(sub_df['True_Adherence'], sub_df['Predicted_Adherence'])
            sub_prec = precision_score(sub_df['True_Adherence'], sub_df['Predicted_Adherence'], zero_division=0)
            sub_rec = recall_score(sub_df['True_Adherence'], sub_df['Predicted_Adherence'], zero_division=0)
            sub_f1 = f1_score(sub_df['True_Adherence'], sub_df['Predicted_Adherence'], zero_division=0)
            subgroup_results.append({
                'Subgroup_Category': grp_col,
                'Subgroup': str(grp_val),
                'Sample_Size': len(sub_df),
                'Accuracy': round(sub_acc, 4),
                'Precision': round(sub_prec, 4),
                'Recall': round(sub_rec, 4),
                'F1-Score': round(sub_f1, 4)
            })

subgroup_df = pd.DataFrame(subgroup_results)
print("Subgroup Fairness Analysis:\n", subgroup_df.to_string(index=False))

plt.figure(figsize=(9, 4.5))
sns.barplot(x='Subgroup', y='F1-Score', hue='Subgroup_Category', data=subgroup_df, dodge=False, palette='Set1')
plt.title('Subgroup F1-Score Fairness Analysis (Gender & Age Cohorts)', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('F1 Score', fontsize=11)
plt.ylim(0.5, 1.0)
plt.xticks(rotation=20)
plt.legend(title='Category', loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fairness_subgroup_analysis.png'), dpi=300)
plt.close()

print("--- Step 8: Error Analysis ---")
error_df = X_test.copy()
error_df['Actual'] = y_test
error_df['Predicted'] = y_pred_final
error_df['Probability_Adherent'] = y_proba_final
error_df['Error_Type'] = np.where(
    (error_df['Actual'] == 1) & (error_df['Predicted'] == 0), 'False Negative (Adherent predicted Non-Adherent)',
    np.where(
        (error_df['Actual'] == 0) & (error_df['Predicted'] == 1), 'False Positive (Non-Adherent predicted Adherent)',
        'Correct'
    )
)

fn_count = (error_df['Error_Type'] == 'False Negative (Adherent predicted Non-Adherent)').sum()
fp_count = (error_df['Error_Type'] == 'False Positive (Non-Adherent predicted Adherent)').sum()
correct_count = (error_df['Error_Type'] == 'Correct').sum()

print(f"Error summary: Correct={correct_count}, False Negatives={fn_count}, False Positives={fp_count}")

print("--- Step 9: Exporting Final Models & Metadata ---")
# 1. Full pipeline (Preprocessor + Model)
model_path = os.path.join(MODELS_DIR, 'medication_adherence_model.pkl')
joblib.dump(best_pipeline, model_path)
print(f"Saved full model pipeline to: {model_path}")

# 2. Standalone preprocessor
prep_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')
joblib.dump(fitted_preprocessor, prep_path)
print(f"Saved preprocessor to: {prep_path}")

# 3. Model Metadata JSON
metadata = {
    "project_title": "Medication Adherence Analysis: Predicting Patient Medication Compliance Using Machine Learning",
    "application_name": "MedAdhere AI",
    "version": "1.0.0",
    "model_name": "Random Forest Classifier (Tuned)",
    "target": "Medication_Adherence",
    "classification_type": "binary",
    "classes": {"0": "Non-Adherent", "1": "Adherent"},
    "random_state": 42,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": {
        "numerical": num_features,
        "categorical": cat_features
    },
    "metrics": {
        "accuracy": round(acc_final, 4),
        "precision": round(prec_final, 4),
        "recall": round(rec_final, 4),
        "f1_score": round(f1_final, 4),
        "roc_auc": round(roc_auc_final, 4)
    },
    "risk_thresholds": {
        "low_risk": "Adherence Probability >= 0.70",
        "moderate_risk": "Adherence Probability 0.40 - 0.69",
        "high_risk": "Adherence Probability < 0.40"
    },
    "top_features": feat_imp_df.head(10)['Feature'].tolist(),
    "academic_disclaimer": "This system is an educational and analytical machine-learning prototype designed for research and academic demonstration. It is NOT a clinical diagnostic tool and must not be used to alter medication regimens or guide clinical treatment decisions."
}

metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"Saved model metadata to: {metadata_path}")

print("=== PIPELINE EXECUTION COMPLETE ===")
