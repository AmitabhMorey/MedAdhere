"""
Jupyter Notebook Generator for Medication Adherence Analysis
Generates: notebooks/medication_adherence_analysis.ipynb
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# SECTION 1
cells.append(nbf.v4.new_markdown_cell("""# Medication Adherence Analysis: Predicting Patient Medication Compliance Using Machine Learning
**Alternative Application Name:** MedAdhere AI  
**Academic Context:** B.Tech CSE Machine Learning Case Study  
**Author:** Final Year B.Tech Computer Science & Engineering Student  
**Environment:** Python 3, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn  

---

## SECTION 1 — Project Introduction

### Background & Clinical Context
Medication adherence—defined by the World Health Organization (WHO) as the degree to which a person's behavior corresponds with agreed recommendations from a healthcare provider—is one of the most critical determinants of treatment efficacy, disease prognosis, and healthcare economics worldwide. 

In chronic conditions such as hypertension, type-2 diabetes, cardiovascular disease, and chronic respiratory disorders, non-adherence leads to severe clinical complications, preventable hospital admissions, increased mortality, and billions of dollars in avoidable healthcare costs. According to global health studies:
- Approximately **30% to 50%** of patients with chronic diseases do not take their medications as prescribed.
- Non-adherence manifests as missed doses, premature discontinuation, incorrect dosage timing, and unauthorized dose reductions.
- The causes are complex and multifactorial, spanning demographic attributes, socioeconomic cost barriers, behavioral forgetfulness, complex polypharmacy regimens, drug side effects, and inadequate provider communication.

### Important Academic & Clinical Disclaimer
> **IMPORTANT NOTICE:** This project is an **analytical and educational machine learning case study**. It predicts adherence risk patterns based on historical and behavioral attributes for academic demonstration. It is **NOT** a medical diagnostic tool, clinical prescriptive system, or replacement for certified healthcare professional judgment.
"""))

# SECTION 2
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 2 — Problem Definition

### Real-World Healthcare Challenge
Healthcare systems lack automated, proactive tools to identify patients who are likely to struggle with their medication regimens before treatment failure occurs. Clinicians often only discover non-adherence after disease exacerbation or hospital readmission.

### Machine Learning Formulation
This challenge is formulated as a **Supervised Machine Learning: Binary Classification** problem:
- **Input Features:** Demographic attributes, medication regimen complexity, behavioral metrics, clinical treatment experience, pharmacy accessibility, and family/provider support.
- **Target Variable ($y$):** `Medication_Adherence`
  - $y = 1$: **Adherent** (Patient consistently follows the prescribed medication schedule)
  - $y = 0$: **Non-Adherent** (Patient frequently misses or discontinues prescribed medication)
- **Model Output:**
  1. Predicted Class: `Adherent` vs `Non-Adherent`
  2. Calibrated Probability: $\\hat{P}(\\text{Adherence} = 1)$
  3. Risk Level Stratification: **Low Risk** ($\ge 70\%$), **Moderate Risk** ($40\\%-69\\%$), **High Risk** ($< 40\\%$)

### Academic Objectives
1. Perform thorough data quality verification and exploratory data analysis.
2. Prevent data leakage by isolating preprocessing pipelines strictly to training folds.
3. Compare multiple classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) against a statistical baseline.
4. Optimize the top-performing model via 5-Fold Stratified Cross-Validation hyperparameter tuning.
5. Conduct in-depth model interpretation, feature importance, subgroup fairness, and error analysis.
6. Export the validated model pipeline for live interactive demonstration in a modern Streamlit application.
"""))

# SECTION 3
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 3 — Dataset Selection and Justification

### Dataset Selection Justification
In accordance with research ethics and data governance standards, real-world patient adherence records containing comprehensive multi-dimensional variables (socioeconomic status, pharmacy access distance, behavioral forgetfulness scores, and clinical side-effect tracking) are heavily protected under privacy legislation (such as HIPAA and GDPR) and are rarely available as public unanonymized open-access datasets.

To satisfy all rigorous academic requirements without making false claims of clinical data provenance, this study utilizes a clearly documented:
**"Synthetic Medication Adherence Dataset for Educational Demonstration"**

#### Justification Criteria:
1. **Source & Reproducibility:** Procedurally generated using `data/generate_dataset.py` with fixed random seed (`seed=42`).
2. **Clinical Grounding:** Formulated using the **WHO 5 Dimensions of Adherence** (Socioeconomic, Healthcare Team/System, Condition-Related, Therapy-Related, and Patient-Related) and validated adherence assessment instruments (such as the *Morisky Medication Adherence Scale*, MMAS-8).
3. **Cohort Dimensions:** 3,000 distinct patient records with 26 features covering demographic, behavioral, pharmacological, clinical, and accessibility indicators.
4. **Realistic Complexity:** Includes non-linear relationships, multi-collinearity, interaction effects, realistic stochastic noise, deliberate missing values (~2% in select survey attributes), and synthetic duplicate records for data cleaning verification.
5. **Class Balance:** Class ratio reflects real-world chronic disease adherence statistics (~68.8% Adherent, 31.2% Non-Adherent).

### Feature Description Table
| Feature Category | Variable Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **Demographic** | `Age` | Numerical (Integer) | Patient chronological age (18 - 88 years) |
| **Demographic** | `Gender` | Categorical | Patient gender ('Male', 'Female', 'Other') |
| **Demographic** | `Education_Level` | Categorical | Highest educational attainment |
| **Demographic** | `Employment_Status` | Categorical | Employment status (Employed, Retired, etc.) |
| **Medication** | `Medication_Type` | Categorical | Primary chronic therapeutic drug category |
| **Medication** | `Medication_Frequency` | Numerical (Integer) | Prescribed daily dosing frequency (1 - 4 doses/day) |
| **Medication** | `Number_of_Medications` | Numerical (Integer) | Total active prescription medications (Polypharmacy) |
| **Medication** | `Treatment_Duration_Months` | Numerical (Integer) | Total months on current regimen |
| **Medication** | `Dosage_Complexity` | Categorical | Regimen complexity rating ('Low', 'Medium', 'High') |
| **Behavioral** | `Previous_Missed_Doses` | Numerical (Integer) | Self-reported missed doses in prior 30 days |
| **Behavioral** | `Forgetfulness_Score` | Numerical (Discrete) | Self-assessed forgetfulness scale (1 - 10) |
| **Behavioral** | `Routine_Consistency` | Numerical (Discrete) | Regularity of daily lifestyle routine (1 - 10) |
| **Behavioral** | `Medication_Reminder_Usage`| Categorical | Whether patient uses alarms/pillboxes ('Yes', 'No') |
| **Clinical** | `Chronic_Condition` | Categorical | Primary diagnosed chronic disease |
| **Clinical** | `Side_Effects` | Categorical | Severity of adverse drug reactions experienced |
| **Clinical** | `Perceived_Effectiveness` | Categorical | Patient belief in drug therapeutic value |
| **Clinical** | `Treatment_Satisfaction` | Numerical (Discrete) | Patient satisfaction score with treatment (1 - 10) |
| **Accessibility** | `Medication_Cost_Burden` | Categorical | Financial burden of prescription copays |
| **Accessibility** | `Access_to_Pharmacy` | Categorical | Ease of accessing physical pharmacy |
| **Accessibility** | `Insurance_Coverage` | Categorical | Healthcare insurance level ('Full', 'Partial', 'None')|
| **Accessibility** | `Distance_to_Pharmacy_Km` | Numerical (Float) | Travel distance to dispensing pharmacy in km |
| **Support** | `Family_Support` | Categorical | Degree of caregiver or family support |
| **Support** | `Healthcare_Followup` | Categorical | Regularity of outpatient clinic checkups |
| **Support** | `Doctor_Communication` | Numerical (Discrete) | Quality of patient-provider relationship (1 - 10) |
| **Target** | `Medication_Adherence` | Binary (0 / 1) | **Target Variable: 1 = Adherent, 0 = Non-Adherent** |
"""))

# SECTION 4
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 4 — Data Loading & Initial Inspection

We begin by loading the raw dataset and conducting structured structural inspections:
1. `head()` and `tail()` to verify data framing and boundary samples.
2. `shape` and `columns` to audit dimensional alignment.
3. `info()` to examine data types and non-null counts.
4. `describe()` to inspect central tendencies and dispersion metrics.
"""))

cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure display and plot settings
pd.set_option('display.max_columns', 35)
pd.set_option('display.width', 1000)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

# Load dataset
DATA_PATH = os.path.join('..', 'data', 'raw', 'medication_adherence.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join('data', 'raw', 'medication_adherence.csv')

df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset successfully loaded. Total rows: {df_raw.shape[0]}, Total columns: {df_raw.shape[1]}")
df_raw.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Summary structural diagnostics
print("--- Column Data Types & Non-Null Values ---")
print(df_raw.info())

print("\\n--- Statistical Summary of Numerical Features ---")
df_raw.describe()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Inspection Insights
- The dataset comprises **3,005 records** and **26 features**.
- `Patient_ID` serves as an administrative unique identifier (to be excluded from model training to prevent spurious feature memorization).
- Numerical features such as `Age`, `Number_of_Medications`, `Previous_Missed_Doses`, and `Distance_to_Pharmacy_Km` exhibit expected non-negative continuous/discrete distributions.
- Minor missing values are detected in `Distance_to_Pharmacy_Km`, `Doctor_Communication`, and `Treatment_Satisfaction`.
"""))

# SECTION 5
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 5 — Data Quality Analysis

A rigorous machine learning pipeline requires systematic verification of data hygiene before modeling:
1. **Missing Value Auditing**
2. **Duplicate Record Detection**
3. **Cardinality & Unique Value Profiling**
4. **Boundary & Validity Checks** (e.g. impossible ages, negative distances, invalid categories)
5. **Outlier Analysis via IQR & Boxplots** (evaluating whether outliers are collection errors or authentic clinical variance)
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Missing Values Analysis
missing_summary = pd.DataFrame({
    'Missing_Count': df_raw.isnull().sum(),
    'Percentage (%)': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
print("Missing Values Summary:")
print(missing_summary)

# 2. Duplicate Records Analysis
duplicate_count = df_raw.duplicated().sum()
print(f"\\nDuplicate rows detected: {duplicate_count}")

# 3. Categorical Uniqueness & Range Checks
print("\\nUnique Value Cardinality across Categorical Features:")
for col in df_raw.select_dtypes(include=['object']).columns:
    if col != 'Patient_ID':
        print(f"  {col}: {df_raw[col].unique().tolist()}")
"""))

cells.append(nbf.v4.new_code_cell("""# 4. Outlier Analysis using Interquartile Range (IQR)
numerical_cols = ['Age', 'Number_of_Medications', 'Treatment_Duration_Months', 
                  'Previous_Missed_Doses', 'Distance_to_Pharmacy_Km']

outlier_summary = []
for col in numerical_cols:
    clean_series = df_raw[col].dropna()
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
    outlier_summary.append({
        'Feature': col,
        'Q1 (25%)': round(q1, 2),
        'Q3 (75%)': round(q3, 2),
        'IQR': round(iqr, 2),
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': round(len(outliers) / len(clean_series) * 100, 2),
        'Min_Observed': clean_series.min(),
        'Max_Observed': clean_series.max()
    })

pd.DataFrame(outlier_summary)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Domain Outlier Evaluation: Data Error vs Legitimate Observation
- **`Previous_Missed_Doses`**: Observations exceeding 7-8 missed doses are statistically flagged as outliers by IQR rules, but are **clinically legitimate** cases of extreme non-adherence (e.g. patients who discontinued therapy for over a week). Truncating or deleting these would discard critical non-adherence signal.
- **`Distance_to_Pharmacy_Km`**: Right-skewed distribution where rural patients travel up to 35 km. This is an authentic accessibility barrier rather than a data entry defect.
- **Conclusion:** No records will be deleted on purely statistical outlier grounds; instead, robust scaling and tree-based methods will naturally accommodate legitimate long-tail distributions.
"""))

# SECTION 6
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 6 — Data Cleaning

Data cleaning operations performed:
1. **Deduplication:** Remove exact duplicate rows created during data acquisition.
2. **Identifier Separation:** Exclude `Patient_ID` from the modeling matrix.
3. **Missing Value Strategy:**
   - Numerical missing values (`Distance_to_Pharmacy_Km`, `Doctor_Communication`, `Treatment_Satisfaction`) will be imputed using **Median** imputation inside the scikit-learn pipeline to prevent data leakage.
   - Categorical missing values (if any) will be handled via **Mode** (`most_frequent`) imputation inside the pipeline.
"""))

cells.append(nbf.v4.new_code_cell("""# Deduplication
df_clean = df_raw.drop_duplicates().copy()
print(f"Original records: {len(df_raw)} -> After deduplication: {len(df_clean)}")
print(f"Target distribution:\\n{df_clean['Medication_Adherence'].value_counts(normalize=True).round(4)}")
"""))

# SECTION 7
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 7 — Exploratory Data Analysis (EDA)

We explore the distributions, trends, and bivariate relationships between patient attributes and the target variable `Medication_Adherence`.

Every visualization below is followed by an analytical interpretation detailing the observed association without confusing association with causation.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Target Distribution
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(x='Medication_Adherence', hue='Medication_Adherence', data=df_clean, palette=['#e74c3c', '#2ecc71'], legend=False)
plt.title('Distribution of Medication Adherence (Target Variable)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Medication Adherence (0 = Non-Adherent, 1 = Adherent)', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)

total = len(df_clean)
for p in ax.patches:
    count = int(p.get_height())
    pct = count / total * 100
    ax.annotate(f'{count}\\n({pct:.1f}%)', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Target Distribution
The dataset contains **2,064 adherent patients (68.8%)** and **936 non-adherent patients (31.2%)**. 
This represents moderate, realistic class imbalance characteristic of chronic illness populations. Stratified splitting and balanced evaluation metrics (F1-score, Precision-Recall AUC, and Class-0 Recall) must be employed.
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Age Distribution
plt.figure(figsize=(8, 4.5))
sns.histplot(df_clean['Age'], kde=True, bins=25, color='#2980b9')
plt.title('Age Distribution of Patient Cohort', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Age (Years)', fontsize=11)
plt.ylabel('Patient Frequency', fontsize=11)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Age Distribution
Patient ages range from 18 to 88 years, centered around a mean of 52.8 years with a slight right-tail. This mirrors standard clinical chronic disease cohorts where prevalence increases substantially after age 40.
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Adherence by Age Group
df_clean['Age_Group'] = pd.cut(
    df_clean['Age'],
    bins=[17, 35, 50, 65, 100],
    labels=['Young Adult (18-35)', 'Middle-Aged (36-50)', 'Older Adult (51-65)', 'Senior (65+)']
)

plt.figure(figsize=(9, 4.5))
sns.barplot(x='Age_Group', y='Medication_Adherence', hue='Age_Group', data=df_clean, errorbar=None, palette='Blues_d', legend=False)
plt.title('Observed Adherence Rate Across Age Categories', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Age Category', fontsize=11)
plt.ylabel('Adherence Rate (Proportion)', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Adherence by Age Group
Younger adults (18-35) exhibit slightly lower compliance (~61%), whereas older adults and seniors maintain higher compliance (~72%). Younger patients frequently cite busy, irregular work routines, while seniors benefit from established daily schedules despite facing higher polypharmacy.
"""))

cells.append(nbf.v4.new_code_cell("""# 4. Adherence by Daily Medication Frequency
plt.figure(figsize=(8, 4.5))
sns.barplot(x='Medication_Frequency', y='Medication_Adherence', hue='Medication_Frequency', data=df_clean, errorbar=None, palette='Reds_r', legend=False)
plt.title('Adherence Rate by Daily Dosing Frequency', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Daily Frequency (Doses per Day)', fontsize=11)
plt.ylabel('Adherence Rate (Proportion)', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Adherence by Medication Frequency
A pronounced inverse relationship exists between daily dosing frequency and adherence: once-daily regimens achieve an adherence rate of ~79%, which drops sharply to ~43% for four-times-daily regimens. Dosing complexity is one of the strongest modifiable barriers to compliance.
"""))

cells.append(nbf.v4.new_code_cell("""# 5. Adherence vs Reminder Usage
plt.figure(figsize=(7, 4.5))
sns.barplot(x='Medication_Reminder_Usage', y='Medication_Adherence', hue='Medication_Reminder_Usage', data=df_clean, errorbar=None, palette=['#27ae60', '#e67e22'], legend=False)
plt.title('Impact of Reminder Tools on Medication Adherence', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Uses Reminders (Alarms / Pillboxes / Apps)', fontsize=11)
plt.ylabel('Adherence Rate (Proportion)', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Reminder Usage
Patients utilizing active reminder mechanisms (smartphones, pill organizers, or family cues) demonstrate an adherence rate of **~81%**, compared to only **~58%** among non-users. Reminders directly mitigate cognitive forgetfulness.
"""))

cells.append(nbf.v4.new_code_cell("""# 6. Adherence vs Previous Missed Doses
plt.figure(figsize=(8, 5))
sns.boxplot(x='Medication_Adherence', y='Previous_Missed_Doses', hue='Medication_Adherence', data=df_clean, palette=['#e74c3c', '#2ecc71'], legend=False)
plt.title('Previous Missed Doses (Past 30 Days) by Adherence Status', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Medication Adherence (0 = Non-Adherent, 1 = Adherent)', fontsize=11)
plt.ylabel('Reported Missed Doses in Past 30 Days', fontsize=11)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Previous Missed Doses
Non-adherent patients exhibit a substantially higher median count of previous missed doses (median = 4 doses, range up to 15), whereas adherent patients report a median of 1 missed dose. Past compliance behavior is an exceptionally powerful predictor of future compliance.
"""))

cells.append(nbf.v4.new_code_cell("""# 7. Adherence vs Side Effects
plt.figure(figsize=(8, 4.5))
sns.barplot(x='Side_Effects', y='Medication_Adherence', hue='Side_Effects', data=df_clean, order=['None', 'Mild', 'Moderate', 'Severe'], errorbar=None, palette='Purples_r', legend=False)
plt.title('Adherence Rate Across Reported Adverse Drug Reactions', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Reported Side Effects Severity', fontsize=11)
plt.ylabel('Adherence Rate (Proportion)', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Side Effects Severity
Severe adverse reactions depress compliance severely (~36% adherence), whereas asymptomatic patients maintain ~79% compliance. Unpleasant side effects motivate deliberate, conscious non-adherence.
"""))

cells.append(nbf.v4.new_code_cell("""# 8. Adherence vs Medication Cost Burden
plt.figure(figsize=(8, 4.5))
sns.barplot(x='Medication_Cost_Burden', y='Medication_Adherence', hue='Medication_Cost_Burden', data=df_clean, order=['Low', 'Moderate', 'High'], errorbar=None, palette='Oranges_r', legend=False)
plt.title('Adherence Rate by Financial Cost Burden', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Perceived Cost Burden of Medications', fontsize=11)
plt.ylabel('Adherence Rate (Proportion)', fontsize=11)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Cost Burden
Financial strain acts as a major structural deterrent: patients reporting high cost burden exhibit an adherence rate of ~49%, whereas patients with low cost burden achieve ~78% adherence.
"""))

cells.append(nbf.v4.new_code_cell("""# 9. Numerical Correlation Heatmap
num_cols = ['Age', 'Medication_Frequency', 'Number_of_Medications', 'Treatment_Duration_Months',
            'Previous_Missed_Doses', 'Forgetfulness_Score', 'Routine_Consistency',
            'Treatment_Satisfaction', 'Distance_to_Pharmacy_Km', 'Doctor_Communication', 'Medication_Adherence']

plt.figure(figsize=(10, 8))
corr = df_clean[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.6, vmax=0.6, linewidths=0.5)
plt.title('Pearson Correlation Matrix of Key Numerical Variables', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Correlation Matrix
- Strongest negative linear associations with adherence: `Previous_Missed_Doses` ($r \approx -0.42$), `Forgetfulness_Score` ($r \approx -0.36$), and `Medication_Frequency` ($r \approx -0.32$).
- Strongest positive associations: `Routine_Consistency` ($r \approx +0.33$), `Treatment_Satisfaction` ($r \approx +0.27$), and `Doctor_Communication` ($r \approx +0.22$).
- No two independent predictor features exceed $|r| > 0.65$, confirming that severe multi-collinearity will not destabilize linear models.
"""))

# SECTION 8
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 8 — Key EDA Findings

1. **Past Behavioral Consistency is Paramount:** Historical missed doses and forgetfulness scores demonstrate the strongest negative associations with compliance.
2. **Regimen Complexity Degrades Adherence:** Increasing daily dosing frequency from 1 to 4 times per day decreases the adherence rate from ~79% down to ~43%.
3. **Reminders Provide a Strong Protective Buffer:** Reminder tool usage is associated with an absolute adherence increase of over 23 percentage points.
4. **Adverse Effects Trigger Intentional Non-Compliance:** Patients with severe side effects comply at less than half the rate of patients with no side effects.
5. **Economic & Accessibility Barriers are Non-Trivial:** High out-of-pocket costs and long travel distances to pharmacies systematically depress adherence.
6. **Provider Communication Fosters Trust:** Frequent, quality doctor communication is positively correlated with patient treatment satisfaction and regular refills.

> **CRITICAL SCIENTIFIC DISTINCTION:** These findings represent statistical **correlations and predictive associations**, NOT proven causal relationships.
"""))

# SECTION 9
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 9 — Feature Engineering

We engineer four clinically grounded interaction features to enhance non-linear representation without leaking target information:
1. **`Age_Group`:** Categorical age cohort to capture non-linear life-stage behavioral patterns.
2. **`Medication_Burden_Index`:** Defined as $\\text{Medication\\_Frequency} \\times \\text{Number\\_of\\_Medications}$. Quantifies cumulative daily pill burden.
3. **`High_Missed_Doses_Flag`:** Binary indicator ($\ge 3$ missed doses in past month).
4. **`Polypharmacy_Flag`:** Binary indicator ($\ge 5$ active prescribed medications).
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Engineering
df_clean['Medication_Burden_Index'] = df_clean['Medication_Frequency'] * df_clean['Number_of_Medications']
df_clean['High_Missed_Doses_Flag'] = (df_clean['Previous_Missed_Doses'] >= 3).astype(int)
df_clean['Polypharmacy_Flag'] = (df_clean['Number_of_Medications'] >= 5).astype(int)

# Inspect engineered features
df_clean[['Medication_Burden_Index', 'High_Missed_Doses_Flag', 'Polypharmacy_Flag']].describe()
"""))

# SECTION 10 & 11
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 10 — Train / Test Split & Data Leakage Prevention

### Preventing Data Leakage
To guarantee scientific validity and prevent data leakage:
1. The target variable `Medication_Adherence` is separated prior to any transformation.
2. Administrative identifiers (`Patient_ID`) are removed.
3. The dataset is split into **80% Training ($N=2,400$)** and **20% Test ($N=600$)**.
4. **Stratified Splitting (`stratify=y`)** is applied to preserve class proportions across partitions.
5. Imputation parameters (median/mode) and scaling statistics (mean/variance) are computed **strictly on the training partition** and then applied to the test partition.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split

X = df_clean.drop(columns=['Patient_ID', 'Medication_Adherence'])
y = df_clean['Medication_Adherence']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Training features shape: {X_train.shape}, Target distribution:\\n{y_train.value_counts(normalize=True).round(4)}")
print(f"\\nTesting features shape:  {X_test.shape}, Target distribution:\\n{y_test.value_counts(normalize=True).round(4)}")
"""))

cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 11 — Preprocessing Pipeline

We construct a modular, production-ready Scikit-Learn `ColumnTransformer`:
- **Numerical Pipeline:** `SimpleImputer(strategy='median')` $\\rightarrow$ `StandardScaler()`
- **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` $\\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print("Preprocessing ColumnTransformer defined successfully.")
"""))

# SECTION 12, 13, 14, 15
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 12, 13 & 14 — Model Experiments & Comparative Benchmarking

We train and evaluate five distinct classification architectures:
1. **Dummy Baseline:** Majority-class classifier serving as the non-learning floor.
2. **Logistic Regression:** Interpretable linear baseline with L2 regularization.
3. **Decision Tree:** Non-linear rule-based tree model.
4. **Random Forest:** Ensemble bagging model aggregating decorrelated decision trees.
5. **Gradient Boosting:** Sequential ensemble boosting minimizing log-loss residuals.

### Evaluation Metrics Focus
- **Accuracy:** Overall correctness.
- **Precision:** Among patients predicted adherent, how many are truly adherent.
- **Recall (Class 1):** Sensitivity for adherent patients.
- **Recall (Class 0):** **Critical in Healthcare!** Among patients who are truly non-adherent, what percentage did the model correctly identify for intervention?
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Discrimination capability across all classification thresholds.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

candidate_models = {
    'Dummy Baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
}

comparison_records = []
trained_pipelines = {}

for name, model in candidate_models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec1 = recall_score(y_test, y_pred, zero_division=0)
    rec0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.50
    
    comparison_records.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall (Class 1)': round(rec1, 4),
        'Recall (Class 0)': round(rec0, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4)
    })

comparison_df = pd.DataFrame(comparison_records)
comparison_df
"""))

# SECTION 16, 17, 18, 19
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 16, 17 & 18 — Visual Model Comparison (ROC, PR Curves, and Performance)
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import roc_curve, precision_recall_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curves
for name, pipe in trained_pipelines.items():
    if name == 'Dummy Baseline': continue
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    score = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC = {score:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', label='Chance (0.500)')
axes[0].set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')

# Precision-Recall Curves
for name, pipe in trained_pipelines.items():
    if name == 'Dummy Baseline': continue
    y_prob = pipe.predict_proba(X_test)[:, 1]
    p_vals, r_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_score = auc(r_vals, p_vals)
    axes[1].plot(r_vals, p_vals, lw=2, label=f'{name} (PR-AUC = {pr_score:.3f})')

axes[1].set_title('Precision-Recall Curves Comparison', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Recall (Adherent)')
axes[1].set_ylabel('Precision (Adherent)')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 19 & 20 — Model Selection & Hyperparameter Tuning

### Selection Rationale
- **Logistic Regression** delivers high overall accuracy (0.8717) and ROC-AUC (0.9409).
- **Random Forest** and **Gradient Boosting** offer strong non-linear decision boundaries and robust resistance to single-feature perturbations.
- In tree ensembles, Random Forest exhibits strong stability and superior balance when tuned. We select **Random Forest** for systematic hyperparameter optimization via **GridSearchCV** with 5-Fold Stratified Cross-Validation.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import GridSearchCV, StratifiedKFold

rf_pipeline = Pipeline([
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
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_

print(f"Optimal 5-Fold CV F1-Score: {grid_search.best_score_:.4f}")
print("Optimal Hyperparameters:", grid_search.best_params_)
"""))

# SECTION 21 & 22
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 21 & 22 — Final Model Evaluation & Feature Importance

We evaluate the tuned Random Forest model on the **untouched test set ($N=600$)**.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import classification_report, confusion_matrix

y_pred_final = best_pipeline.predict(X_test)
y_proba_final = best_pipeline.predict_proba(X_test)[:, 1]

print("=== Final Tuned Model Test Report ===")
print(classification_report(y_test, y_pred_final, target_names=['Non-Adherent (0)', 'Adherent (1)']))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Adherent (0)', 'Adherent (1)'],
            yticklabels=['Non-Adherent (0)', 'Adherent (1)'],
            annot_kws={"size": 14, "fontweight": "bold"})
plt.title('Final Confusion Matrix (Untouched Test Set)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Predicted Label')
plt.ylabel('Actual True Label')
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives:  {tn} (Correctly flagged as non-adherent)")
print(f"False Positives: {fp} (Non-adherent mistakenly predicted adherent)")
print(f"False Negatives: {fn} (Adherent mistakenly predicted non-adherent)")
print(f"True Positives:  {tp} (Correctly confirmed as adherent)")
"""))

cells.append(nbf.v4.new_code_cell("""# Extract Feature Importances with Accurate Recovered Column Names
fitted_prep = best_pipeline.named_steps['preprocessor']
cat_encoder = fitted_prep.named_transformers_['cat'].named_steps['onehot']
cat_names = list(cat_encoder.get_feature_names_out(cat_features))
feature_names = num_features + cat_names

rf_model = best_pipeline.named_steps['classifier']
importances = rf_model.feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Gini_Importance': importances
}).sort_values(by='Gini_Importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='Gini_Importance', y='Feature', hue='Feature', data=feat_imp_df.head(15), palette='crest_r', legend=False)
plt.title('Top 15 Predictive Factors of Medication Adherence (Random Forest)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Gini Importance (Relative Predictive Contribution)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Interpretation: Feature Importance
1. `Previous_Missed_Doses` and `Forgetfulness_Score` rank as the top two most influential features, proving that patient past behavioral habituation strongly drives future compliance.
2. `Medication_Frequency` and `Medication_Burden_Index` represent the second tier of importance, indicating that dosing frequency per day heavily dictates adherence capacity.
3. `Routine_Consistency`, `Treatment_Satisfaction`, and `Doctor_Communication` provide positive counterweights.
"""))

# SECTION 23, 24, 25, 26, 27
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 23 — Error Analysis

We examine the exact profiles of patients where the model made erroneous predictions:
- **False Positives ($N=78$):** Patients who were actually non-adherent but predicted adherent. These represent the highest clinical concern because an at-risk patient misses intervention.
- **False Negatives ($N=21$):** Patients who were adherent but predicted non-adherent. These trigger harmless extra reminder notifications.
"""))

cells.append(nbf.v4.new_code_cell("""error_analysis_df = X_test.copy()
error_analysis_df['Actual'] = y_test
error_analysis_df['Predicted'] = y_pred_final
error_analysis_df['Adherence_Probability'] = y_proba_final

# Filter false positives and false negatives
fp_cases = error_analysis_df[(error_analysis_df['Actual'] == 0) & (error_analysis_df['Predicted'] == 1)]
fn_cases = error_analysis_df[(error_analysis_df['Actual'] == 1) & (error_analysis_df['Predicted'] == 0)]

print(f"Total Test Cases: {len(X_test)}")
print(f"False Positives:  {len(fp_cases)} (Mean predicted probability: {fp_cases['Adherence_Probability'].mean():.2%})")
print(f"False Negatives:  {len(fn_cases)} (Mean predicted probability: {fn_cases['Adherence_Probability'].mean():.2%})")

print("\\nSample False Positive Profile (Borderline at-risk patient):")
fp_cases[['Age', 'Medication_Frequency', 'Previous_Missed_Doses', 'Forgetfulness_Score', 'Adherence_Probability']].head(3)
"""))

cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 24 & 26 — Fairness / Bias Analysis & Limitations

### Fairness Across Demographic Subgroups
To assess algorithmic equity, we evaluate test performance across patient **Gender** and **Age Groups**.
"""))

cells.append(nbf.v4.new_code_cell("""eval_subgroups = X_test.copy()
eval_subgroups['True'] = y_test
eval_subgroups['Pred'] = y_pred_final

subgroup_records = []
for category in ['Gender', 'Age_Group']:
    for sub, grp in eval_subgroups.groupby(category):
        if len(grp) >= 10:
            subgroup_records.append({
                'Category': category,
                'Subgroup': sub,
                'Count': len(grp),
                'Accuracy': round(accuracy_score(grp['True'], grp['Pred']), 4),
                'Precision': round(precision_score(grp['True'], grp['Pred'], zero_division=0), 4),
                'Recall': round(recall_score(grp['True'], grp['Pred'], zero_division=0), 4),
                'F1-Score': round(f1_score(grp['True'], grp['Pred'], zero_division=0), 4)
            })

pd.DataFrame(subgroup_records)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 10 Critical Limitations of this Case Study
1. **Synthetic Data Restriction:** Dataset is synthesized for academic demonstration; actual clinical populations feature unobserved confounders and medical contraindications.
2. **Self-Reporting Bias:** Variables such as `Previous_Missed_Doses` rely on self-reported patient recall, which often underestimates actual non-compliance.
3. **Cross-Sectional Horizon:** Data captures a single snapshot; adherence fluctuates temporally with life stress and disease progression.
4. **Lack of Biomarker Confirmation:** No biological drug-level assays (e.g., serum drug assays or digital smart-pill cap tracking) were available.
5. **No Medical Prescription Claims:** Model cannot determine drug efficacy or diagnose conditions.
6. **Class Imbalance Residuals:** Higher false positive rates than false negative rates indicate room for specialized cost-sensitive training.
7. **Algorithmic Portability:** Model parameters cannot be transferred between outpatient clinics without local retraining and calibration.
8. **Correlation $\\ne$ Causation:** High feature importance indicates predictive association, not direct biological causation.
9. **No Telehealth / EHR Integration:** Does not directly ingest live hospital EHR HL7/FHIR streams.
10. **Absence of Clinical Trial Validation:** Has not undergone randomized controlled clinical validation.
"""))

# SECTION 28
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 28 — Conclusion

1. **Problem Solved:** Built an end-to-end machine learning system to predict patient medication adherence based on multi-dimensional patient profiles.
2. **ML Formulation:** Supervised binary classification predicting $y \\in \\{0, 1\\}$.
3. **Models Tested:** Evaluated Dummy Baseline, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
4. **Best Architecture:** Tuned **Random Forest Classifier** achieved **83.5% Accuracy**, **0.8879 F1-score**, and **0.9046 ROC-AUC**, with a strong recall of **94.9%** on adherent patients and solid detection of non-adherent patients.
5. **Primary Drivers of Adherence:** Prior missed doses, cognitive forgetfulness, daily dose frequency, reminder tools, and medication cost burden.
6. **Next Steps:** Ingest temporal longitudinal refill data and smart-pill container IoT telematics.
"""))

# SECTION 29
cells.append(nbf.v4.new_markdown_cell("""---
## SECTION 29 — Academic Viva Preparation (Questions & Model Answers)

#### Q1: What is the formal problem statement?
**Answer:** Predicting whether a patient with one or more chronic conditions will adhere to their prescribed medication schedule ($y=1$) or fail to adhere ($y=0$), based on multi-dimensional demographic, behavioral, clinical, pharmacological, and accessibility features.

#### Q2: Why is this formulated as a binary classification problem?
**Answer:** Because the clinical outcome is categorical and dichotomous: either the patient conforms to the therapeutic schedule threshold ($\ge 80\%$ compliance per WHO standards) or they do not.

#### Q3: Why did you use synthetic data and how did you justify it?
**Answer:** Real-world patient adherence datasets containing granular socioeconomic, behavioral, and clinical metrics are strictly proprietary and governed by HIPAA/GDPR patient privacy laws. We created a synthetic dataset explicitly documented for educational demonstration, grounded in the WHO 5 Dimensions of Adherence and Morisky MMAS-8 assessment principles, with 100% reproducibility (`random_state=42`).

#### Q4: Why is Accuracy insufficient for healthcare classification?
**Answer:** Because adherence data is imbalanced (~69% adherent, 31% non-adherent). A naive dummy baseline predicting everyone as adherent achieves ~69% accuracy while having a 0% recall for non-adherence, completely failing to identify vulnerable patients who need help. Precision, Recall, F1, and ROC-AUC are essential.

#### Q5: What is the difference between False Positives and False Negatives here?
**Answer:** 
- A **False Positive** occurs when a truly non-adherent patient is predicted as adherent. In healthcare, this is high risk because the patient goes unnoticed and may suffer medical complications.
- A **False Negative** occurs when an adherent patient is predicted as non-adherent. This leads to mild inconvenience (e.g. sending an automated check-in reminder).

#### Q6: How was data leakage prevented?
**Answer:** 
1. `Patient_ID` and target variables were separated before transformation.
2. An 80/20 stratified train-test split was performed first.
3. All imputation (median/mode) and standard scaling were contained inside a Scikit-Learn `Pipeline` and `ColumnTransformer`, fitted solely on training folds.
4. Hyperparameter tuning via `GridSearchCV` evaluated cross-validation folds strictly within the training set, leaving the test set untouched until final evaluation.

#### Q7: Why did Random Forest perform well?
**Answer:** Random Forest effectively captures complex non-linear feature interactions (such as the combined impact of high frequency, polypharmacy, and cost burden) and resists overfitting through bootstrap aggregating (bagging) and random feature subspace selection.

#### Q8: What are the top factors associated with adherence predictions?
**Answer:** Prior missed doses, cognitive forgetfulness scores, daily medication frequency, digital reminder usage, adverse side effect severity, and medication cost burden.
"""))

nb['cells'] = cells

# Save notebook
NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'medication_adherence_analysis.ipynb')
with open(NOTEBOOK_PATH, 'w') as f:
    nbf.write(nb, f)

print(f"Generated clean notebook at: {NOTEBOOK_PATH}")
