# Medication Adherence Analysis: Predicting Patient Medication Compliance Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


## 📌 Executive Overview

Medication adherence—defined by the World Health Organization (WHO) as the extent to which a patient's medication-taking behavior corresponds with agreed recommendations from a healthcare provider—is one of the most critical determinants of therapeutic outcomes in chronic disease management. 

According to global epidemiological research, approximately **30% to 50% of patients with chronic diseases (such as hypertension, type-2 diabetes, cardiovascular disease, and asthma) do not follow their prescribed medication regimens**. Non-adherence leads to disease progression, preventable hospitalizations, increased mortality, and avoidable healthcare expenditures.

**MedAdhere AI** is an academic machine-learning system designed to predict patient medication compliance risk based on multi-dimensional demographic, pharmacological, behavioral, clinical, and accessibility features.

---

## 🎯 Problem Statement & ML Formulation

### Real-World Challenge
Healthcare teams currently lack automated, proactive methods to identify which chronic patients are at high risk of discontinuing or skipping their medications before adverse clinical events occur.

### Machine Learning Formulation
This problem is formulated as a **Supervised Machine Learning: Binary Classification** task:
- **Input Features ($\mathbf{x}$):** 28 clinical, pharmacological, behavioral, socioeconomic, and support indicators.
- **Target Variable ($y$):** `Medication_Adherence`
  - $y = 1$: **Adherent** (Patient consistently follows the prescribed medication schedule)
  - $y = 0$: **Non-Adherent** (Patient frequently misses, discontinues, or alters doses)
- **Model Output:**
  - **Predicted Class:** `Adherent` vs `Non-Adherent`
  - **Adherence Probability:** $\hat{P}(\text{Adherence} = 1) \in [0.0, 1.0]$
  - **Stratified Risk Level:**
    - 🟢 **Low Risk:** $\hat{P} \ge 70\%$
    - 🟡 **Moderate Risk:** $40\% \le \hat{P} < 70\%$
    - 🔴 **High Risk:** $\hat{P} < 40\%$

---

## 📂 Dataset Selection & Academic Justification

### Dataset Selection Justification
In healthcare research, patient-level compliance logs containing comprehensive behavioral and socioeconomic indicators are protected under HIPAA/GDPR patient privacy laws and are rarely available in public open-access repositories.

To maintain complete academic transparency and avoid false claims of clinical data provenance, this study utilizes a clearly documented:
**"Synthetic Medication Adherence Dataset for Educational Demonstration"**

- **Clinical Grounding:** Synthesized based on the **WHO 5 Dimensions of Adherence** (Socioeconomic, Healthcare Team/System, Condition-Related, Therapy-Related, Patient-Related) and validated adherence assessment instruments (such as the *Morisky Medication Adherence Scale*, MMAS-8).
- **Cohort Size:** 3,000 patient records across 26 primary features.
- **Reproducibility:** 100% reproducible via `data/generate_dataset.py` with `random_state=42`.
- **Realistic Class Distribution:** ~68.8% Adherent ($N=2,064$) and 31.2% Non-Adherent ($N=936$), accurately reflecting real-world chronic illness compliance rates.

### Primary Feature Categories
1. **Demographics:** Age, Gender, Education Level, Employment Status.
2. **Medication & Regimen:** Medication Category, Daily Dosing Frequency (1-4x/day), Active Medication Count (Polypharmacy), Treatment Duration, Dosage Complexity.
3. **Behavioral Metrics:** Previous Missed Doses (past 30 days), Forgetfulness Score (1-10), Routine Consistency (1-10), Medication Reminder Usage.
4. **Clinical Experience:** Primary Chronic Condition, Side Effect Severity, Perceived Effectiveness, Treatment Satisfaction (1-10).
5. **Accessibility & Support:** Cost Burden, Pharmacy Accessibility, Insurance Coverage, Distance to Pharmacy (km), Family Support, Healthcare Followup Regularity, Doctor Communication Score (1-10).

---

## 🛠️ Feature Engineering & Data Leakage Prevention

### Clinically Grounded Engineered Features
1. **`Age_Group`:** Categorical age cohorts (Young Adult 18-35, Middle-Aged 36-50, Older Adult 51-65, Senior 65+) to capture life-stage routine stability.
2. **`Medication_Burden_Index`:** Defined as $\text{Medication\_Frequency} \times \text{Number\_of\_Medications}$. Quantifies cumulative daily pill friction.
3. **`High_Missed_Doses_Flag`:** Binary indicator ($\ge 3$ missed doses in past month).
4. **`Polypharmacy_Flag`:** Binary indicator ($\ge 5$ active prescribed medications).

### Preventing Data Leakage
- Administrative IDs (`Patient_ID`) were stripped before modeling.
- Partitioning: **80% Training ($N=2,400$)** and **20% Test ($N=600$)** using **Stratified Splitting (`stratify=y`)**.
- All imputation (Median/Mode) and scaling (StandardScaler) operations were wrapped inside a Scikit-Learn `ColumnTransformer` and `Pipeline`, strictly fitted only on training folds.

---

## 📊 Cross-Model Comparison & Evaluation

Five distinct classification models were benchmarked on the untouched test partition ($N=600$):

| Model Architecture | Accuracy | Precision | Recall (Adherent) | Recall (Non-Adherent) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Majority)** | 0.6883 | 0.6883 | 1.0000 | 0.0000 | 0.8154 | 0.5000 |
| **Logistic Regression** | **0.8717** | **0.8889** | 0.9298 | **0.7433** | **0.9089** | **0.9409** |
| **Decision Tree (Pruned)** | 0.7883 | 0.8056 | 0.9128 | 0.5134 | 0.8558 | 0.8102 |
| **Random Forest (Tuned)** | **0.8350** | **0.8340** | **0.9492** | **0.5829** | **0.8879** | **0.9046** |
| **Gradient Boosting** | 0.8533 | 0.8571 | 0.9443 | 0.6524 | 0.8986 | 0.9216 |

### Metric Prioritization in Healthcare
In clinical adherence screening, **Recall for the Non-Adherent class** is crucial: failing to flag a patient who will discontinue medication (a False Positive in classification terms) deprives them of essential clinical intervention. Random Forest was tuned via **5-Fold Stratified Cross-Validation** using F1-score to achieve balanced sensitivity across both classes.

---

## 🔍 Key Feature Importance Findings

Extraction of Gini feature importances from the tuned Random Forest revealed the primary predictors of medication adherence:
1. **`Previous_Missed_Doses`:** Strongest empirical predictor; past behavioral compliance strongly predicts future compliance.
2. **`Forgetfulness_Score`:** Cognitive memory load is a primary driver of unintentional non-adherence.
3. **`Medication_Frequency` & `Medication_Burden_Index`:** Daily multi-dose regimens ($3\times$ or $4\times$ daily) sharply degrade compliance.
4. **`Medication_Reminder_Usage`:** Protects adherence, boosting compliance by over 23 percentage points.
5. **`Side_Effects` & `Medication_Cost_Burden`:** Drive intentional non-adherence due to physical discomfort or out-of-pocket expenses.

---

## 🚀 Streamlit Web Application (MedAdhere AI)

The interactive dashboard provides a clinical decision-support interface:
- **Dashboard:** Population adherence statistics, algorithm benchmark graphs, and KPI metrics.
- **Prediction:** Patient intake form with **Quick Demo Presets** (`🟢 Compliant Patient` and `🔴 At-Risk Patient`) and instant adherence probability gauge.
- **Patient Insights:** Actionable breakdown of protective vs risk factors for the evaluated patient.
- **Model Information:** Technical specifications, Scikit-learn pipeline diagram, hyperparameter configurations, and demographic subgroup fairness analysis.
- **About:** Academic background and project architecture.

---

## 💻 Quickstart & Installation

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/medication-adherence-ml.git
cd medication-adherence-ml

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline (Optional Re-training)
```bash
python notebooks/build_pipeline.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Project Directory Structure

```text
medication-adherence-ml/
│
├── data/
│   ├── raw/
│   │   └── medication_adherence.csv              # Raw synthetic cohort dataset (N=3,005)
│   ├── processed/
│   │   └── medication_adherence_processed.csv    # Engineered & cleaned dataset
│   └── generate_dataset.py                       # Reproducible data generator script
│
├── notebooks/
│   ├── medication_adherence_analysis.ipynb       # Fully executed academic Jupyter notebook
│   ├── generate_notebook.py                      # Programmatic notebook generator
│   └── build_pipeline.py                         # End-to-end ML training & export script
│
├── models/
│   ├── medication_adherence_model.pkl            # Serialized full pipeline (Preprocessor + Model)
│   ├── preprocessing_pipeline.pkl                # Standalone ColumnTransformer
│   └── model_metadata.json                       # Model parameters, metrics & feature list
│
├── app/
│   ├── app.py                                    # Main Streamlit web application
│   ├── components.py                             # Reusable UI cards, gauges & demo profiles
│   └── styles.css                                # Healthcare analytics CSS design system
│
├── reports/
│   ├── figures/                                  # High-resolution publication figures
│   │   ├── target_distribution.png
│   │   ├── age_distribution.png
│   │   ├── adherence_by_medication_frequency.png
│   │   ├── adherence_vs_reminder_usage.png
│   │   ├── correlation_matrix.png
│   │   ├── model_comparison_metrics.png
│   │   ├── confusion_matrix_best_model.png
│   │   ├── roc_curves_comparison.png
│   │   └── feature_importance.png
│   └── model_results.csv                         # Calculated comparative metrics
│
├── requirements.txt                              # Pinned Python package dependencies
├── README.md                                     # Comprehensive academic project documentation
└── .gitignore                                    # Git ignore rules for virtual environments
```

---

## ⚠️ Academic Limitations & Clinical Disclaimer

### Limitations
1. **Synthetic Nature:** Synthesized for educational demonstration; real clinical populations feature unobserved comorbidities, drug-drug interactions, and clinical contraindications.
2. **Self-Reporting Bias:** Metrics like missed doses rely on subjective patient recall, which often underestimates non-compliance.
3. **Cross-Sectional Horizon:** Reflects a single time window; patient adherence is longitudinal and fluctuates with acute life events.
4. **No Biomarker Assays:** Lacks biological verification (e.g. serum drug concentrations or smart-cap electronic pill bottle tracking).
5. **Portability:** Requires local re-calibration before applying to distinct outpatient or hospital populations.

### Clinical Disclaimer
> **NOTICE:** **MedAdhere AI** is an **educational and analytical machine-learning demonstration** created for a B.Tech Computer Science & Engineering degree project. It is **NOT** a medical diagnostic tool or clinical decision system and must **NEVER** be used to alter, prescribe, or discontinue medications without direct evaluation by a certified physician or pharmacist.
