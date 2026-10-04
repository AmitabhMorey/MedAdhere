"""
Synthetic Medication Adherence Dataset Generator for Educational Demonstration
==============================================================================
Project: Medication Adherence Analysis (MedAdhere AI)
Academic Context: B.Tech CSE Machine Learning Case Study

Purpose:
Generates a realistic, documented, reproducible synthetic dataset modeled on
WHO adherence multidimensional guidelines and clinical adherence literature
(Morisky Medication Adherence Scale principles).

Explicit Academic Disclosure:
This dataset is synthetic and created exclusively for educational analysis and
machine learning pipeline demonstration. It does NOT represent any real patient
clinical cohort, and models trained on it must not be used for clinical decision-making.
"""

import numpy as np
import pandas as pd
import os

def generate_medication_adherence_data(n_samples=3000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Demographic Features
    patient_ids = [f"MED-{i+1001:05d}" for i in range(n_samples)]
    
    # Age: realistic chronic condition patient distribution (mean~53, std~14, bounded 18-88)
    ages = np.clip(np.random.normal(loc=53, scale=14, size=n_samples).astype(int), 18, 88)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.48, 0.50, 0.02])
    
    education_levels = np.random.choice(
        ['None', 'High School', 'Bachelor', 'Master', 'Doctorate'],
        size=n_samples,
        p=[0.08, 0.42, 0.33, 0.14, 0.03]
    )
    
    # Employment correlated slightly with age
    employment_statuses = []
    for age in ages:
        if age >= 65:
            p_emp = [0.08, 0.05, 0.82, 0.00, 0.05]
        elif age <= 24:
            p_emp = [0.35, 0.15, 0.00, 0.40, 0.10]
        else:
            p_emp = [0.65, 0.10, 0.02, 0.03, 0.20]
        emp = np.random.choice(['Employed', 'Unemployed', 'Retired', 'Student', 'Self-Employed'], p=p_emp)
        employment_statuses.append(emp)
        
    # 2. Medication Features
    medication_types = np.random.choice(
        ['Antihypertensive', 'Antidiabetic', 'Cardiovascular', 'Respiratory', 'Psychiatric', 'Other Chronic'],
        size=n_samples,
        p=[0.30, 0.25, 0.18, 0.12, 0.10, 0.05]
    )
    
    # Daily medication frequency: 1 (once), 2 (twice), 3 (thrice), 4 (four times daily)
    medication_frequencies = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.40, 0.35, 0.18, 0.07])
    
    # Number of medications (polypharmacy)
    # Higher for older patients
    num_meds = []
    for age in ages:
        mean_meds = 2.0 + (age / 25.0)
        med_count = int(np.clip(np.random.poisson(lam=mean_meds), 1, 10))
        num_meds.append(med_count)
    num_meds = np.array(num_meds)
    
    treatment_durations = np.clip(np.random.gamma(shape=2.5, scale=12.0, size=n_samples).astype(int) + 1, 1, 120)
    
    dosage_complexities = []
    for freq, n_med in zip(medication_frequencies, num_meds):
        score = freq * 0.6 + n_med * 0.4
        if score > 3.8:
            comp = 'High'
        elif score > 2.2:
            comp = 'Medium'
        else:
            comp = 'Low'
        dosage_complexities.append(comp)
        
    # 3. Behavioral Features
    # Forgetfulness score: 1 to 10
    forgetfulness_scores = np.clip(np.round(np.random.normal(loc=4.5, scale=2.2, size=n_samples)), 1, 10).astype(int)
    
    # Routine consistency: 1 to 10
    routine_consistencies = np.clip(np.round(np.random.normal(loc=6.2, scale=2.1, size=n_samples)), 1, 10).astype(int)
    
    reminder_usage = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.46, 0.54])
    
    # Previous missed doses in past 30 days (strongly correlated with forgetfulness, frequency, reminders)
    previous_missed = []
    for forg, rout, rem, freq in zip(forgetfulness_scores, routine_consistencies, reminder_usage, medication_frequencies):
        base_lambda = 0.5 + 0.45 * forg - 0.3 * rout + (1.2 if rem == 'No' else -0.4) + 0.4 * freq
        base_lambda = max(0.1, base_lambda)
        missed = int(np.clip(np.random.poisson(lam=base_lambda), 0, 20))
        previous_missed.append(missed)
    previous_missed = np.array(previous_missed)
    
    # 4. Clinical / Treatment Experience Features
    conditions = []
    for med in medication_types:
        if med == 'Antihypertensive':
            c = np.random.choice(['Hypertension', 'Multimorbidity'], p=[0.75, 0.25])
        elif med == 'Antidiabetic':
            c = np.random.choice(['Diabetes', 'Multimorbidity'], p=[0.75, 0.25])
        elif med == 'Cardiovascular':
            c = np.random.choice(['Heart Disease', 'Multimorbidity'], p=[0.70, 0.30])
        elif med == 'Respiratory':
            c = np.random.choice(['Asthma/COPD', 'Multimorbidity'], p=[0.85, 0.15])
        elif med == 'Psychiatric':
            c = np.random.choice(['Depression/Anxiety', 'Multimorbidity'], p=[0.85, 0.15])
        else:
            c = 'Multimorbidity'
        conditions.append(c)
        
    side_effects = np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], size=n_samples, p=[0.48, 0.30, 0.16, 0.06])
    
    perceived_effectiveness = []
    for se in side_effects:
        if se == 'Severe':
            pe = np.random.choice(['Low', 'Moderate', 'High'], p=[0.55, 0.35, 0.10])
        elif se == 'Moderate':
            pe = np.random.choice(['Low', 'Moderate', 'High'], p=[0.30, 0.50, 0.20])
        else:
            pe = np.random.choice(['Low', 'Moderate', 'High'], p=[0.08, 0.37, 0.55])
        perceived_effectiveness.append(pe)
        
    treatment_satisfactions = []
    for pe, se in zip(perceived_effectiveness, side_effects):
        base_sat = 7.0
        if pe == 'High': base_sat += 1.8
        elif pe == 'Low': base_sat -= 2.5
        if se == 'Severe': base_sat -= 2.2
        elif se == 'Moderate': base_sat -= 1.0
        sat = int(np.clip(np.round(np.random.normal(loc=base_sat, scale=1.5)), 1, 10))
        treatment_satisfactions.append(sat)
    treatment_satisfactions = np.array(treatment_satisfactions)
    
    # 5. Accessibility & Support Features
    cost_burdens = np.random.choice(['Low', 'Moderate', 'High'], size=n_samples, p=[0.44, 0.36, 0.20])
    pharmacy_access = np.random.choice(['Easy', 'Moderate', 'Difficult'], size=n_samples, p=[0.55, 0.32, 0.13])
    insurance_coverages = np.random.choice(['Full', 'Partial', 'None'], size=n_samples, p=[0.50, 0.38, 0.12])
    
    # Distance to pharmacy: log-normal distribution (median ~3.5 km, tail up to 35 km)
    distances = np.clip(np.random.lognormal(mean=1.2, sigma=0.75, size=n_samples), 0.3, 35.0).round(1)
    
    family_supports = np.random.choice(['High', 'Moderate', 'Low'], size=n_samples, p=[0.42, 0.40, 0.18])
    healthcare_followups = np.random.choice(['Regular', 'Occasional', 'Rare'], size=n_samples, p=[0.52, 0.35, 0.13])
    
    doctor_communications = np.clip(np.round(np.random.normal(loc=6.8, scale=2.0, size=n_samples)), 1, 10).astype(int)
    
    # 6. Target Generation: Medication_Adherence (1 = Adherent, 0 = Non-Adherent)
    # Log-odds modeling based on WHO 5 Dimensions of Adherence
    # Logit = beta_0 + sum(beta_i * X_i) + epsilon
    
    # Base intercept tuned to yield ~64% adherence rate (realistic WHO chronic disease benchmark)
    logit = 0.55
    
    # Behavioral contributions (strongest empirical adherence determinants)
    logit += -0.32 * (previous_missed - 2.0)
    logit += -0.22 * (forgetfulness_scores - 4.5)
    logit += 0.20 * (routine_consistencies - 6.0)
    logit += np.where(np.array(reminder_usage) == 'Yes', 0.55, -0.45)
    
    # Medication schedule complexity
    logit += -0.35 * (medication_frequencies - 1.8)
    logit += -0.15 * (num_meds - 3.5)
    logit += np.where(np.array(dosage_complexities) == 'High', -0.50,
             np.where(np.array(dosage_complexities) == 'Medium', -0.05, 0.35))
    
    # Clinical & Side effects
    logit += np.where(np.array(side_effects) == 'Severe', -0.85,
             np.where(np.array(side_effects) == 'Moderate', -0.40,
             np.where(np.array(side_effects) == 'Mild', -0.10, 0.30)))
    logit += np.where(np.array(perceived_effectiveness) == 'High', 0.45,
             np.where(np.array(perceived_effectiveness) == 'Low', -0.60, 0.0))
    logit += 0.14 * (treatment_satisfactions - 6.5)
    
    # Accessibility and socio-economic
    logit += np.where(np.array(cost_burdens) == 'High', -0.70,
             np.where(np.array(cost_burdens) == 'Moderate', -0.20, 0.35))
    logit += np.where(np.array(insurance_coverages) == 'None', -0.65,
             np.where(np.array(insurance_coverages) == 'Full', 0.35, 0.0))
    logit += np.where(np.array(pharmacy_access) == 'Difficult', -0.45,
             np.where(np.array(pharmacy_access) == 'Easy', 0.25, 0.0))
    logit += -0.04 * (distances - 4.0)
    
    # Support & Provider interaction
    logit += np.where(np.array(family_supports) == 'High', 0.35,
             np.where(np.array(family_supports) == 'Low', -0.45, 0.0))
    logit += np.where(np.array(healthcare_followups) == 'Regular', 0.40,
             np.where(np.array(healthcare_followups) == 'Rare', -0.50, 0.0))
    logit += 0.12 * (doctor_communications - 6.5)
    
    # Demographic nuances (age-related curve: slight positive mid-life, slight dip for very young or fragile elderly with polypharmacy)
    logit += 0.015 * (ages - 50) - 0.0004 * ((ages - 50) ** 2)
    
    # Add stochastic residual noise (human behavioral stochasticity)
    noise = np.random.normal(loc=0.0, scale=0.85, size=n_samples)
    final_logit = logit + noise
    
    probabilities = 1.0 / (1.0 + np.exp(-final_logit))
    # Threshold at 0.50
    adherence = (probabilities >= 0.50).astype(int)
    
    # Construct DataFrame
    df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': ages,
        'Gender': genders,
        'Education_Level': education_levels,
        'Employment_Status': employment_statuses,
        'Medication_Type': medication_types,
        'Medication_Frequency': medication_frequencies,
        'Number_of_Medications': num_meds,
        'Treatment_Duration_Months': treatment_durations,
        'Dosage_Complexity': dosage_complexities,
        'Previous_Missed_Doses': previous_missed,
        'Forgetfulness_Score': forgetfulness_scores,
        'Routine_Consistency': routine_consistencies,
        'Medication_Reminder_Usage': reminder_usage,
        'Chronic_Condition': conditions,
        'Side_Effects': side_effects,
        'Perceived_Effectiveness': perceived_effectiveness,
        'Treatment_Satisfaction': treatment_satisfactions,
        'Medication_Cost_Burden': cost_burdens,
        'Access_to_Pharmacy': pharmacy_access,
        'Insurance_Coverage': insurance_coverages,
        'Distance_to_Pharmacy_Km': distances,
        'Family_Support': family_supports,
        'Healthcare_Followup': healthcare_followups,
        'Doctor_Communication': doctor_communications,
        'Medication_Adherence': adherence
    })
    
    # Introduce small realistic missingness (~2-3%) in 3 non-target columns to allow genuine cleaning and imputation demonstration
    # (e.g. Distance_to_Pharmacy_Km, Doctor_Communication, Treatment_Satisfaction)
    missing_idx_dist = np.random.choice(n_samples, size=int(0.025 * n_samples), replace=False)
    df.loc[missing_idx_dist, 'Distance_to_Pharmacy_Km'] = np.nan
    
    missing_idx_doc = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    df.loc[missing_idx_doc, 'Doctor_Communication'] = np.nan
    
    missing_idx_sat = np.random.choice(n_samples, size=int(0.015 * n_samples), replace=False)
    df.loc[missing_idx_sat, 'Treatment_Satisfaction'] = np.nan

    # Add 5 duplicate rows to test duplicate handling during EDA & cleaning
    dup_rows = df.iloc[:5].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    return df

if __name__ == '__main__':
    output_path = os.path.join(os.path.dirname(__file__), 'raw', 'medication_adherence.csv')
    df = generate_medication_adherence_data()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records (including simulated duplicates and nulls).")
    print(f"Saved to: {output_path}")
    print(f"Target distribution:\n{df['Medication_Adherence'].value_counts(normalize=True)}")
