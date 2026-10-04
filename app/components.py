"""
MedAdhere AI - UI Components & Visualization Module
===================================================
Contains modular UI widgets, KPI cards, Plotly charts,
and demographic / clinical preset demo profiles.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def load_css(css_path):
    """Inject custom CSS stylesheet."""
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load custom CSS: {e}")

def render_header(title="MedAdhere AI", subtitle="Medication Adherence Risk Analysis"):
    """Render the top hero banner with healthcare branding and badges."""
    st.markdown(f"""
    <div class="app-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <div class="header-badges">
            <span class="header-badge">🏥 Clinical Decision Support Prototype</span>
            <span class="header-badge">🌲 Tuned Random Forest Classifier</span>
            <span class="header-badge">📊 WHO 5-Dimensions Adherence Grounding</span>
            <span class="header-badge">🎓 B.Tech CSE Machine Learning Case Study</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards(accuracy=0.8350, f1=0.8879, roc_auc=0.9046, n_samples=2400):
    """Render top performance metrics in responsive card grid."""
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Model Accuracy</div>
            <div class="kpi-value">{accuracy:.1%}</div>
            <div class="kpi-sub">Overall test set correctness</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">F1-Score (Primary)</div>
            <div class="kpi-value">{f1:.3f}</div>
            <div class="kpi-sub">Harmonic mean of precision & recall</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">ROC-AUC Score</div>
            <div class="kpi-value">{roc_auc:.3f}</div>
            <div class="kpi-sub">Discrimination across thresholds</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Training Cohort</div>
            <div class="kpi-value">{n_samples:,}</div>
            <div class="kpi-sub">80/20 Stratified Train/Test split</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_gauge_chart(probability):
    """
    Renders an interactive, medical-grade Plotly gauge chart
    displaying predicted adherence probability and risk boundaries.
    """
    pct = probability * 100
    
    if pct >= 70:
        bar_color = "#10b981"  # Emerald
        risk_label = "Low Non-Adherence Risk"
    elif pct >= 40:
        bar_color = "#f59e0b"  # Amber
        risk_label = "Moderate Non-Adherence Risk"
    else:
        bar_color = "#ef4444"  # Rose
        risk_label = "High Non-Adherence Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 44, 'family': 'Plus Jakarta Sans', 'color': '#0f172a'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickvals': [0, 25, 40, 50, 70, 85, 100]},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},     # High risk zone
                {'range': [40, 70], 'color': '#fef3c7'},    # Moderate risk zone
                {'range': [70, 100], 'color': '#d1fae5'}    # Low risk zone
            ],
            'threshold': {
                'line': {'color': "#0f172a", 'width': 3},
                'thickness': 0.8,
                'value': pct
            }
        }
    ))

    fig.update_layout(
        height=240,
        margin=dict(l=25, r=25, t=25, b=15),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Plus Jakarta Sans"}
    )
    return fig

def render_result_banner(is_adherent, probability):
    """Renders prominent adherence status banner with probability and risk badge."""
    pct = probability * 100
    
    if is_adherent:
        status_text = "ADHERENT"
        banner_class = "result-adherent"
        icon = "✅"
    else:
        status_text = "NON-ADHERENT"
        banner_class = "result-nonadherent"
        icon = "⚠️"

    if pct >= 70:
        risk_tag = '<span class="risk-tag low">🟢 Low Risk of Non-Adherence</span>'
        risk_desc = "Patient profile indicates strong behavioral consistency and manageable regimen burden."
    elif pct >= 40:
        risk_tag = '<span class="risk-tag moderate">🟡 Moderate Risk of Non-Adherence</span>'
        risk_desc = "Patient exhibits mixed adherence signals (e.g. moderate complexity, occasional missed doses, or cost pressure)."
    else:
        risk_tag = '<span class="risk-tag high">🔴 High Risk of Non-Adherence</span>'
        risk_desc = "Patient profile exhibits multiple severe barriers (frequent missed doses, high forgetfulness, high polypharmacy, or severe side effects)."

    st.markdown(f"""
    <div class="result-banner {banner_class}">
        <div class="result-header-text">Model Predicted Compliance Outcome</div>
        <div class="result-main-status">{icon} Predicted Status: {status_text}</div>
        <div style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.8rem;">
            Estimated Adherence Probability: <span style="font-size: 1.35rem; font-weight: 800;">{pct:.1f}%</span>
        </div>
        <div style="margin-bottom: 0.6rem;">{risk_tag}</div>
        <div style="font-size: 0.88rem; opacity: 0.9; margin-top: 0.4rem;">{risk_desc}</div>
    </div>
    """, unsafe_allow_html=True)

def render_insights(patient_dict, probability):
    """
    Evaluates key input variables and renders actionable, non-prescriptive
    behavioral drivers for the patient.
    """
    st.markdown("### 🔍 Patient Behavioral & Clinical Insights")
    st.markdown("""
    The model identified the following notable factors in the submitted patient profile. 
    *These factors represent statistical associations from the training data and do not constitute direct medical causation.*
    """)

    factors = []

    # Frequency factor
    freq = patient_dict.get('Medication_Frequency', 1)
    if freq >= 3:
        factors.append(('negative', f'High Dosing Frequency ({freq}x daily)', 'Multi-dose daily schedules significantly increase friction and missed doses.'))
    elif freq == 1:
        factors.append(('positive', 'Once-Daily Regimen', 'Single daily dosing is strongly associated with sustained habit formation.'))

    # Missed doses factor
    missed = patient_dict.get('Previous_Missed_Doses', 0)
    if missed >= 3:
        factors.append(('negative', f'Prior Missed Doses ({missed} in 30 days)', 'Past compliance pattern is the strongest predictive indicator of ongoing non-adherence.'))
    elif missed == 0:
        factors.append(('positive', 'Zero Reported Missed Doses', 'History of perfect recent compliance demonstrates stable routine.'))

    # Reminder usage
    rem = patient_dict.get('Medication_Reminder_Usage', 'No')
    if rem == 'Yes':
        factors.append(('positive', 'Active Reminders Used', 'Digital alarms or pill organizers mitigate cognitive forgetfulness.'))
    else:
        factors.append(('negative', 'No Reminder Tools Used', 'Reliance on unassisted recall increases vulnerability to daily schedule disruptions.'))

    # Side effects
    se = patient_dict.get('Side_Effects', 'None')
    if se in ['Severe', 'Moderate']:
        factors.append(('negative', f'{se} Side Effects', 'Adverse drug events are a primary psychological driver of intentional omission.'))
    elif se == 'None':
        factors.append(('positive', 'No Reported Side Effects', 'Good drug tolerability supports long-term regimen continuation.'))

    # Cost burden
    cost = patient_dict.get('Medication_Cost_Burden', 'Low')
    if cost == 'High':
        factors.append(('negative', 'High Medication Cost Burden', 'Out-of-pocket expenses frequently lead patients to ration or skip refills.'))
    elif cost == 'Low':
        factors.append(('positive', 'Low Medication Cost Burden', 'Financial accessibility reduces structural barriers to medication refills.'))

    # Routine consistency
    routine = patient_dict.get('Routine_Consistency', 5)
    if routine <= 3:
        factors.append(('negative', f'Irregular Routine ({routine}/10)', 'Unpredictable lifestyle schedules make routine dosing difficult.'))
    elif routine >= 8:
        factors.append(('positive', f'Highly Structured Routine ({routine}/10)', 'Consistent daily habits strongly reinforce medication adherence.'))

    # Doctor communication
    doc = patient_dict.get('Doctor_Communication', 5)
    if doc >= 8:
        factors.append(('positive', f'Strong Provider Communication ({doc}/10)', 'High trust and open dialogue with healthcare providers correlate with better compliance.'))
    elif doc <= 3:
        factors.append(('neutral', f'Low Provider Communication ({doc}/10)', 'Limited communication may leave patient questions and concerns unaddressed.'))

    # Render factor pills
    for tone, title, desc in factors:
        st.markdown(f"""
        <div style="margin-bottom: 0.6rem; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #e2e8f0; background: {'#fef2f2' if tone=='negative' else '#ecfdf5' if tone=='positive' else '#f8fafc'};">
            <span class="factor-pill {tone}"><b>{'⚠️ Risk Factor' if tone=='negative' else '✅ Protective Factor' if tone=='positive' else 'ℹ️ Context Factor'}</b>: {title}</span>
            <div style="font-size: 0.85rem; color: #475569; margin-top: 0.25rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

def render_disclaimer():
    """Renders mandatory educational and non-diagnostic clinical disclaimer."""
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Academic & Healthcare Disclaimer:</strong><br>
        <strong>MedAdhere AI</strong> is an academic, educational machine-learning demonstration developed as part of a 
        B.Tech Computer Science & Engineering case study. It is calibrated on synthetic demographic and behavioral 
        distributions to illustrate predictive analytics principles. It is <strong>NOT</strong> an approved medical diagnostic 
        device and must <strong>NEVER</strong> be used to alter, initiate, or discontinue prescribed medications without direct 
        consultation and clinical authorization from a licensed healthcare provider.
    </div>
    """, unsafe_allow_html=True)

# Preset Demo Profiles
DEMO_PATIENT_1_ADHERENT = {
    'Age': 58,
    'Gender': 'Female',
    'Education_Level': 'Bachelor',
    'Employment_Status': 'Employed',
    'Medication_Type': 'Antihypertensive',
    'Medication_Frequency': 1,
    'Number_of_Medications': 3,
    'Treatment_Duration_Months': 36,
    'Dosage_Complexity': 'Low',
    'Previous_Missed_Doses': 0,
    'Forgetfulness_Score': 2,
    'Routine_Consistency': 9,
    'Medication_Reminder_Usage': 'Yes',
    'Chronic_Condition': 'Hypertension',
    'Side_Effects': 'None',
    'Perceived_Effectiveness': 'High',
    'Treatment_Satisfaction': 9,
    'Medication_Cost_Burden': 'Low',
    'Access_to_Pharmacy': 'Easy',
    'Insurance_Coverage': 'Full',
    'Distance_to_Pharmacy_Km': 2.4,
    'Family_Support': 'High',
    'Healthcare_Followup': 'Regular',
    'Doctor_Communication': 9
}

DEMO_PATIENT_2_NON_ADHERENT = {
    'Age': 44,
    'Gender': 'Male',
    'Education_Level': 'High School',
    'Employment_Status': 'Unemployed',
    'Medication_Type': 'Psychiatric',
    'Medication_Frequency': 4,
    'Number_of_Medications': 7,
    'Treatment_Duration_Months': 8,
    'Dosage_Complexity': 'High',
    'Previous_Missed_Doses': 7,
    'Forgetfulness_Score': 9,
    'Routine_Consistency': 2,
    'Medication_Reminder_Usage': 'No',
    'Chronic_Condition': 'Depression/Anxiety',
    'Side_Effects': 'Severe',
    'Perceived_Effectiveness': 'Low',
    'Treatment_Satisfaction': 2,
    'Medication_Cost_Burden': 'High',
    'Access_to_Pharmacy': 'Difficult',
    'Insurance_Coverage': 'None',
    'Distance_to_Pharmacy_Km': 18.0,
    'Family_Support': 'Low',
    'Healthcare_Followup': 'Rare',
    'Doctor_Communication': 3
}
