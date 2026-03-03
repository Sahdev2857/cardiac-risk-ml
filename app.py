import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import tempfile
from datetime import datetime

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="heart",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee0979);
        color: white; padding: 24px; border-radius: 16px; text-align: center;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white; padding: 24px; border-radius: 16px; text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white; padding: 24px; border-radius: 16px; text-align: center;
    }
    .suggestion-card {
        background: white;
        border-left: 4px solid #ee0979;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stButton > button {
        background: linear-gradient(135deg, #ee0979, #ff6b6b);
        color: white; border: none; border-radius: 10px;
        padding: 12px 24px; font-weight: 600; font-size: 16px; width: 100%;
    }
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #4a90e2;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Train model on Cleveland data
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("cleveland_heart.csv")
    df.dropna(inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns.tolist()

# ─────────────────────────────────────────────
# Suggestions
# ─────────────────────────────────────────────
def get_suggestions(age, chol, trestbps, fbs, exang, cp, thalach):
    suggestions = []
    if chol > 200:
        suggestions.append(f"Your cholesterol ({chol} mg/dL) is high. Reduce saturated fats, fried foods, and red meat.")
    if trestbps > 130:
        suggestions.append(f"Your resting BP ({trestbps} mmHg) is elevated. Reduce salt intake and consult a doctor.")
    if fbs == 1:
        suggestions.append("Your fasting blood sugar is above 120 mg/dL — a diabetes risk factor. Monitor sugar intake closely.")
    if exang == 1:
        suggestions.append("Exercise-induced chest pain is a serious warning sign. Get a stress test done by a cardiologist.")
    if cp in [1, 2, 3]:
        suggestions.append("You reported chest pain symptoms. This should be evaluated by a doctor promptly.")
    if thalach < 120:
        suggestions.append(f"Your max heart rate ({thalach} bpm) is low for your age. Regular cardio exercise can help improve this.")
    if age > 55:
        suggestions.append("Age is a non-modifiable risk factor. Focus on controlling other factors like BP, cholesterol, and exercise.")
    if not suggestions:
        suggestions.append("Your indicators look good! Keep maintaining a heart-healthy lifestyle with regular exercise and balanced diet.")
    return suggestions



# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("Heart Disease Risk Predictor")
st.markdown("##### Trained on the UCI Cleveland Heart Disease Dataset — Real Clinical Data")

st.markdown("""
<div class="info-box">
    This model is trained on the <b>UCI Cleveland Heart Disease Dataset</b> — actual patient records 
    collected at the Cleveland Clinic Foundation. Used in hundreds of peer-reviewed research papers.
</div>
""", unsafe_allow_html=True)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    age     = st.slider("Age", 20, 80, 50)
    sex     = st.selectbox("Sex", ["Male", "Female"])
    cp      = st.selectbox("Chest Pain Type", [
        "0 - Typical Angina",
        "1 - Atypical Angina",
        "2 - Non-Anginal Pain",
        "3 - Asymptomatic"
    ])
    cp_val  = int(cp[0])

with col2:
    st.markdown("**Clinical Measurements**")
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol     = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
    fbs_val  = 1 if fbs == "Yes" else 0
    restecg  = st.selectbox("Resting ECG Results", [
        "0 - Normal",
        "1 - ST-T Wave Abnormality",
        "2 - Left Ventricular Hypertrophy"
    ])
    restecg_val = int(restecg[0])

with col3:
    st.markdown("**Exercise & Other**")
    thalach  = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang    = st.selectbox("Exercise Induced Chest Pain?", ["No", "Yes"])
    exang_val = 1 if exang == "Yes" else 0
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope    = st.selectbox("Slope of Peak ST Segment", [
        "0 - Upsloping",
        "1 - Flat",
        "2 - Downsloping"
    ])
    slope_val = int(slope[0])
    ca       = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", [
        "1 - Normal",
        "2 - Fixed Defect",
        "3 - Reversible Defect"
    ])
    thal_val = int(thal[0])

st.divider()

if st.button("Analyse My Heart Risk"):
    try:
        model, explainer, feature_cols = train_model()

        sex_val = 1 if sex == "Male" else 0

        input_data = pd.DataFrame([{
            'age': age, 'sex': sex_val, 'cp': cp_val,
            'trestbps': trestbps, 'chol': chol, 'fbs': fbs_val,
            'restecg': restecg_val, 'thalach': thalach, 'exang': exang_val,
            'oldpeak': oldpeak, 'slope': slope_val, 'ca': ca, 'thal': thal_val
        }])
        input_data = input_data.reindex(columns=feature_cols, fill_value=0)

        prob = model.predict_proba(input_data)[0][1]
        risk_label = "HIGH RISK" if prob >= 0.6 else "MODERATE RISK" if prob >= 0.35 else "LOW RISK"
        css_class  = "risk-high" if prob >= 0.6 else "risk-moderate" if prob >= 0.35 else "risk-low"

        st.markdown(f"""
        <div class="{css_class}">
            <h2 style="margin:0">{risk_label}</h2>
            <h3 style="margin:4px 0">{prob:.1%} probability of heart disease</h3>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob))
        st.markdown("")

        tab1, tab2 = st.tabs(["Why this result? (SHAP)", "What should I do?"])

        shap_img_path = None

        with tab1:
            st.markdown("#### Which factors are driving your risk?")
            st.caption("Red bars push risk UP. Blue bars push risk DOWN.")
            with st.spinner("Calculating explanation..."):
                shap_values = explainer.shap_values(input_data)
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                    sv = shap_values[0, :, 1]
                else:
                    sv = shap_values[0]

                feature_labels = {
                    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type',
                    'trestbps': 'Resting BP', 'chol': 'Cholesterol',
                    'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
                    'thalach': 'Max Heart Rate', 'exang': 'Exercise Chest Pain',
                    'oldpeak': 'ST Depression', 'slope': 'ST Slope',
                    'ca': 'Vessels Colored', 'thal': 'Thalassemia'
                }

                shap_df = pd.DataFrame({
                    'Feature': [feature_labels.get(f, f) for f in feature_cols],
                    'SHAP Value': sv
                }).sort_values('SHAP Value', key=abs, ascending=True)

                colors = ['#ee0979' if v > 0 else '#56ab2f' for v in shap_df['SHAP Value']]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_xlabel("SHAP Value (impact on prediction)")
                ax.set_title("Factors Influencing Your Risk", fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()

                shap_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                plt.savefig(shap_img_path, dpi=150, bbox_inches='tight')
                st.pyplot(fig)
                plt.close()

        with tab2:
            st.markdown("#### Personalised Health Recommendations")
            suggestions = get_suggestions(age, chol, trestbps, fbs_val, exang_val, cp_val, thalach)
            for s in suggestions:
                st.markdown(f'<div class="suggestion-card">{s}</div>', unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("cleveland_heart.csv not found. Make sure it's in the same folder as app.py")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    
        st.divider()
st.caption("⚠️ Disclaimer: This is a student ML project for educational purposes only. Not medically certified. Always consult a qualified doctor for health advice.")
