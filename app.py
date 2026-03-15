import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. PAGE CONFIG & CSS INJECTION ---
st.set_page_config(page_title="CORE-Vitals AI", layout="wide")

def inject_cyber_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        /* Background & Typography */
        .stApp {
            background: linear-gradient(135deg, #0b0b1a 0%, #1a1a2e 100%);
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }

        /* Hide default header & reduce padding */
        header {visibility: hidden;}
        .main .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}

        /* Bento Card Styling for columns */
        div[data-testid="column"] {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(15px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 35px !important;
            box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.6);
        }

        /* Neon Labels */
        label {
            color: #00f2ff !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.8rem !important;
        }

        /* Gradient Predict Button */
        .stButton>button {
            background: linear-gradient(90deg, #00f2ff, #ff00ff);
            color: white;
            border: none;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-weight: 800;
            transition: 0.4s ease;
            box-shadow: 0 4px 15px rgba(0, 242, 255, 0.3);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 0, 255, 0.5);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

inject_cyber_styles()

# --- 2. LOAD MODEL ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("KNN_heart.pkl")   # matches actual filename on disk
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_assets()
except Exception as e:
    st.error(f"Asset Load Error: {e}. Ensure KNN_heart.pkl, scaler.pkl, and columns.pkl are present.")
    st.stop()

# --- 3. HEADER ---
st.title("HEART STROKE PREDICTION")
st.caption("AI Diagnostic Engine | Developed by Rohit More")
st.markdown("<br>", unsafe_allow_html=True)

# --- 4. BENTO GRID LAYOUT ---
col_input, col_viz = st.columns([0.45, 0.55], gap="large")

with col_input:
    st.subheader("🧬 Patient Parameters")

    tab_vitals, tab_clinical, tab_lifestyle = st.tabs(["Vitals", "Clinical Tests", "Lifestyle"])

    with tab_vitals:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        Cholesterol = st.number_input("Cholesterol (mg/dL)", 80, 603, 200)

    with tab_clinical:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.2, 1.0)

    with tab_lifestyle:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ EXECUTE DIAGNOSTIC INFERENCE")

# --- 5. VISUALIZATION PANEL ---
with col_viz:
    if predict_btn:
        # Build the one-hot encoded input row
        raw_input = {
            'Age': [age],
            'RestingBP': [resting_bp],
            'Cholesterol': [Cholesterol],
            'FastingBS': [fasting_bs],
            'MaxHR': [max_hr],
            'Oldpeak': [oldpeak],
            f'Sex_{sex}': [1],
            f'ChestPainType_{chest_pain}': [1],
            f'RestingECG_{resting_ecg}': [1],
            f'ExerciseAngina_{exercise_angina}': [1],
            f'ST_Slope_{st_slope}': [1]
        }

        df_input = pd.DataFrame(raw_input)

        # Fill any missing columns the model expects with 0
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        # Ensure column order matches training
        df_input = df_input[expected_columns]

        try:
            scaled_input = scaler.transform(df_input)
            pred = model.predict(scaled_input)[0]

            # Risk probability
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(scaled_input)[0][1]
            else:
                prob = 0.92 if pred == 1 else 0.08

            # Gauge chart
            bar_color = "#ff00ff" if pred == 1 else "#00ff88"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={'suffix': "%", 'font': {'color': bar_color, 'size': 36}},
                title={'text': "Cardiac Risk Level", 'font': {'color': "#00f2ff", 'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#00f2ff"},
                    'bar': {'color': bar_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 50],  'color': "rgba(0, 255, 136, 0.08)"},
                        {'range': [50, 100], 'color': "rgba(255, 0, 255, 0.08)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
                margin=dict(t=60, b=20, l=20, r=20),
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)

            # Result banner
            if pred == 1:
                st.markdown(
                    f"<div style='padding:20px; border-radius:15px; "
                    f"background:rgba(255,0,0,0.1); border:1px solid #ff4b4b; color:#ff4b4b;'>"
                    f"🚨 <b>CRITICAL WARNING:</b> High cardiac risk detected &nbsp;|&nbsp; "
                    f"Risk Score: <b>{prob:.2%}</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='padding:20px; border-radius:15px; "
                    f"background:rgba(0,255,136,0.1); border:1px solid #00ff88; color:#00ff88;'>"
                    f"✅ <b>STABLE:</b> Low cardiac risk profile &nbsp;|&nbsp; "
                    f"Confidence: <b>{1 - prob:.2%}</b></div>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Inference Error: {e}")

    else:
        # Ready-state placeholder
        st.markdown(
            "<div style='margin-top:100px; text-align:center;'>"
            "<p style='color:#00f2ff; font-size:1.4rem; font-weight:800; letter-spacing:3px;'>◈ SYSTEM READY ◈</p>"
            "<p style='color:#444; font-size:0.9rem; margin-top:10px;'>Enter patient parameters and click<br>EXECUTE DIAGNOSTIC INFERENCE</p>"
            "</div>",
            unsafe_allow_html=True
        )

# --- 6. FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem; letter-spacing:2px;'>"
    "BIOMEDICAL AI UNIT &nbsp;•&nbsp; SECURE DIAGNOSTIC INTERFACE &nbsp;•&nbsp; ROHIT MORE</p>",
    unsafe_allow_html=True
)
