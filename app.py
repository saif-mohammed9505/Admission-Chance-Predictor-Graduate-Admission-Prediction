import base64
import os
import pickle

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------- Setup ----------

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "admission.pkl")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

st.set_page_config(
    page_title="Admission Chance Predictor",
    page_icon="🎓",
    layout="centered",
)


def svg_data_uri(filename):
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


model, load_error = load_model()

# ---------- Styling ----------

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(160deg, #4f7cff 0%, #7c5cff 100%);
        border-radius: 20px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 1.7rem; }
    .hero p { margin: 0; opacity: 0.92; }
    .feature-row {
        display: flex;
        gap: 10px;
        margin-top: 16px;
        flex-wrap: wrap;
    }
    .feature-pill {
        background: rgba(255,255,255,0.15);
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .feature-pill img { width: 16px; height: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero ----------

st.markdown(
    f"""
    <div class="hero">
      <img src="{svg_data_uri('university.svg')}" width="120" style="display:block;margin:0 auto 12px auto;">
      <h1>🎓 Admission Chance Predictor</h1>
      <p>Estimate your chance of graduate admission from GRE, TOEFL, CGPA, SOP, LOR and research experience.</p>
      <div class="feature-row">
        <div class="feature-pill"><img src="{svg_data_uri('book.svg')}"> Trained on admission records</div>
        <div class="feature-pill"><img src="{svg_data_uri('chart.svg')}"> Linear regression model</div>
        <div class="feature-pill"><img src="{svg_data_uri('target.svg')}"> Instant results</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.warning(
        f"**admission.pkl not found or could not be loaded.**\n\n"
        f"Place your trained model file named `admission.pkl` in this app's folder "
        f"(next to `app.py`) and restart the app.\n\nDetails: {load_error}"
    )

# ---------- Form ----------

col1, col2 = st.columns(2)

with col1:
    gre = st.number_input("GRE Score (out of 340)", min_value=0, max_value=340, value=320, step=1)
    university_rating = st.slider("University Rating (1–5)", 1, 5, 3)
    lor = st.slider("Letter of Recommendation - LOR (1–5)", 1.0, 5.0, 3.5, step=0.5)
    research = st.checkbox("Research Experience")

with col2:
    toefl = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=110, step=1)
    sop = st.slider("Statement of Purpose - SOP (1–5)", 1.0, 5.0, 3.5, step=0.5)
    cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5, step=0.01)

predict_clicked = st.button("Predict My Chances", type="primary", use_container_width=True)

# ---------- Prediction ----------

if predict_clicked:
    if model is None:
        st.error("Can't predict — admission.pkl isn't loaded. See the warning above.")
    else:
        features = np.array([[gre, toefl, university_rating, sop, lor, cgpa, 1 if research else 0]])
        try:
            raw_pred = float(model.predict(features)[0])
            chance = max(0.0, min(1.0, raw_pred)) * 100

            if chance >= 75:
                color = "#22b573"
                message = "Strong chance of admission — your profile looks competitive!"
            elif chance >= 50:
                color = "#f2a93b"
                message = "Moderate chance — a well-written SOP and strong LORs could tip things in your favor."
            else:
                color = "#e35b5b"
                message = "Consider strengthening your profile — a higher CGPA, GRE score or research experience can help."

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=chance,
                    number={"suffix": "%", "font": {"size": 40}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "bgcolor": "#e6e9f0",
                        "steps": [
                            {"range": [0, 50], "color": "#fdecec"},
                            {"range": [50, 75], "color": "#fdf3e2"},
                            {"range": [75, 100], "color": "#e8f8f0"},
                        ],
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<p style='text-align:center;color:#6b7280;'>{message}</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Predictions are estimates from a statistical model, not a guarantee of admission.")
