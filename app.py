from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "sms_model.pkl"

st.set_page_config(
    page_title="SMS Fraud Detection",
    page_icon="📩",
    layout="centered"
)

st.markdown("""
<style>
    .block-container {max-width: 850px; padding-top: 2.2rem;}
    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #e0f2fe, #eef2ff);
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .hero h1 {color: #172554; margin-bottom: .35rem;}
    .hero p {color: #475569; margin: 0;}
    .small-note {color: #64748b; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📩 SMS Fraud Detection</h1>
    <p>Check whether a text message resembles spam or a normal SMS.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.warning(
        "The trained model has not been created yet. Follow these steps:\n\n"
        "1. Put the UCI dataset file `SMSSpamCollection` inside the `dataset` folder.\n"
        "2. Run `python train_model.py` in the project terminal.\n"
        "3. Refresh this page."
    )
    st.stop()

model = load_model()

message = st.text_area(
    "Enter or paste an SMS message",
    placeholder="Example: Hey, are we meeting at college tomorrow?",
    height=150,
    max_chars=5000
)

if st.button("Analyse SMS", type="primary", use_container_width=True):
    if not message.strip():
        st.info("Please enter a message first.")
    else:
        prediction = int(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index]) * 100

        st.divider()
        if prediction == 1:
            st.error("🚨 This message is classified as SPAM")
            st.write(
                "It resembles spam messages in the training dataset. "
                "Do not click unexpected links or share passwords, PINs, or OTPs."
            )
        else:
            st.success("✅ This message is classified as NORMAL")
            st.write(
                "It resembles normal messages in the training dataset. "
                "This result does not guarantee that the message is safe."
            )

        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption(
            "Confidence is the model's estimated class probability, not a "
            "verified fraud score. The model can make mistakes."
        )

with st.expander("Try sample messages"):
    st.write("**Normal example:** Hey, are you coming to class tomorrow?")
    st.write("**Spam-like example:** Congratulations! You have won a prize. Click this link now.")

st.divider()
