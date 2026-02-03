import streamlit as st
import joblib
import numpy as np
import time

# --- WINDOW CONFIG ---
st.set_page_config(page_title="Vegas Edge | Scanner", layout="centered")

# --- CUSTOM CSS (CREATIVE ELEMENTS) ---
st.markdown("""
    <style>
    /* 1. Futuristic Background */
    .stApp {
        background: radial-gradient(circle at center, #1a1a2e 0%, #050505 100%);
        color: #ffffff;
    }

    /* 2. Glassmorphism Card for Inputs */
    [data-testid="stVerticalBlock"] > div:nth-child(2) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    /* 3. Animated Scanning Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ffd700, #b8860b);
        color: black !important;
        font-weight: 900 !important;
        font-size: 20px !important;
        border: none !important;
        border-radius: 50px !important;
        height: 3.5em !important;
        transition: 0.5s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px #ffd700;
        transform: scale(1.02);
    }

    /* 4. Custom Styling for Sliders and Labels */
    .stSlider label, .stSelectbox label, .stRadio label {
        color: #ffd700 !important;
        font-weight: bold !important;
        letter-spacing: 1px;
    }
    
    /* 5. Result Box Glow Effect */
    .result-card {
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        margin-top: 30px;
        border-width: 2px;
        border-style: solid;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(255, 215, 0, 0.2); }
        50% { box-shadow: 0 0 30px rgba(255, 215, 0, 0.5); }
        100% { box-shadow: 0 0 10px rgba(255, 215, 0, 0.2); }
    }
    </style>
    """, unsafe_allow_html=True)

# Load AI
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --- HEADER WITH ICON ---
st.markdown("<h1 style='text-align: center; color: #ffd700; font-size: 50px;'>🎰 VEGAS EDGE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-style: italic;'>Surveillance Terminal Alpha-V4</p>", unsafe_allow_html=True)
st.write("---")

# --- INPUT SECTION ---
st.markdown("### 📡 FLOOR TELEMETRY")
col1, col2 = st.columns(2)

with col1:
    vol = st.selectbox("Betting Pattern", [0, 1, 2], 
                       format_func=lambda x: ["Steady (Same bets)", "Changing (Normal)", "Wild (Very Erratic)"][x])
    win = st.slider("How much are they winning?", 0, 100, 50)
    size = st.number_input("Average Bet Amount ($)", 5, 10000, 100)

with col2:
    drink = st.radio("Is the player drinking alcohol?", [1, 0], 
                     format_func=lambda x: "🥃 Yes, drinking" if x==1 else "💧 No, sober")
    dur = st.number_input("Time at table (Minutes)", 5, 600, 30)

st.markdown("<br>", unsafe_allow_html=True)

# --- ONE-CLICK ACTION ---
if st.button("RUN SCANNER"):
    # Creative Scanning Animation
    progress_text = "INITIALIZING BIOMETRIC SCAN..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.5)
    my_bar.empty()
    
    # Predict
    raw = np.array([[vol, win, size, drink, dur]])
    scaled = scaler.transform(raw)
    prediction = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]
    confidence = np.max(probs) * 100

    # Logic for status
    if prediction == 0:
        border = "#00ff96"
        status = "Standard Customer"
        advice = "No action needed. Regular player behavior."
        icon = "🟢"
    elif prediction == 1:
        border = "#ffd700"
        status = "VIP High-Roller"
        advice = "Treat well! Offer a free drink or room upgrade."
        icon = "💎"
    else:
        border = "#ff4b4b"
        status = "Suspicious Behavior"
        advice = "Alert Security. Pattern matches potential card-counting."
        icon = "🚨"

    # --- RESULTS DISPLAY WITH GRAPHIC FEEL ---
    st.markdown(f"""
        <div class="result-card" style="background: rgba(0,0,0,0.4); border-color: {border};">
            <h2 style="color: {border}; margin:0;">{icon} {status}</h2>
            <hr style="border: 0.1px solid {border}; opacity: 0.2;">
            <p style="font-size: 20px; color: white;"><b>Recommended Protocol:</b><br>{advice}</p>
            <p style="font-size: 14px; color: grey;">AI CERTAINTY: {confidence:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

st.write("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px; color: #444;'>AI TRAINING SET NASED ON NAIVE BAYES ALGORITHM</p>", unsafe_allow_html=True)