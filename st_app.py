import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from PIL import Image
import cv2
import seaborn as sns 

# NEW BACKEND IMPORTS
from project_backend import (
    EmotionCNN, 
    generate_session,
    DataHandler,
    EMOTIONS, 
    INSTRUMENTS,
    RAGAS
)

# Set page config (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Advanced Emotion Music Generator (CNN + TRPO)",
    page_icon="🧠",
    layout="wide"
)

# Initialize Components
# Removed cache to ensure strict model reloading
def get_components():
    cnn = EmotionCNN()
    # Path to user CSV
    data_handler = DataHandler("final2.O_merged_instrument_dataset(2054 audios).csv")
    data_handler.load_data()
    return cnn, data_handler

cnn_model, data_handler = get_components()
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #6C5CE7; text-align: center; font-weight: bold; }
    .sub-header { font-size: 1.2rem; color: #444; text-align: center; margin-bottom: 2rem; }
    .stButton>button { width: 100%; background-color: #6C5CE7; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Cortex Music Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Keras CNN & TRPO Reinforcement Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    input_mode = st.radio("Input Mode", ["Upload Image", "Manual Selection"])
    
    detected_emotion = None
    
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload Face Image", type=['jpg', 'png'])
        if uploaded_file:
            # Match User's app.py logic: Read as bytes -> OpenCV (BGR)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1) # BGR Format
            
            # Display requires RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption='Input Image', use_column_width=True)
            
            # Pass BGR to backend (it expects BGR for cvtColor to Gray)
            with st.spinner('CNN Model Predicting...'):
                detected_emotion = cnn_model.predict(img)
                st.success(f"CNN Detected: **{detected_emotion.upper()}**")
    
    st.divider()
    
    # Selection Logic
    if input_mode == "Upload Image" and detected_emotion:
        st.write("🔒 **Emotion Locked from CNN**")
        # Just show it as a static value
        st.warning(f"Using Emotion: **{detected_emotion.upper()}**")
        selected_emotion = detected_emotion
    else:
        # Manual Mode
        selected_emotion = st.selectbox("Select Emotion", EMOTIONS)
        
    selected_instrument = st.selectbox("Select Instrument", INSTRUMENTS)
    
    duration = st.slider("Duration (seconds)", 5, 30, 10)

# Main Area
output_container = st.container()
col1, col2 = st.columns([1, 2])

with col1:
    st.info("Click to run the TRPO Agent.")
    if st.button("🚀 Generate with AI", type="primary"):
        with st.spinner(f"TRPO Agent generating {selected_instrument} melody..."):
            
            # CALL NEW BACKEND
            audio_data, rate, policy = generate_session(
                selected_emotion, 
                selected_instrument, 
                data_handler, 
                duration
            )
            
            st.session_state['audio'] = audio_data
            st.session_state['rate'] = rate
            st.session_state['policy'] = policy # To visualize agent brain
            st.session_state['meta'] = f"{selected_emotion} - {selected_instrument}"
            
            st.success("Synthesis Complete!")

if 'audio' in st.session_state:
    audio = st.session_state['audio']
    rate = st.session_state['rate']
    
    with output_container: # Warning: Variable output_container might not be defined in new scope, verify
        # Actually col2 is better
        pass

    with col2:
        st.subheader(f"🎶 Result: {st.session_state['meta']}")
        st.audio(audio, sample_rate=rate)
        
        # Download
        # Normalize int16
        scaled = np.int16(audio * 32767)
        wavfile.write("temp_out.wav", rate, scaled)
        with open("temp_out.wav", "rb") as f:
            st.download_button("⬇️ Download WAV", f, file_name="generated.wav")
            
        # Vis
        tab1, tab2 = st.tabs(["Waveform", "Agent Policy"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(audio[0:rate*2])
            ax.set_title("Waveform Segment")
            st.pyplot(fig)
            
        with tab2:
            fig2, ax2 = plt.subplots()
            # Visualize the policy matrix (Brain of the agent)
            sns.heatmap(st.session_state['policy'], ax=ax2, cmap="viridis")
            ax2.set_title("TRPO Policy Matrix (State-Action probs)")
            ax2.set_xlabel("Next Note Action")
            ax2.set_ylabel("Current Note State")
            st.pyplot(fig2)


# End of App
