import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from PIL import Image
import cv2
import seaborn as sns 

# NEW BACKEND IMPORTS
import project_backend

# Set page config (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Advanced Emotion Music Generator (CNN + RL)",
    page_icon="🧠",
    layout="wide"
)

# Initialize Components
# Removed cache to ensure strict model reloading
def get_components(music_type="Indian Classical"):
    cnn = project_backend.EmotionCNN()
    # Path to user CSV
    if music_type == "Western":
        csv_path = "western_instruments_features.csv"
    else:
        csv_path = "final2.O_merged_instrument_dataset(2054 audios).csv"
        
    data_handler = project_backend.DataHandler(csv_path)
    data_handler.load_data()
    return cnn, data_handler

# Initial load will be handled inside sidebar to be reactive
cnn_model = project_backend.EmotionCNN()
# ... (intermediate styles unchanged)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #6C5CE7; text-align: center; font-weight: bold; }
    .sub-header { font-size: 1.2rem; color: #444; text-align: center; margin-bottom: 2rem; }
    .stButton>button { width: 100%; background-color: #6C5CE7; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Cortex Music Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Keras CNN & Reinforcement Learning (TRPO/SAC)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    input_mode = st.radio("Input Mode", ["Upload Image", "Manual Selection"])
    
    st.divider()
    music_type = st.radio("Music Type", ["Indian Classical", "Western"], index=0)
    agent_type = st.radio("Agent Type (RL Algorithm)", ["TRPO", "SAC"], index=0)
    
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
        selected_emotion = st.selectbox("Select Emotion", project_backend.EMOTIONS)
        
    if music_type == "Western":
        selected_instrument = st.selectbox("Select Instrument", project_backend.WESTERN_INSTRUMENTS)
    else:
        selected_instrument = st.selectbox("Select Instrument", project_backend.INSTRUMENTS)
    
    # Reload DataHandler if music type changed
    _, data_handler = get_components(music_type)
    
    duration = st.slider("Duration (seconds)", 5, 30, 10)

# Main Area
output_container = st.container()
col1, col2 = st.columns([1, 2])

with col1:
    st.info(f"Click to run the {agent_type} Agent.")
    if st.button("🚀 Generate with AI", type="primary"):
        with st.spinner(f"{agent_type} Agent generating {selected_instrument} melody..."):
            
            try:
                # CALL NEW BACKEND
                audio_data, rate, policy = project_backend.generate_session(
                    selected_emotion, 
                    selected_instrument, 
                    data_handler, 
                    duration,
                    agent_type=agent_type,
                    music_type=music_type
                )
                
                st.session_state['audio'] = audio_data
                st.session_state['rate'] = rate
                st.session_state['policy'] = policy # To visualize agent brain
                st.session_state['meta'] = f"{selected_emotion} - {selected_instrument} ({agent_type})"
                st.session_state['agent_type'] = agent_type
                
                st.success("Synthesis Complete!")
            except Exception as e:
                st.error(f"Error in backend: {type(e).__name__}: {str(e)}")
                # Log traceback for user to see
                import traceback
                st.code(traceback.format_exc())

if 'audio' in st.session_state:
    audio = st.session_state['audio']
    rate = st.session_state['rate']
    current_agent = st.session_state.get('agent_type', 'Agent')
    
    with col2:
        st.subheader(f"🎶 Result: {st.session_state['meta']}")
        st.audio(audio, sample_rate=rate)
        
        # Download
        # Normalize int16
        scaled = np.int16(audio * 32767)
        wavfile.write("temp_out.wav", rate, scaled)
        with open("temp_out.wav", "rb") as f:
            st.download_button("⬇️ Download WAV", f, file_name=f"generated_{current_agent.lower()}.wav")
            
        # Vis
        tab1, tab2 = st.tabs(["Waveform", "Agent Policy"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(audio[0:rate*2])
            ax.set_title(
                f"{st.session_state['meta']} – Waveform"
            )
            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
        with tab2:
            fig2, ax2 = plt.subplots()
            # Visualize the policy matrix (Brain of the agent)
            sns.heatmap(st.session_state['policy'], ax=ax2, cmap="viridis")
            ax2.set_title(f"{current_agent} Policy Matrix (State-Action probs)")
            ax2.set_xlabel("Next Note Action")
            ax2.set_ylabel("Current Note State")
            st.pyplot(fig2)


# Agent Performance Section
st.divider()
st.markdown('<div class="main-header">📊 Agent Analysis & Metrics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comparative Performance and Learning Curves</div>', unsafe_allow_html=True)

mt_suffix = "western" if music_type == "Western" else "indian"

perf_tab1, perf_tab2 = st.tabs(["🚀 SAC Performance", "🏹 TRPO Performance"])

with perf_tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"Performance Matrix ({music_type})")
        img_path = f"assets/sac_performance_matrix_{mt_suffix}.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.error(f"SAC Performance Matrix asset missing: {img_path}")
    with col_b:
        st.subheader(f"Learning Performance ({music_type})")
        img_path = f"assets/sac_learning_graph_{mt_suffix}.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.error(f"SAC Learning Graph asset missing: {img_path}")

with perf_tab2:
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader(f"Performance Matrix ({music_type})")
        img_path = f"assets/trpo_performance_matrix_{mt_suffix}.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.error(f"TRPO Performance Matrix asset missing: {img_path}")
    with col_d:
        st.subheader(f"Learning Performance ({music_type})")
        img_path = f"assets/trpo_learning_graph_{mt_suffix}.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.error(f"TRPO Learning Graph asset missing: {img_path}")

# End of App
