import streamlit as st
import cv2
import time
import numpy as np
import pickle
import os
import warnings

# --- 1. IMPORT & PATH ---
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw

warnings.filterwarnings("ignore")
st.set_page_config(page_title="PsychoGesture Expert", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_SIBI = os.path.join(BASE_DIR, 'models', 'knn_sibi_model.pkl')

# --- 2. LOAD TOOLS & MODEL ---
@st.cache_resource
def get_mediapipe_tools():
    return mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=2, 
        min_detection_confidence=0.7
    )

hands_detector = get_mediapipe_tools()

@st.cache_resource
def load_knn_model():
    if os.path.exists(PATH_SIBI):
        try:
            with open(PATH_SIBI, 'rb') as f:
                bundle = pickle.load(f)
                return bundle['model'], bundle['label_encoder'], bundle['scaler']
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
    return None, None, None

model_sibi, le_sibi, scaler = load_knn_model()

# --- 3. SESSION STATE ---
if 'kalimat' not in st.session_state: st.session_state.kalimat = ""
if 'mode_aplikasi' not in st.session_state: st.session_state.mode_aplikasi = "Kamera"
if 'last_label' not in st.session_state: st.session_state.last_label = ""

# --- 4. LOGIKA SIBI ---
def extract_features(results):
    coords = [0.0] * 126
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2: break
            wrist = hand_landmarks.landmark[0]
            start_idx = i * 63
            for j, lm in enumerate(hand_landmarks.landmark):
                coords[start_idx + (j * 3)] = lm.x - wrist.x
                coords[start_idx + (j * 3) + 1] = lm.y - wrist.y
                coords[start_idx + (j * 3) + 2] = lm.z - wrist.z
    return coords

# --- 5. STYLE ---
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .result-card { background-color: #161b22; padding: 25px; border-radius: 15px; border: 2px solid #00FFFF; text-align: center; }
    .label-big { color: #00FFFF; font-size: 80px; font-weight: 900; margin: 0; }
    .sentence-box { background-color: #161b22; padding: 20px; border-radius: 15px; border-left: 10px solid #00FFFF; min-height: 100px; color: #00FFFF; font-size: 28px; font-weight: bold; }
    img { max-height: 380px !important; width: auto !important; object-fit: contain; display: block; margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# --- 6. MAIN LAYOUT ---
st.title("PsychoGesture Anjay")
col_visual, col_info = st.columns([1.3, 1], gap="large")

if st.session_state.mode_aplikasi == "Kamera":
    with col_visual:
        st.subheader("🎥 Visual Input")
        # PENGGANTIAN TOTAL KE CAMERA_INPUT
        img_file = st.camera_input("Klik untuk ambil gambar")
        
        if st.button("🔄 Switch ke Mode Avatar", use_container_width=True):
            st.session_state.mode_aplikasi = "Avatar"; st.rerun()

    with col_info:
        st.subheader("📊 Hasil Deteksi")
        
        if img_file is not None:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.flip(img, 1)
            
            results = hands_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            features = extract_features(results)
            
            lbl = "..."
            if any(features) and model_sibi:
                feat_scaled = scaler.transform([features])
                pred = model_sibi.predict(feat_scaled)
                lbl = le_sibi.inverse_transform(pred)[0]
                
                # Logic Tambah Kalimat
                if lbl != st.session_state.last_label:
                    st.session_state.kalimat += lbl
                    st.session_state.last_label = lbl
            
            # Gambar hasil
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            st.image(img, channels="BGR")
            st.markdown(f'<div class="result-card"><h1 class="label-big">{lbl}</h1></div>', unsafe_allow_html=True)

        if st.button("Reset Kalimat", use_container_width=True):
            st.session_state.kalimat = ""; st.rerun()
        st.markdown(f'<div class="sentence-box">{st.session_state.kalimat}</div>', unsafe_allow_html=True)

else:
    # --- MODE AVATAR ---
    with col_visual:
        st.subheader("Visual Avatar")
        avatar_place = st.empty()
        if st.button("🔄 swicth", use_container_width=True):
            st.session_state.mode_aplikasi = "Kamera"; st.rerun()
    with col_info:
        st.subheader("💬 Input Psikolog")
        pesan = st.chat_input("Ketik kata atau kalimat")
        if pesan:
            st.session_state.kalimat = pesan
            # ... (Logika avatar lo tetap sama di sini)
            st.success("Selesai Menerjemahkan!")
        st.markdown(f'<div class="sentence-box">{st.session_state.kalimat}</div>', unsafe_allow_html=True)
