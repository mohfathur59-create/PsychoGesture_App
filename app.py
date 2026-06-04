import streamlit as st
import cv2
import av
import time
import numpy as np
import pickle
import os
import warnings
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
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
if 'confirm_counter' not in st.session_state: st.session_state.confirm_counter = 0

# --- 4. LOGIKA SIBI (126 COORDS - LOCK) ---
def extract_features_synced(results):
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

class SibiExpertProcessor(VideoProcessorBase):
    def __init__(self):
        self.label = "..."
        self.conf = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = hands_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        features = extract_features_synced(results)
        
        if any(features) and model_sibi:
            try:
                feat_scaled = scaler.transform([features])
                probs = model_sibi.predict_proba(feat_scaled)
                self.conf = int(np.max(probs) * 100)
                if self.conf > 80:
                    pred = model_sibi.predict(feat_scaled)
                    self.label = le_sibi.inverse_transform(pred)[0]
                else: self.label = "..."
            except: pass
        else:
            self.label = "..."
            self.conf = 0
            
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
        # FIX: Konfigurasi RTC untuk browser user
        RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        ctx = webrtc_streamer(
            key="cam", 
            video_processor_factory=SibiExpertProcessor, 
            rtc_configuration=RTC_CONFIG,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False}
        )
        if st.button("🔄 Switch ke Mode Avatar", use_container_width=True):
            st.session_state.mode_aplikasi = "Avatar"; st.rerun()

    with col_info:
        st.subheader("📊 Hasil Deteksi")
        res_place = st.empty(); sent_place = st.empty()
        if st.button(" Reset Kalimat", use_container_width=True):
            st.session_state.kalimat = ""; st.rerun()

        # FIX: Mengganti loop while dengan pengambilan data dari context
        if ctx.video_processor:
            lbl = getattr(ctx.video_processor, 'label', "...")
            cnf = getattr(ctx.video_processor, 'conf', 0)
            
            if lbl != "..." and lbl != st.session_state.last_label:
                st.session_state.confirm_counter += 1
                if st.session_state.confirm_counter > 15:
                    st.session_state.kalimat += lbl
                    st.session_state.last_label = lbl
                    st.session_state.confirm_counter = 0
            elif lbl == "...":
                st.session_state.last_label = ""

            res_place.markdown(f'<div class="result-card"><h1 class="label-big">{lbl}</h1><p>{cnf}% Confidence</p></div>', unsafe_allow_html=True)
            sent_place.markdown(f'<div class="sentence-box">{st.session_state.kalimat}</div>', unsafe_allow_html=True)

else:
    # --- MODE AVATAR (LOGIKA TETAP SAMA) ---
    with col_visual:
        st.subheader(" Visual Avatar")
        avatar_place = st.empty()
        if not st.session_state.kalimat:
            avatar_place.markdown('<div style="height:350px; border:2px dashed #333; border-radius:15px; display:flex; align-items:center; justify-content:center; color:#555;">Menunggu Input Psikolog...</div>', unsafe_allow_html=True)
        if st.button("🔄 swicth", use_container_width=True):
            st.session_state.mode_aplikasi = "Kamera"; st.rerun()
    with col_info:
        st.subheader("💬 Input Psikolog")
        pesan = st.chat_input("Ketik kata atau kalimat")
        if pesan:
            st.session_state.kalimat = pesan
            raw_words = pesan.strip().upper().split()
            daftar_kata = []
            i = 0
            while i < len(raw_words):
                if i < len(raw_words) - 1 and f"{raw_words[i]} {raw_words[i+1]}" in ["GAMPANG LELAH", "SULIT TIDUR", "SAKIT MUAL", "BEBAN PIKIRAN"]:
                    daftar_kata.append(f"{raw_words[i]} {raw_words[i+1]}"); i += 2
                else:
                    daftar_kata.append(raw_words[i]); i += 1
            folder_base = "data_uji/citra BISINDO"
            for kata in daftar_kata:
                kata_folder = kata
                if kata in ["MUDAH MARAH", "PUTUS ASA", "MATI RASA", "TIDAK SEMANGAT", "RAGU RAGU"]:
                    kata_folder = kata.replace(" ", "-")
                path_kata = os.path.join(folder_base, kata_folder)
                if os.path.exists(path_kata):
                    imgs = [f for f in os.listdir(path_kata) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if imgs:
                        img_p = os.path.join(path_kata, imgs[0])
                        avatar_place.image(img_p, width=400, caption=f"Kata Isyarat: {kata}")
                        time.sleep(2.0)
                else:
                    for huruf in kata:
                        path_huruf = os.path.join(folder_base, huruf)
                        if os.path.exists(path_huruf):
                            imgs = [f for f in os.listdir(path_huruf) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            if imgs:
                                img_p = os.path.join(path_huruf, imgs[0])
                                avatar_place.image(img_p, width=400, caption=f"Mengeja: {huruf}")
                                time.sleep(1.0)
            st.success("Selesai Menerjemahkan!")
        st.write("**History Teks:**")
        st.markdown(f'<div class="sentence-box">{st.session_state.kalimat}</div>', unsafe_allow_html=True)
