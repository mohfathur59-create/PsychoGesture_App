import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
from collections import deque # Buat sistem voting

# --- 1. SETUP PATH & MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_SIBI = os.path.join(BASE_DIR, 'models', 'knn_sibi_model.pkl')

model_sibi, le_sibi, scaler_sibi = None, None, None
# STABILIZER: Simpan 10 prediksi terakhir biar gak loncat-loncat
prediction_history = deque(maxlen=10)

try:
    with open(PATH_SIBI, 'rb') as f:
        bundle = pickle.load(f)
        model_sibi = bundle['model']
        le_sibi = bundle['label_encoder']
        scaler_sibi = bundle['scaler']
    print(f"✅ Model SIBI Berhasil Disinkronkan (Acc: {bundle.get('accuracy', 0):.2f}%)")
except Exception as e:
    print(f"❌ Gagal Load Model: {e}")

# --- 2. SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,      # Fokus 1 tangan dulu biar lebih presisi buat D, F, H, dkk
    model_complexity=1,   # Naikin ke 1 biar tracking titik jari lebih kuat
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

def deteksi_isyarat(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    all_coords = [0.0] * 126 

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 1: break # Fokus satu tangan
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            wrist = hand_landmarks.landmark[0]
            for j, lm in enumerate(hand_landmarks.landmark):
                all_coords[j * 3] = lm.x - wrist.x
                all_coords[j * 3 + 1] = lm.y - wrist.y
                all_coords[j * 3 + 2] = lm.z - wrist.z

    coords_input = np.array([all_coords], dtype=np.float64)

    try:
        if any(all_coords) and model_sibi:
            coords_scaled = scaler_sibi.transform(coords_input)
            probs = model_sibi.predict_proba(coords_scaled)
            prob_s = np.max(probs)

            # Threshold diperketat ke 0.85 biar huruf bandel gak asal lewat
            if prob_s > 0.85:
                pred_idx = model_sibi.predict(coords_scaled)
                curr_label = le_sibi.inverse_transform(pred_idx)[0]
                prediction_history.append(curr_label)
            else:
                prediction_history.append("...")

            # JURUS STABILIZER: Ambil suara terbanyak dari sejarah prediksi
            if len(prediction_history) > 0:
                # Cari label yang paling sering muncul di deque
                final_label = max(set(prediction_history), key=list(prediction_history).count)
            else:
                final_label = "MENGIDENTIFIKASI..."

            # Jangan tampilin label kalau isinya cuma titik-titik (ragu)
            display_text = f"SIBI: {final_label}" if final_label != "..." else "MENGIDENTIFIKASI..."
            return frame, f"{display_text} ({int(prob_s*100)}%)"
        
        prediction_history.clear() # Reset kalau tangan ilang
        return frame, "MENUNGGU TANGAN..."
    except Exception as e:
        return frame, "IDENTIFIKASI..."

# --- 3. MAIN LOOP ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        frame, hasil_label = deteksi_isyarat(frame)
        
        cv2.rectangle(frame, (0, 0), (500, 70), (0, 0, 0), -1)
        cv2.putText(frame, hasil_label, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("PsychoGesture Stabilized Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()