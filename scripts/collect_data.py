import cv2
import mediapipe as mp
import pandas as pd
import os
import csv

# --- 1. SETUP PATH UTAMA ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Pastikan nama file ini sama dengan yang dipake buat training
SAVE_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_sibi_final_126.csv')

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Input Label
label = input("Masukkan Label Huruf/Keluhan (misal: A, B, PUSING): ").upper()
print(f"--- MEREKAM UNTUK LABEL: {label} ---")
print(f"📂 Data akan ditambahkan ke: {SAVE_PATH}")
print(f"🚀 Tekan 'S' untuk SIMPAN, 'Q' untuk SELESAI.")

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Inisialisasi 126 titik nol
    all_coords = [0.0] * 126 

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2: break 
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Normalisasi berbasis Wrist (Pergelangan Tangan)
            wrist = hand_landmarks.landmark[0]
            start_idx = i * 63
            for j, lm in enumerate(hand_landmarks.landmark):
                all_coords[start_idx + (j * 3)] = lm.x - wrist.x
                all_coords[start_idx + (j * 3) + 1] = lm.y - wrist.y
                all_coords[start_idx + (j * 3) + 2] = lm.z - wrist.z

    # UI Overlay
    cv2.rectangle(frame, (0,0), (350, 80), (0,0,0), -1)
    cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Data Terkumpul: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Expert Data Collector", frame)

    key = cv2.waitKey(1)
    
    # Simpan Data (Tekan S)
    if key == ord('s'):
        if any(all_coords):
            file_exists = os.path.isfile(SAVE_PATH)
            
            # Mode 'a' (Append) = Nambahin ke bawah, gak nimpa file
            with open(SAVE_PATH, mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                
                # Tulis Header cuma kalau filenya bener-bener baru dibikin
                if not file_exists:
                    header = ['label'] + [f'feat_{i}' for i in range(126)]
                    writer.writerow(header)
                
                writer.writerow([label] + all_coords)
            
            count += 1
            print(f"✅ [{label}] Data ke-{count} berhasil masuk ke dataset utama!")
        else:
            print("⚠️ Tangan gak kelihatan bray, gagal simpan!")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"--- SELESAI! Total data baru untuk {label}: {count} ---")