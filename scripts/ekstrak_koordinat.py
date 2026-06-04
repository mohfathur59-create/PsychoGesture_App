import cv2
import mediapipe as mp
import csv
import os

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Folder 'scripts'

# Lokasi Foto: PsychoGesture_App > data_uji > citra BISINDO
FOLDER_SUMBER = os.path.join(BASE_DIR, '..', 'data_uji', 'citra BISINDO')

# Lokasi Simpan CSV: PsychoGesture_App > dataset > dataset_sibi_final_126.csv
FILE_CSV = os.path.join(BASE_DIR, '..', 'dataset', 'dataset_sibi_final_126.csv')

EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)

# Cek apakah folder sumber ada
if not os.path.exists(FOLDER_SUMBER):
    print(f"❌ ERROR: Folder sumber tidak ditemukan di: {os.path.abspath(FOLDER_SUMBER)}")
    exit()

# Cek apakah folder 'dataset' ada, kalau belum ada kita buat otomatis
folder_dataset = os.path.dirname(FILE_CSV)
if not os.path.exists(folder_dataset):
    os.makedirs(folder_dataset)
    print(f"📁 Folder 'dataset' baru saja dibuat.")

# Header CSV
header = ['label']
for i in range(42):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

print(f"🚀 Mulai ekstrak dari: {os.path.abspath(FOLDER_SUMBER)}")
print(f"📂 Hasil akan disimpan ke: {os.path.abspath(FILE_CSV)}")

with open(FILE_CSV, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(header)

    total_success = 0

    # Looping folder A-Z di dalam 'citra BISINDO'
    for folder_huruf in sorted(os.listdir(FOLDER_SUMBER)):
        path_folder = os.path.join(FOLDER_SUMBER, folder_huruf)
        
        if not os.path.isdir(path_folder): continue
        
        print(f"📂 Memproses Huruf [{folder_huruf}]", end="... ")
        count_file = 0
        
        for file_gambar in os.listdir(path_folder):
            if not file_gambar.lower().endswith(EXTENSIONS): continue
            
            img = cv2.imread(os.path.join(path_folder, file_gambar))
            if img is None: continue
            
            # Deteksi Landmark
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            data_koordinat = [0.0] * 126 
            
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if i >= 2: break 
                    
                    # Normalisasi Relatif terhadap Wrist (Pergelangan Tangan)
                    wrist = hand_landmarks.landmark[0]
                    start_idx = i * 63
                    for j, lm in enumerate(hand_landmarks.landmark):
                        data_koordinat[start_idx + (j*3)] = lm.x - wrist.x
                        data_koordinat[start_idx + (j*3)+1] = lm.y - wrist.y
                        data_koordinat[start_idx + (j*3)+2] = lm.z - wrist.z
                
                writer.writerow([folder_huruf] + data_koordinat)
                total_success += 1
                count_file += 1
        
        print(f"Selesai ({count_file} gambar)")

print(f"\n✨ PROSES SELESAI!")
print(f"📊 Total data berhasil diekstrak: {total_success}")
print(f"📂 Cek folder dataset untuk melihat hasilnya!")