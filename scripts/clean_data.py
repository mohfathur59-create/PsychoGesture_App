import pandas as pd
import os
import numpy as np

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_ASAL = os.path.join(BASE_DIR, 'dataset', 'dataset_sibi_final_126.csv')
FILE_BERSIH = os.path.join(BASE_DIR, 'dataset', 'dataset_sibi_final_126_clean.csv')

def eksekusi_pembersihan():
    print("-" * 45)
    print("🧹 PROSES PEMBERSIHAN SUPER SENSITIF")
    print("-" * 45)

    if not os.path.exists(FILE_ASAL):
        print(f"❌ File tidak ditemukan!")
        return

    # 1. Load Data
    df = pd.read_csv(FILE_ASAL, sep=';')
    total_awal = len(df)

    # 2. KONVERSI & NORMALISASI DATA
    # Pastikan semua jadi float dan ganti koma ke titik
    for col in df.columns:
        if col != 'label':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    # 3. BUANG DATA NOISE (Hampir Nol)
    # Kadang MediaPipe ngasih angka sangat kecil kayak 1e-07, itu kita anggap 0
    df_numeric = df.drop(columns=['label'])
    df_numeric = df_numeric.mask(df_numeric.abs() < 1e-5, 0) # Angka di bawah 0.00001 dianggap 0
    
    # Gabung balik sama label
    df = pd.concat([df['label'], df_numeric], axis=1)

    # 4. FILTER JURUS "SAYONARA DATA KOTOR"
    # Kita buang baris yang isinya nol lebih dari 63 (artinya tangan kedua gak ada dan tangan pertama gak lengkap)
    # Atau lo bisa sesuaikan thresholdnya. Kalau satu tangan (63 fitur), 
    # mestinya sisanya (63 lagi) boleh nol. Kalau nol lebih dari 100, berarti data rusak.
    threshold_nol = 100
    df_clean = df[(df == 0).sum(axis=1) < threshold_nol]
    
    # 5. HAPUS DUPLIKAT (Sangat Penting!)
    # Kita bulatkan dulu koordinatnya biar duplikat yang beda tipis bisa kehapus
    # Ini supaya AI nggak "overfitting" atau cuma hafal satu posisi
    df_clean = df_clean.drop_duplicates()

    # 6. SIMPAN
    df_clean.to_csv(FILE_BERSIH, index=False, sep=';')

    total_akhir = len(df_clean)
    print(f"📊 Data Awal      : {total_awal}")
    print(f"🗑️ Baris Dibuang   : {total_awal - total_akhir}")
    print(f"💎 Data Berkualitas : {total_akhir}")
    print(f"📂 Lokasi          : {FILE_BERSIH}")
    print("-" * 45)

if __name__ == "__main__":
    eksekusi_pembersihan()