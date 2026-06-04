import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# --- 1. SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_sibi_final_126_clean.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODEL_DIR): 
    os.makedirs(MODEL_DIR)

def training_final_boss():
    print("-" * 45)
    print("🚀 MEMULAI TRAINING AI (FINAL SINKRON)")
    print("-" * 45)
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ File {CSV_PATH} tidak ditemukan!")
        print("💡 Pastikan sudah menjalankan clean_data.py ya bray.")
        return

    # --- 2. LOAD DATA (DENGAN PENGAMAN MULTI-SEPARATOR) ---
    df = None
    try:
        # Coba baca pake Titik Koma (;) dulu sesuai bawaan lo bray
        df = pd.read_csv(CSV_PATH, sep=';', engine='python')
        # Jika kolom 'label' gak ketahuan, berarti pemisahnya bukan titik koma bray
        if 'label' not in df.columns:
            raise KeyError
        print("📊 Menggunakan Data (Separator: Titik Koma ';'): " + str(len(df)) + " baris.")
    except:
        try:
            # Fallback: Coba baca pake Koma (,) biar sinkron sama collect_data terbaru
            df = pd.read_csv(CSV_PATH, sep=',', engine='python')
            if 'label' not in df.columns:
                # Jika tanpa header teks (hanya angka), kita kasih nama manual kolom pertamanya
                df = pd.read_csv(CSV_PATH, header=None, engine='python')
                df.rename(columns={0: 'label'}, inplace=True)
            print("📊 Menggunakan Data (Separator: Koma ','): " + str(len(df)) + " baris.")
        except Exception as e:
            print(f"❌ Gagal baca CSV: {e}")
            return

    # Pisahkan fitur (X) dan label (y)
    X = df.drop(columns=['label']).values.astype(np.float64)
    y = df['label'].values

    # --- 3. PREPROCESSING (SINKRONISASI SKALA) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data: 80% Belajar, 20% Tes Ujian
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- 4. CONFIG KNN OPTIMIZED (SUDAH TOP) ---
    model = KNeighborsClassifier(
        n_neighbors=5, 
        weights='distance', 
        metric='manhattan'
    )
    model.fit(X_train, y_train)

    # Cek Akurasi
    accuracy = model.score(X_test, y_test) * 100

    # --- 5. SAVE MODEL BUNDLE ---
    model_bundle = {
        'model': model, 
        'label_encoder': le,
        'scaler': scaler, 
        'accuracy': accuracy
    }
    
    output_path = os.path.join(MODEL_DIR, 'knn_sibi_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_bundle, f)

    print("-" * 45)
    print(f"🎯 AKURASI FINAL: {accuracy:.2f}%")
    print(f"📂 Model Disimpan ke: {output_path}")
    print("-" * 45)
    print("💡 SELESAI! Sekarang langsung buka Streamlit dan tes hurufnya.")

if __name__ == "__main__":
    training_final_boss()