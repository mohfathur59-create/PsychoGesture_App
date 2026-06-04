import pickle
import pandas as pd
import numpy as np
import os

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn_sibi_model.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_sibi_bersih.csv')

def test_otak_ai():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model .pkl gak ketemu! Training dulu bray.")
        return

    # 1. Load Model
    with open(MODEL_PATH, 'rb') as f:
        data_load = pickle.load(f)
        model = data_load['model']
        le = data_load['label_encoder']

    print("✅ Model AI Berhasil di-load!")

    # 2. Baca CSV
    if os.path.exists(CSV_PATH):
        # Pakai header=0 karena CSV lo ada judul (label;x0;y0...)
        df = pd.read_csv(CSV_PATH, sep=';', header=0, low_memory=False)

        # Buang baris yang mungkin kotor (NaN)
        df_clean = df.dropna()

        if df_clean.empty:
            print("❌ Dataset kosong setelah dibersihkan.")
            return

        try:
            # --- SESUAIKAN INDEX DISINI ---
            # X (Fitur): Ambil semua kolom KECUALI kolom 'label'
            sample_features = df_clean.drop(columns=['label']).iloc[0].values
            sample_data = sample_features.astype(float).reshape(1, -1)
            
            # Y (Target): Ambil nilai dari kolom 'label'
            label_asli = df_clean.iloc[0]['label']

            # Prediksi
            prediksi_angka = model.predict(sample_data)
            prediksi_huruf = le.inverse_transform(prediksi_angka)[0]

            print("\n=== HASIL TES LOGIKA AI ===")
            print(f"🎯 Label Asli di CSV : {label_asli}")
            print(f"🤖 Prediksi dari AI  : {prediksi_huruf}")

            if str(label_asli) == str(prediksi_huruf):
                print("\n✅ MANTAP! AI lo pinter, tebakannya tembus.")
            else:
                print("\n⚠️ AI nebaknya beda. Mungkin butuh lebih banyak data training.")
                
        except Exception as e:
            print(f"❌ Error saat ngetes: {e}")
    else:
        print("❌ File CSV-nya nggak ada di folder dataset bray.")

if __name__ == "__main__":
    test_otak_ai()