import pickle
import pandas as pd
import numpy as np
import os

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn_sibi_model.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_sibi_final_126.csv')

def test_otak_ai():
    print("\n🔎 MENGETES KECERDASAN MODEL AI (EXPERT MODE)...")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ ERROR: File model .pkl belum ada!")
        return

    # 1. Load Model & Scaler
    with open(MODEL_PATH, 'rb') as f:
        data_load = pickle.load(f)
        model = data_load['model']
        le = data_load['label_encoder']
        scaler = data_load.get('scaler') # AMBIL SCALER-NYA JUGA BRAY!

    # 2. Baca CSV & Tes Satu Sampel
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, sep=';', header=0, low_memory=False)
        df_clean = df.dropna()

        if df_clean.empty:
            print("❌ ERROR: Dataset CSV kosong!")
            return

        try:
            # Ambil sampel baris pertama (misal huruf A)
            sample_features = df_clean.drop(columns=['label']).iloc[0].values
            sample_data = sample_features.astype(float).reshape(1, -1)
            
            label_asli = df_clean.iloc[0]['label']

            # --- JURUS PENYEMBUH: GUNAKAN SCALER ---
            if scaler:
                sample_data = scaler.transform(sample_data) # <--- WAJIB ADA INI
            # --------------------------------------

            # Proses Prediksi
            prediksi_angka = model.predict(sample_data)
            prediksi_huruf = le.inverse_transform(prediksi_angka)[0]

            print("-" * 30)
            print(f"📸 Data Sampel (Asli): {label_asli}")
            print(f"🤖 Hasil Tebakan AI : {prediksi_huruf}")
            
            if str(label_asli) == str(prediksi_huruf):
                print("\n✅ MANTAP! AI lo udah sinkron dan pinter.")
            else:
                print("\n⚠️ AI MASIH BINGUNG: Coba tambah data training lagi.")
            print("-" * 30)
                
        except Exception as e:
            print(f"❌ ERROR SAAT PROSES TES: {e}")
    else:
        print(f"❌ ERROR: File CSV tidak ditemukan!")

if __name__ == "__main__":
    test_otak_ai()