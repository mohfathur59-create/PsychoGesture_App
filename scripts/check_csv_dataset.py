import pandas as pd
import os

# Biar path-nya gak error meskipun lo jalanin dari mana aja
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_sibi_final_126.csv')

def audit_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: File {CSV_PATH} gak ada!")
        return

    df = pd.read_csv(CSV_PATH, sep=';')

    print("\n📊 JUMLAH DATA PER HURUF:")
    print(df['label'].value_counts())

    # Cek baris kosong/sampah
    raw_data = df.drop(columns=['label'])
    suspicious_rows = df[(raw_data == 0).sum(axis=1) > 100]

    print("-" * 30)
    if not suspicious_rows.empty:
        print(f"⚠️ DITEMUKAN {len(suspicious_rows)} BARIS MENCURIGAKAN (DATA KOSONG/0):")
        # Nampilin indeks baris biar gampang dihapus di Excel
        print(suspicious_rows.index.tolist())
        print("\n💡 Saran: Buka CSV lo, liat nomor baris di atas, terus hapus baris itu.")
    else:
        print("✅ DATASET BERSIH DARI BARIS KOSONG!")
    print("-" * 30)

if __name__ == "__main__":
    audit_dataset()