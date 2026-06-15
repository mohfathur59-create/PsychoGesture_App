import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# 1. LOAD DATASET (A-Z & Kosa Kata Keluhan) WITH AUTO-DETECTION
# =========================================================================
# Alamat jalur absolut ke folder dataset lu bray
csv_path = 'C:/xampp/htdocs/PsychoGesture_App/dataset/dataset_sibi_final_126_clean.csv'

# Gunakan sep=None dan engine='python' agar Pandas otomatis mendeteksi pemisah kolom (, atau ;)
df = pd.read_csv(csv_path, sep=None, engine='python')

# Bersihkan spasi gaib di nama kolom (jika ada)
df.columns = df.columns.str.strip()

# Hapus kolom yang benar-benar kosong di Excel (jika ada)
df = df.dropna(how='all', axis=1)

# Berdasarkan screenshot lu bray, kolom 'label' ada di PALING KIRI (Kolom pertama)
y = df.iloc[:, 0]    # Mengambil HANYA kolom pertama (indeks 0) sebagai target Label
X = df.iloc[:, 1:]   # Mengambil dari kolom kedua (indeks 1) sampai ujung kanan (Semua feat_x)

print(f"Sukses load data men! Total: {X.shape[0]} baris dan {X.shape[1]} kolom fitur koordinat.")

# =========================================================================
# 2. SPLIT DATASET (80% Training, 20% Testing)
# =========================================================================
# Data Uji (Testing) otomatis dapat porsi 20% dari total data untuk evaluasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# =========================================================================
# 3. PROSES K-NN (Klasifikasi di Latar Belakang)
# =========================================================================
# Menentukan nilai K = 5 (Tetangga Terdekat), standar mantap buat skripsi
model_knn = KNeighborsClassifier(n_neighbors=5)

# Proses melatih model dengan matriks data latih
model_knn.fit(X_train, y_train)

# Proses Prediksi: Komputer menghitung jarak untuk menebak data uji satu per satu bray
y_pred = model_knn.predict(X_test)
y_true = y_test

# =========================================================================
# 4. GENERATE CONFUSION MATRIX (Untuk Taruh di Bab 4)
# =========================================================================
# Menghitung kecocokan aktual vs prediksi
cm = confusion_matrix(y_true, y_pred)

# Ambil list nama unik dari label asli kamu agar urutannya pas men (misal: 'A', 'B', dst.)
labels_unik = np.unique(y_true)

# Membuat visualisasi grafik matriks kotak-kokak besar (Ukuran 16x14 inch biar teks ga numpuk)
plt.figure(figsize=(16, 14))

# Mengisi xticklabels dan yticklabels dengan nama label asli hasil deteksi unik
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_unik, yticklabels=labels_unik)

plt.xlabel('Predicted Label (Tebakan Sistem K-NN)', fontsize=12)
plt.ylabel('Actual Label (Data Asli/Target)', fontsize=12)
plt.title('Confusion Matrix Sistem Klasifikasi Isyarat SIBI', fontsize=14, fontweight='bold')

# Membuat label teks di sumbu X agak miring 45 derajat agar rapi dibaca jika panjang
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Gambar otomatis kesimpan di folder 'scripts' tempat kamu jalanin file ini bray
plt.savefig('confusion_matrix_skripsi_4000.png', dpi=300, bbox_inches='tight') 
plt.show()

# =========================================================================
# 5. OUTPUT ANGKA EVALUASI (Akurasi Total & Detail per Huruf/Keluhan)
# =========================================================================
print("========================================================")
print(f"AKURASI GLOBAL SISTEM: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("========================================================")
print("\nLAPORAN PER KELAS (A-Z & KOSAKATA KELUHAN):")
print(classification_report(y_true, y_pred))