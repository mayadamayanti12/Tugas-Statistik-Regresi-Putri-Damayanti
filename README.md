📊 Analisis Regresi Linear Berganda untuk Memprediksi Jumlah Kalori yang Terbakar Berdasarkan Durasi Olahraga dan Rata-Rata Detak Jantung

Tugas Besar / Praktikum Statistika dan Probabilitas — Program Studi Sistem Informasi.

---

## 👤 Biodata Mahasiswa
* **Nama** : Putri Damayanti
* **NIM** : F5212520050

---

## 📝 Deskripsi Studi Kasus
Proyek ini dibuat untuk membangun sebuah model prediktif menggunakan metode **Analisis Regresi Linear Berganda**. Fokus utama dari studi kasus ini adalah mengevaluasi seberapa besar pengaruh variabel aktivitas fisik manusia terhadap total energi (kalori) yang berhasil dikeluarkan.

### 📌 Variabel Penelitian:
* **Variabel Dependen ($Y$)**: Jumlah Kalori yang Terbakar (Kalori)
* **Variabel Independen ($X_1$)**: Durasi Olahraga (Jam)
* **Variabel Independen ($X_2$)**: Rata-Rata Detak Jantung (BPM)

### 🧮 Model Matematika:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon$$

---

## 📁 Struktur File & Output Sistem
Di dalam repositori ini terdapat beberapa file utama hasil pengerjaan analisis data:
1. 📄 **`regresi.py`** : *Source code* utama berbasis Python yang memuat 100 baris dataset riil, proses *training model* OLS (*Ordinary Least Squares*), dan perintah visualisasi grafik.
2. 📊 **`Hasil_Grafik_Regresi_Putri.png`** : Gambar visualisasi evaluasi model yang memperlihatkan sebaran data asli (*Scatter Plot*) dibandingkan dengan Garis Prediksi Sempurna dari rumus regresi.
3. 📝 **`Hasil_Ringkasan_Statistik_Putri.txt`** : Dokumen teks berisi ringkasan statistik lengkap (*Statistical Summary*) mencakup nilai *R-squared*, koefisien regresi, nilai t-hitung, serta *P-value* untuk uji signifikansi.

---

## 🚀 Cara Menjalankan Program

### 1. Install Library Pendukung
Pastikan perangkat Anda sudah menginstall beberapa *library* data science berikut melalui Terminal / Command Prompt (CMD):
```bash
pip install pandas numpy matplotlib seaborn statsmodels
