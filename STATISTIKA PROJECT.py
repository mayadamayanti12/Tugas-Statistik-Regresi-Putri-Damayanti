import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# DATASET 100 BARIS ASLI
data = {
    'Rata_Rata_Detak_Jantung': [
        157, 151, 154, 160, 151, 153, 167, 137, 151, 143, 162, 134, 147, 163, 166, 162, 169, 132, 165, 148,
        158, 140, 164, 126, 167, 145, 165, 136, 150, 137, 149, 130, 164, 164, 150, 145, 149, 166, 131, 149,
        138, 160, 143, 141, 154, 151, 121, 137, 131, 162, 165, 135, 121, 128, 120, 143, 138, 150, 135, 165,
        122, 123, 159, 141, 156, 138, 140, 136, 148, 124, 122, 160, 142, 168, 169, 158, 133, 155, 162, 124,
        133, 168, 167, 149, 158, 127, 138, 162, 155, 160, 130, 154, 165, 146, 169, 169, 126, 168, 142, 165
    ],
    'Durasi_Latihan_Jam': [
        1.69, 1.30, 1.48, 1.83, 1.25, 0.58, 0.81, 1.31, 1.13, 0.70, 1.54, 1.41, 1.50, 0.99, 1.05, 1.62, 0.91, 1.61, 0.86, 0.55,
        1.25, 0.73, 1.93, 0.77, 1.83, 1.17, 1.37, 0.99, 1.69, 0.60, 0.83, 0.51, 1.83, 1.39, 0.61, 0.73, 0.79, 1.62, 1.67, 1.83,
        1.09, 1.38, 0.75, 1.82, 1.45, 1.55, 1.25, 0.81, 1.04, 1.09, 1.56, 1.97, 1.70, 1.04, 0.95, 0.93, 0.78, 1.05, 1.85, 1.89,
        0.58, 1.15, 1.30, 1.06, 0.79, 1.40, 1.86, 0.67, 1.11, 1.76, 1.89, 0.66, 0.99, 1.57, 0.62, 0.51, 1.23, 1.38, 1.82, 1.85,
        1.03, 1.38, 1.51, 0.99, 1.43, 0.97, 1.85, 1.25, 0.77, 1.08, 1.04, 1.31, 1.58, 1.48, 0.78, 0.58, 1.29, 1.33, 1.15, 1.97
    ],
    'Kalori_Terbakar': [
        1313, 883, 1073, 1485, 831, 269, 584, 765, 755, 410, 1269, 786, 959, 792, 856, 1319, 735, 936, 680, 318,
        910, 375, 1630, 246, 1572, 843, 1146, 563, 1184, 255, 521, 124, 1530, 1133, 360, 428, 517, 1406, 959, 1293,
        617, 1079, 415, 1222, 1058, 1157, 650, 417, 513, 824, 1308, 1214, 950, 525, 420, 555, 439, 691, 1154, 1656,
        137, 532, 977, 652, 608, 866, 1241, 313, 712, 1009, 1136, 442, 606, 1344, 461, 311, 689, 978, 1487, 1072,
        576, 1198, 1283, 685, 1111, 482, 1219, 939, 574, 856, 503, 915, 1295, 1018, 575, 441, 805, 1106, 735, 1625
    ]
}

df = pd.DataFrame(data)

X = df[['Durasi_Latihan_Jam', 'Rata_Rata_Detak_Jantung']]
Y = df['Kalori_Terbakar']

X_dengan_konstanta = sm.add_constant(X)
model = sm.OLS(Y, X_dengan_konstanta).fit()

print("="*70)
print("          HASIL ANALISIS REGRESI LINEAR BERGANDA INDIVIDU         ")
print("          Nama Mahasiswa: Putri Damayanti                         ")
print("          NIM           : F5212520050                             ")
print("="*70)
print(model.summary())
print("="*70)

# Menghapus cache grafik lama agar tidak menumpuk berkali-kali
plt.close('all') 

plt.figure(figsize=(8, 5))
Y_prediksi = model.predict(X_dengan_konstanta)

sns.scatterplot(x=Y, y=Y_prediksi, color='teal', alpha=0.7, label='Data Sampel (100 Orang)')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', lw=2, linestyle='--', label='Garis Prediksi Sempurna')

plt.title('Grafik Evaluasi Model: Kalori Aktual vs Prediksi Regresi', fontsize=12, fontweight='bold')
plt.xlabel('Kalori Aktual (Data Riil Lapangan)', fontsize=10)
plt.ylabel('Kalori Hasil Prediksi Rumus', fontsize=10)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Otomatis menyimpan file gambar hasil grafik ke laptop
plt.savefig('Hasil_Grafik_Regresi_Putri.png', dpi=300, bbox_inches='tight')

# Otomatis menyimpan file teks hasil statistika ke laptop
with open('Hasil_Ringkasan_Statistik_Putri.txt', 'w') as f:
    f.write(model.summary().as_text())

# Menampilkan grafik secara normal
plt.show()