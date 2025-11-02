# Deteksi Ngantuk 😴

Aplikasi berbasis **Python dan OpenCV** untuk mendeteksi tingkat kantuk dan kelelahan pengemudi menggunakan kamera.  
Sistem ini menghitung **EAR (Eye Aspect Ratio)** dan **MAR (Mouth Aspect Ratio)** untuk menentukan apakah seseorang mengantuk atau hanya tersenyum.

---

## 🚀 Fitur Utama
- Deteksi mata tertutup dengan **MediaPipe / OpenCV**
- Peringatan audio saat pengguna mengantuk
- Pengecualian otomatis saat pengguna tersenyum (tidak dihitung ngantuk)
- Kompatibel untuk integrasi ke **Kivy (Android App)**

---

## 🧩 Library yang Digunakan
- `opencv-python`
- `mediapipe`
- `numpy`
- `imutils`
- `plyer` *(untuk notifikasi suara di Android)*

---

## 💻 Cara Menjalankan
1. Clone repositori:
   ```bash
   git clone https://github.com/archeswannn-source/DETEKSINGANTUK-TBC-.git
   cd DETEKSINGANTUK-TBC-
