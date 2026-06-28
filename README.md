# NeuroScreen — EEG Depression Risk Detection
## Cara Menjalankan (5 menit setup)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan file model
Taruh file ini di folder yang sama dengan app.py:
- `best_model.pkl`     ← dari hasil training run_satriadata.py
- `feature_names.pkl`  ← dari hasil training run_satriadata.py

Kalau belum ada, app tetap bisa jalan dalam **Demo Mode**.

### 3. Jalankan server
```bash
python app.py
```

### 4. Buka browser
```
http://localhost:5000
```

---

## Cara Pakai

**Upload file EEG:**
- Upload file .txt dari OpenBCI GUI
- Klik "Analisis EEG"
- Hasil muncul otomatis

**Demo Mode:**
- Klik "Demo Mode" tanpa upload file
- Cocok untuk demo presentasi tanpa data

---

## Struktur Folder
```
neuroscreen/
├── app.py              ← server utama (jalankan ini)
├── requirements.txt    ← dependencies
├── README.md           ← panduan ini
├── best_model.pkl      ← taruh di sini (dari training)
└── feature_names.pkl   ← taruh di sini (dari training)
```

---

## Untuk Demo Presentasi
1. Jalankan `python app.py`
2. Buka `http://localhost:5000`
3. Upload file `OpenBCI-RAW-Con18_Ali_EO.txt` dari dataset Kaggle
4. Atau klik "Demo Mode" untuk demo instan

---
*NeuroScreen · Satria Data 2026 · Bukan alat diagnosis klinis*
