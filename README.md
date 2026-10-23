# Prediksi Niat Pembelian Pengunjung E-Commerce (Regresi Logistik)

Tugas Akhir 3 (TA-03) Pembelajaran Mesin. Aplikasi web Flask yang memprediksi apakah satu sesi
pengunjung toko online akan berakhir dengan pembelian.

- **Dataset:** Online Shoppers Purchasing Intention (12.330 sesi, 17 fitur, target `Revenue`)
- **Model:** Logistic Regression + StandardScaler + SMOTE (hanya pada data latih)
- **Demo:** ISI_LINK_RENDER_DI_SINI
- **Video:** ISI_LINK_YOUTUBE_DI_SINI

## Struktur proyek

```
data/        dataset mentah
notebook/    01_eda_modeling_purchase_intent.ipynb (EDA, training, evaluasi, simpan model)
model/       artefak model (dibuat oleh notebook)
templates/   index.html
static/      style.css
images/      grafik hasil notebook
docs/        laporan DOC dan PDF
app.py       aplikasi Flask
```

## Cara menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac/Linux
pip install -r requirements-notebook.txt
```

1. Buka dan jalankan semua sel `notebook/01_eda_modeling_purchase_intent.ipynb`
   (Run All). Ini membuat file di folder `model/` dan grafik di `images/`.
2. Jalankan aplikasi:
   ```bash
   python app.py
   ```
3. Buka http://127.0.0.1:5000

## Deployment (Render)

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Pastikan folder `model/` ikut di-commit dan versi `scikit-learn` di `requirements.txt`
  sama dengan versi saat training.

## Catatan keterbatasan

- Fitur di luar formulir (mis. `Browser`, `Region`, `TrafficType`) diisi nilai default.
- `PageValues` dihitung dari halaman yang dikunjungi sebelum transaksi, sehingga pada sistem
  nyata nilainya baru tersedia setelah ada aktivitas dalam sesi.
