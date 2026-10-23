"""Aplikasi Flask: prediksi niat pembelian pengunjung e-commerce (Regresi Logistik).

Versi ini KOMPATIBEL dengan struktur eval.json yang dihasilkan notebook
`01_eda_modeling_purchase_intent.ipynb`, yaitu:
    {
      "threshold": 0.5,
      "n_test": 2441,
      "confusion_matrix": {"tn": ..., "fp": ..., "fn": ..., "tp": ...},
      "roc": {"fpr": [...], "tpr": [...], "auc": ..., "point": {...}},
      "y_true": [0, 1, 0, ...],          ← ditambahkan oleh cell update
      "y_pred_proba": [0.12, 0.78, ...]   ← ditambahkan oleh cell update
    }

Fitur live update: setiap submit form, observasi (proba, true_label) ditambahkan
ke SESSION_HISTORY. Confusion matrix & ROC curve dihitung ULANG dari gabungan
(test set asli + history sesi) dan ditampilkan berdampingan dengan metrik asli.
"""
import json
import os
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Muat artefak hasil training (dibuat oleh notebook)
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_DIR / "logreg_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
columns = joblib.load(MODEL_DIR / "columns.joblib")
num_cols = joblib.load(MODEL_DIR / "num_cols.joblib")
defaults = json.loads((MODEL_DIR / "defaults.json").read_text())
config = json.loads((MODEL_DIR / "config.json").read_text())

# Data evaluasi (dibuat oleh cell terakhir di notebook)
EVAL_PATH = MODEL_DIR / "eval.json"
if not EVAL_PATH.exists():
    raise FileNotFoundError(
        "File model/eval.json belum ada. Jalankan sel 'Simpan data evaluasi' "
        "di akhir notebook terlebih dahulu."
    )
eval_raw = json.loads(EVAL_PATH.read_text())

THRESHOLD = float(config.get("threshold", eval_raw.get("threshold", 0.5)))


# ---------------------------------------------------------------------------
# Normalisasi struktur eval.json (kompatibel dengan format notebook user)
# ---------------------------------------------------------------------------
def _normalize_eval(raw: dict) -> dict:
    """Konversi struktur eval.json dari notebook ke format internal yang seragam."""
    cm_raw = raw.get("confusion_matrix", {})
    if isinstance(cm_raw, dict):
        tn = int(cm_raw.get("tn", 0))
        fp = int(cm_raw.get("fp", 0))
        fn = int(cm_raw.get("fn", 0))
        tp = int(cm_raw.get("tp", 0))
    else:
        cm_list = cm_raw if cm_raw else [[0, 0], [0, 0]]
        tn, fp = int(cm_list[0][0]), int(cm_list[0][1])
        fn, tp = int(cm_list[1][0]), int(cm_list[1][1])

    roc_obj = raw.get("roc", {})
    fpr = roc_obj.get("fpr", raw.get("fpr", [0.0, 1.0]))
    tpr = roc_obj.get("tpr", raw.get("tpr", [0.0, 1.0]))
    auc = roc_obj.get("auc", raw.get("auc", 0.0))

    metrics_cfg = config.get("metrics", {})
    accuracy = float(raw.get("accuracy", metrics_cfg.get("Accuracy", 0.0)))
    precision = float(raw.get("precision", metrics_cfg.get("Precision", 0.0)))
    recall = float(raw.get("recall", metrics_cfg.get("Recall", 0.0)))
    f1 = float(raw.get("f1", metrics_cfg.get("F1", 0.0)))

    n_samples = int(raw.get("n_samples", raw.get("n_test", 0)))

    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": list(fpr),
        "tpr": list(tpr),
        "auc": float(auc),
        "n_samples": n_samples,
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
    }


eval_data = _normalize_eval(eval_raw)

# Test set ground-truth & probabilitas prediksi (untuk recompute live)
Y_TEST_TRUE = np.asarray(eval_raw.get("y_true", []), dtype=int)
Y_TEST_PROBA = np.asarray(eval_raw.get("y_pred_proba", []), dtype=float)
HAS_TEST_SET = len(Y_TEST_TRUE) > 0 and len(Y_TEST_PROBA) == len(Y_TEST_TRUE)


# ---------------------------------------------------------------------------
# Session-level history (in-memory, thread-safe)
# ---------------------------------------------------------------------------
SESSION_HISTORY: list[dict] = []
_history_lock = threading.Lock()


def _push_history(proba: float, true_label, payload: dict) -> None:
    with _history_lock:
        SESSION_HISTORY.append({
            "proba": float(proba),
            "true_label": int(true_label) if true_label is not None else None,
            "predicted_label": int(proba >= THRESHOLD),
            "payload": payload,
        })


def _snapshot_history() -> list[dict]:
    with _history_lock:
        return list(SESSION_HISTORY)


# ---------------------------------------------------------------------------
# Definisi form
# ---------------------------------------------------------------------------
NUMERIC_FIELDS = [
    {"name": "ProductRelated", "label": "Halaman produk dilihat",
     "hint": "Jumlah halaman produk yang dibuka dalam sesi ini.",
     "min": 0, "max": 1000, "step": "1"},
    {"name": "ProductRelated_Duration", "label": "Waktu di halaman produk (detik)",
     "hint": "Total detik yang dihabiskan di halaman produk.",
     "min": 0, "max": 100000, "step": "1"},
    {"name": "BounceRates", "label": "Bounce rate",
     "hint": "Antara 0 dan 0,2. Contoh: 0,02 berarti 2%.",
     "min": 0, "max": 0.2, "step": "0.001"},
    {"name": "ExitRates", "label": "Exit rate",
     "hint": "Antara 0 dan 0,2. Contoh: 0,04 berarti 4%.",
     "min": 0, "max": 0.2, "step": "0.001"},
    {"name": "PageValues", "label": "Page value",
     "hint": "Rata-rata nilai halaman yang dikunjungi sebelum transaksi.",
     "min": 0, "max": 500, "step": "0.1"},
]

MONTH_LABELS = {
    "Jan": "Januari", "Feb": "Februari", "Mar": "Maret", "Apr": "April",
    "May": "Mei", "June": "Juni", "Jul": "Juli", "Aug": "Agustus",
    "Sep": "September", "Oct": "Oktober", "Nov": "November", "Dec": "Desember",
}
VISITOR_LABELS = {
    "New_Visitor": "Pengunjung baru",
    "Returning_Visitor": "Pengunjung kembali",
    "Other": "Lainnya",
}

MONTH_OPTIONS = [(m, MONTH_LABELS.get(m, m)) for m in config["months"]]
VISITOR_OPTIONS = [(v, VISITOR_LABELS.get(v, v)) for v in config["visitor_types"]]
WEEKEND_OPTIONS = [("0", "Hari kerja"), ("1", "Akhir pekan")]
ACTUAL_LABEL_OPTIONS = [("", "— pilih jika tahu hasil aslinya —"),
                       ("1", "Ya, benar-benar membeli"),
                       ("0", "Tidak jadi membeli")]

PRESETS = {
    "berminat": {"ProductRelated": "30", "ProductRelated_Duration": "1000",
                 "BounceRates": "0.01", "ExitRates": "0.03", "PageValues": "8",
                 "Month": "May", "VisitorType": "Returning_Visitor", "Weekend": "0",
                 "actual_label": "1"},
    "ragu": {"ProductRelated": "3", "ProductRelated_Duration": "80",
             "BounceRates": "0.08", "ExitRates": "0.12", "PageValues": "0",
             "Month": "Feb", "VisitorType": "Returning_Visitor", "Weekend": "1",
             "actual_label": "0"},
}


# ---------------------------------------------------------------------------
# Logika prediksi
# ---------------------------------------------------------------------------
def parse_form(form):
    values, errors = {}, {}

    for f in NUMERIC_FIELDS:
        raw = (form.get(f["name"]) or "").strip().replace(",", ".")
        if raw == "":
            errors[f["name"]] = "Wajib diisi."
            continue
        try:
            num = float(raw)
        except ValueError:
            errors[f["name"]] = "Masukkan angka."
            continue
        if not (f["min"] <= num <= f["max"]):
            errors[f["name"]] = f"Isi antara {f['min']} dan {f['max']}."
            continue
        values[f["name"]] = num

    month = form.get("Month", "")
    if month not in config["months"]:
        errors["Month"] = "Pilih bulan."
    else:
        values["Month"] = month

    visitor = form.get("VisitorType", "")
    if visitor not in config["visitor_types"]:
        errors["VisitorType"] = "Pilih tipe pengunjung."
    else:
        values["VisitorType"] = visitor

    weekend = form.get("Weekend", "")
    if weekend not in ("0", "1"):
        errors["Weekend"] = "Pilih hari kunjungan."
    else:
        values["Weekend"] = int(weekend)

    actual_raw = (form.get("actual_label") or "").strip()
    if actual_raw == "":
        values["actual_label"] = None
    elif actual_raw in ("0", "1"):
        values["actual_label"] = int(actual_raw)
    else:
        errors["actual_label"] = "Pilih ya / tidak, atau biarkan kosong."

    return values, errors


def build_features(values):
    row = {c: 0.0 for c in columns}
    for c in num_cols:
        row[c] = defaults[c]
    for f in NUMERIC_FIELDS:
        row[f["name"]] = values[f["name"]]
    row["Weekend"] = values["Weekend"]
    for prefix in ("Month", "VisitorType"):
        col = f"{prefix}_{values[prefix]}"
        if col in row:
            row[col] = 1.0
    df_in = pd.DataFrame([row], columns=columns)
    df_in[num_cols] = scaler.transform(df_in[num_cols])
    return df_in


def predict(values):
    proba = float(model.predict_proba(build_features(values))[0, 1])
    will_buy = proba >= THRESHOLD
    _push_history(proba, values.get("actual_label"), payload={
        "ProductRelated": values.get("ProductRelated"),
        "PageValues": values.get("PageValues"),
        "Month": values.get("Month"),
        "VisitorType": values.get("VisitorType"),
        "Weekend": values.get("Weekend"),
    })
    return {
        "will_buy": will_buy,
        "label": "Kemungkinan besar membeli" if will_buy else "Kemungkinan tidak membeli",
        "probability": round(proba * 100, 1),
        "threshold": round(THRESHOLD * 100, 1),
        "advice": (
            "Pengunjung ini sudah condong membeli. Permudah proses checkout dan hindari diskon besar yang tidak perlu."
            if will_buy else
            "Pengunjung ini belum condong membeli. Pertimbangkan voucher, pengingat keranjang, atau rekomendasi produk."
        ),
    }


# ---------------------------------------------------------------------------
# Hitung ulang metrik live (test set asli + history sesi)
# ---------------------------------------------------------------------------
def _metrics_from_arrays(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true hanya berisi satu kelas, ROC tidak terdefinisi.")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        fpr, tpr, auc = [0.0, 1.0], [0.0, 1.0], float("nan")
    return {
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": auc,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
    }


def compute_live_metrics() -> dict:
    history = _snapshot_history()
    labeled = [h for h in history if h["true_label"] is not None]

    y_true_list: list[int] = []
    y_proba_list: list[float] = []

    if HAS_TEST_SET:
        y_true_list.extend(Y_TEST_TRUE.tolist())
        y_proba_list.extend(Y_TEST_PROBA.tolist())

    for h in labeled:
        y_true_list.append(h["true_label"])
        y_proba_list.append(h["proba"])

    if not y_true_list:
        return {
            "confusion_matrix": [[0, 0], [0, 0]],
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": float("nan"),
            "n_samples": 0, "n_positive": 0, "n_negative": 0,
            "n_session": len(history),
            "n_session_labeled": len(labeled),
            "source": "empty",
        }

    y_true = np.asarray(y_true_list, dtype=int)
    y_proba = np.asarray(y_proba_list, dtype=float)
    out = _metrics_from_arrays(y_true, y_proba)
    out["n_session"] = len(history)
    out["n_session_labeled"] = len(labeled)
    out["source"] = "test_set+session" if HAS_TEST_SET else "session_only"
    return out


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    form_values, errors, result = {}, {}, None

    if request.method == "POST":
        form_values = request.form.to_dict()
        clean, errors = parse_form(request.form)
        if not errors:
            result = predict(clean)

    live_metrics = compute_live_metrics()

    return render_template(
        "index.html",
        numeric_fields=NUMERIC_FIELDS,
        month_options=MONTH_OPTIONS,
        visitor_options=VISITOR_OPTIONS,
        weekend_options=WEEKEND_OPTIONS,
        actual_label_options=ACTUAL_LABEL_OPTIONS,
        presets=PRESETS,
        form=form_values,
        errors=errors,
        result=result,
        config=config,
        ev=eval_data,
        ev_live=live_metrics,
        threshold=THRESHOLD,
        has_test_set=HAS_TEST_SET,
        session_count=live_metrics["n_session"],
        session_labeled=live_metrics["n_session_labeled"],
    )


@app.route("/health")
def health():
    return jsonify(status="ok", model=config.get("model_name", "LogisticRegression"),
                   threshold=THRESHOLD)


@app.route("/reset", methods=["POST"])
def reset_session():
    with _history_lock:
        SESSION_HISTORY.clear()
    return jsonify(status="ok", message="Session history cleared.")


# Untuk Vercel: app di-export sebagai WSGI handler (otomatis terdeteksi)
# Untuk lokal: jalankan dengan `python app.py`
if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))