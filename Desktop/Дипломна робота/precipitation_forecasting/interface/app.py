import os
import sys
import glob
import pickle
import warnings
import json
import datetime
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─── ШЛЯХИ ────────────────────────────────────────────────────────────────────
BASE_DIR        = "/Users/darynanagaevskaya/Desktop/Дипломна робота/precipitation_forecasting"
DATA_DIR        = os.path.join(BASE_DIR, "data_preprocessing/processed")
SCALERS_DIR     = os.path.join(DATA_DIR, "scalers")
MODEL_PATH      = os.path.join(BASE_DIR, "models/transformer/best_model.pt")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models/classifier/best_classifier.json")
LOG_PATH        = os.path.join(BASE_DIR, "forecast_log.json")

PYTHON       = "/opt/anaconda3/bin/python"
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_collection/data_raw")

SCRIPT_UPDATE_DATA = os.path.join(BASE_DIR, "data_collection/update_data.py")
SCRIPT_PREPROCESS  = os.path.join(BASE_DIR, "data_preprocessing/preprocess_weather_data.py")
SCRIPT_REBUILD_SC  = os.path.join(BASE_DIR, "data_preprocessing/processed/rebuild_feature_scalers.py")

CITIES = ["Kyiv", "Lviv", "Odesa", "Dnipro", "Kharkiv", "Uzhhorod", "Chernihiv"]
CITIES_UA = {
    "Kyiv": "Київ", "Lviv": "Львів", "Odesa": "Одеса",
    "Dnipro": "Дніпро", "Kharkiv": "Харків",
    "Uzhhorod": "Ужгород", "Chernihiv": "Чернігів"
}
CITY_COORDS = {
    "Kyiv":      (50.45, 30.52), "Lviv":      (49.84, 24.03),
    "Odesa":     (46.48, 30.73), "Dnipro":    (48.46, 34.99),
    "Kharkiv":   (49.99, 36.23), "Uzhhorod":  (48.62, 22.29),
    "Chernihiv": (51.50, 31.29),
}

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UkrainePrecip · Прогноз опадів",
    page_icon="🌧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f4f8; color: #1e293b; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #64748b !important; font-size: 13px; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.page-title { font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.15rem; }
.page-subtitle { font-size: 13px; color: #94a3b8; letter-spacing:.08em; text-transform:uppercase; font-weight:500; margin-bottom:1.25rem; }
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:1.25rem 0; }
.metric-box {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
    padding:1.1rem 1rem; text-align:center; transition:box-shadow 0.2s,transform 0.2s;
}
.metric-box:hover { box-shadow:0 4px 16px rgba(0,0,0,0.08); transform:translateY(-2px); }
.metric-icon  { font-size:1.4rem; margin-bottom:.3rem; }
.metric-label { font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.25rem; }
.metric-value { font-family:'JetBrains Mono',monospace; font-size:1.55rem; font-weight:600; color:#1e293b; }
.metric-unit  { font-size:12px; color:#94a3b8; margin-left:2px; }
.weather-badge {
    display:inline-flex; align-items:center; gap:8px;
    padding:9px 20px; border-radius:100px; font-size:14px; font-weight:600; margin-bottom:1.1rem;
}
.badge-rain   { background:#eff6ff; border:1.5px solid #93c5fd; color:#1d4ed8; }
.badge-clear  { background:#fffbeb; border:1.5px solid #fcd34d; color:#92400e; }
.badge-traces { background:#f0fdf4; border:1.5px solid #86efac; color:#166534; }
.section-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:1.25rem 1.5rem; margin-bottom:1rem; }
.section-title { font-size:1rem; font-weight:600; color:#1e293b; margin-bottom:.75rem; }
[data-testid="stTabs"] [role="tablist"] {
    background:#ffffff; border-radius:12px; padding:4px; border:1px solid #e2e8f0; gap:2px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius:8px !important; color:#64748b !important; font-weight:500 !important;
    font-size:13px !important; padding:7px 16px !important; transition:all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background:#eff6ff !important; color:#2563eb !important; }
.stButton > button {
    background:#ffffff !important; border:1px solid #e2e8f0 !important;
    color:#374151 !important; border-radius:10px !important; font-weight:500 !important; transition:all 0.2s !important;
}
.stButton > button:hover {
    background:#f8fafc !important; border-color:#93c5fd !important;
    color:#2563eb !important; box-shadow:0 2px 8px rgba(37,99,235,0.1) !important;
}
[data-testid="stSelectbox"] > div > div {
    background:#ffffff !important; border-color:#e2e8f0 !important; color:#1e293b !important; border-radius:10px !important;
}
[data-testid="stExpander"] { background:#ffffff !important; border:1px solid #e2e8f0 !important; border-radius:12px !important; }
.log-row {
    display:flex; align-items:center; padding:10px 14px; border-radius:10px;
    margin-bottom:6px; background:#f8fafc; border:1px solid #e2e8f0;
    font-size:13px; gap:12px; transition:background 0.15s;
}
.log-row:hover { background:#eff6ff; border-color:#bfdbfe; }
.log-time  { color:#94a3b8; font-family:'JetBrains Mono',monospace; font-size:11px; min-width:135px; }
.log-city  { color:#2563eb; font-weight:600; min-width:75px; }
.log-val   { color:#374151; }
.log-rain  { color:#2563eb; }
.log-clear { color:#d97706; }
/* Шкала інтенсивності */
.intensity-row { display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid #f1f5f9; }
.intensity-dot { width:12px; height:12px; border-radius:50%; flex-shrink:0; }
.intensity-bar-wrap { flex:1; height:6px; background:#f1f5f9; border-radius:4px; overflow:hidden; }
.intensity-bar-fill { height:100%; border-radius:4px; }
.intensity-name  { color:#1e293b; min-width:120px; font-size:13px; font-weight:500; }
.intensity-range { font-family:'JetBrains Mono',monospace; color:#64748b; font-size:11px; min-width:90px; text-align:right; }
/* Вивід скрипту */
.script-success { background:#f0fdf4; border:1px solid #86efac; border-radius:12px; padding:1rem 1.25rem; margin-bottom:1rem; }
.script-error   { background:#fef2f2; border:1px solid #fca5a5; border-radius:12px; padding:1rem 1.25rem; margin-bottom:1rem; }
.script-header  { font-weight:600; font-size:14px; margin-bottom:.5rem; }
.script-cities  { display:flex; flex-wrap:wrap; gap:6px; margin-top:.5rem; }
.city-chip { background:#dbeafe; color:#1d4ed8; border-radius:20px; padding:3px 10px; font-size:12px; font-weight:500; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#f1f5f9; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:4px; }
@media (max-width:768px) { .metric-grid { grid-template-columns:repeat(2,1fr); } }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ШКАЛА ІНТЕНСИВНОСТІ
#  Єдине місце визначення — використовується скрізь у коді
# ══════════════════════════════════════════════════════════════════════════════
#
#  Категорії та пороги (мм/год):
#  0.000              → Без опадів
#  0.000 – 0.100      → Сліди опадів   (роса, туман)
#  0.100 – 0.500      → Мікроопади     (ледь помітний дощ)
#  0.500 – 2.500      → Слабкий дощ
#  2.500 – 7.500      → Помірний дощ
#  7.500 – 15.000     → Сильний дощ
#  > 15.000           → Злива
#
#  Поріг "година з опадами" = 0.1 мм/год (той самий що в класифікаторі)

INTENSITY_LEVELS = [
    # (верхній поріг, назва, колір крапки, колір бару)
    (0.0,   "Без опадів",    "#e2e8f0", "#94a3b8"),
    (0.1,   "Сліди опадів",  "#bfdbfe", "#60a5fa"),
    (0.5,   "Мікроопади",    "#93c5fd", "#3b82f6"),
    (2.5,   "Слабкий дощ",   "#60a5fa", "#2563eb"),
    (7.5,   "Помірний дощ",  "#3b82f6", "#1d4ed8"),
    (15.0,  "Сильний дощ",   "#1d4ed8", "#1e3a8a"),
    (9999,  "Злива",         "#1e3a8a", "#0f172a"),
]
RAIN_THRESHOLD = 0.1   # мм/год — поріг для підрахунку "годин з опадами"

def intensity_info(mm):
    for threshold, label, dot_col, bar_col in INTENSITY_LEVELS:
        if mm <= threshold:
            return label, dot_col, bar_col
    return "Злива", "#1e3a8a", "#0f172a"

def intensity_label(mm): return intensity_info(mm)[0]
def intensity_color(mm): return intensity_info(mm)[1]


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАВАНТАЖЕННЯ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_classifier(classifier_path):
    from xgboost import XGBClassifier
    json_path = classifier_path if classifier_path.endswith(".json") else classifier_path.replace(".pkl", ".json")
    if not os.path.exists(json_path):
        return None, None
    clf = XGBClassifier()
    clf.load_model(json_path)
    return clf, None


@st.cache_resource(show_spinner=False)
def load_transformer(model_path):
    import torch
    import torch.nn as nn

    class PrecipitationTransformer(nn.Module):
        def __init__(self, n_features, pred_len, cfg):
            super().__init__()
            d = cfg["d_model"]
            self.input_proj = nn.Sequential(nn.Linear(n_features, d), nn.LayerNorm(d), nn.ReLU())
            self.cls_token  = nn.Parameter(torch.randn(1, 1, d))
            self.pos_emb    = nn.Embedding(cfg["seq_len"] + 10, d)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg["n_heads"], dim_feedforward=cfg["d_ff"],
                dropout=cfg["dropout"], batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg["n_layers"])
            self.regression_head = nn.Sequential(
                nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(cfg["dropout"]),
                nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(cfg["dropout"]),
                nn.Linear(d, pred_len),
            )
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, x):
            B, T, _ = x.shape
            x   = self.input_proj(x)
            pos = torch.arange(T, device=x.device)
            x   = x + self.pos_emb(pos).unsqueeze(0)
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)
            x   = self.encoder(x)
            return self.regression_head(x[:, 0])

    device = torch.device("cpu")
    ckpt   = torch.load(model_path, map_location=device)
    cfg    = ckpt["config"]
    model  = PrecipitationTransformer(ckpt["n_features"], cfg["pred_len"], cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, device, torch


@st.cache_resource(show_spinner=False)
def load_scalers(scalers_dir, city):
    feat_path = os.path.join(scalers_dir, f"{city}_feature_scaler.pkl")
    tgt_path  = os.path.join(scalers_dir, f"{city}_target_scaler.pkl")
    feat_scaler, feat_cols = None, None
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            d = pickle.load(f)
        feat_scaler, feat_cols = d["scaler"], d["columns"]
    else:
        for p in glob.glob(os.path.join(scalers_dir, "*_feature_scaler.pkl")):
            with open(p, "rb") as f:
                d = pickle.load(f)
            feat_scaler, feat_cols = d["scaler"], d["columns"]
            break
    tgt_scaler = None
    if os.path.exists(tgt_path):
        with open(tgt_path, "rb") as f:
            tgt_scaler = pickle.load(f)
    else:
        for p in glob.glob(os.path.join(scalers_dir, "*_target_scaler.pkl")):
            with open(p, "rb") as f:
                tgt_scaler = pickle.load(f)
            break
    return feat_scaler, feat_cols, tgt_scaler


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАПУСК ЗОВНІШНІХ СКРИПТІВ
# ══════════════════════════════════════════════════════════════════════════════
def run_script(script_path: str, label: str, extra_args: list = None):
    if not os.path.exists(script_path):
        return False, f"Скрипт не знайдено: {script_path}"
    try:
        cmd = [PYTHON, script_path] + (extra_args or [])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, f"{label}: перевищено ліміт часу (10 хв)"
    except Exception as e:
        return False, str(e)


def parse_script_output(output: str, script_type: str) -> dict:
    """
    Розбирає вивід скрипту і повертає структуровані дані для красивого відображення.
    """
    lines  = output.splitlines()
    cities_found = [c for c in CITIES if any(c in l for l in lines)]
    rows_info = {}
    for line in lines:
        for city in CITIES:
            if city in line and ("train:" in line or "rows" in line.lower()):
                rows_info[city] = line.strip()
    errors = [l for l in lines if "error" in l.lower() or "❌" in l or "exception" in l.lower()]
    warnings = [l for l in lines if "⚠" in l or "warning" in l.lower()]
    return {
        "cities": cities_found,
        "rows_info": rows_info,
        "errors": errors,
        "warnings": warnings,
        "script_type": script_type,
        "raw": output,
    }


def render_script_result(title: str, ok: bool, output: str, script_type: str):
    """Красивий вивід результату виконання скрипту."""
    parsed = parse_script_output(output, script_type)

    if ok:
        st.markdown(f"""
        <div class="script-success">
          <div class="script-header" style="color:#166534;">✅ {title} — виконано успішно</div>
        """, unsafe_allow_html=True)

        if script_type == "update":
            st.markdown(f"""
            <div style="font-size:13px;color:#374151;margin-bottom:.5rem;">
              Дані оновлено для {len(parsed['cities'])} міст:
            </div>
            <div class="script-cities">
              {''.join(f'<span class="city-chip">{CITIES_UA.get(c, c)}</span>' for c in parsed["cities"])}
            </div>
            """, unsafe_allow_html=True)

        elif script_type == "preprocess":
            st.markdown(f"""
            <div style="font-size:13px;color:#374151;margin-bottom:.5rem;">
              Препроцесинг виконано для {len(parsed['cities'])} міст. Scalers збережено.
            </div>
            <div class="script-cities">
              {''.join(f'<span class="city-chip">{CITIES_UA.get(c, c)}</span>' for c in parsed["cities"])}
            </div>
            """, unsafe_allow_html=True)

        elif script_type == "scalers":
            st.markdown(f"""
            <div style="font-size:13px;color:#374151;margin-bottom:.5rem;">
              Feature scalers перебудовано для {len(parsed['cities'])} міст.
            </div>
            <div class="script-cities">
              {''.join(f'<span class="city-chip">{CITIES_UA.get(c, c)}</span>' for c in parsed["cities"])}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if parsed["warnings"]:
            with st.expander(f"⚠ Попередження ({len(parsed['warnings'])})"):
                for w in parsed["warnings"]:
                    st.markdown(f"<div style='font-size:12px;color:#92400e;'>{w}</div>", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="script-error">
          <div class="script-header" style="color:#991b1b;">❌ {title} — помилка</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Деталі помилки", expanded=True):
            st.code(output or "Немає виводу", language="text")

    if ok:
        with st.expander("Повний вивід"):
            st.code(output, language="text")


# ══════════════════════════════════════════════════════════════════════════════
#  ЛОГУВАННЯ
# ══════════════════════════════════════════════════════════════════════════════
def load_log():
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_log(entry: dict):
    log = load_log()
    log.append(entry)
    log = log[-200:]
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  ІНФЕРЕНС
# ══════════════════════════════════════════════════════════════════════════════
def load_city_window(data_dir, city, seq_len):
    for split in ("test", "val", "train"):
        path = os.path.join(data_dir, f"all_cities_{split}.csv")
        if not os.path.exists(path):
            continue
        df  = pd.read_csv(path)
        sub = df[df["city"] == city].sort_values("datetime").reset_index(drop=True)
        if len(sub) >= seq_len:
            return sub.tail(seq_len).copy()
    raise FileNotFoundError(f"Недостатньо даних для {city}")


def build_classifier_features(X_norm):
    return np.concatenate([
        X_norm.mean(0), X_norm.std(0), X_norm.min(0), X_norm.max(0),
        X_norm[-1], X_norm[-3:].mean(0), X_norm[-6:].mean(0), X_norm[-12:].mean(0),
    ]).reshape(1, -1)


@st.cache_data(show_spinner=False, ttl=300)
def run_forecast(city):
    clf, _                             = load_classifier(CLASSIFIER_PATH)
    model, cfg, device, torch          = load_transformer(MODEL_PATH)
    feat_scaler, feat_cols, tgt_scaler = load_scalers(SCALERS_DIR, city)

    if tgt_scaler is None:
        return None, None, None, None

    window_df      = load_city_window(DATA_DIR, city, cfg["seq_len"])
    last_dt        = pd.to_datetime(window_df["datetime"].iloc[-1])
    forecast_hours = [last_dt + pd.Timedelta(hours=i + 1) for i in range(cfg["pred_len"])]

    X_raw  = window_df[feat_cols].values.astype(np.float32)
    X_raw  = np.nan_to_num(X_raw, nan=0.0)
    X_norm = feat_scaler.transform(X_raw).astype(np.float32) if feat_scaler else X_raw
    X_norm = np.clip(X_norm, -10, 10)

    clf_result = {"label": 1, "confidence": 0.5, "prob_rain": 0.5, "prob_no_rain": 0.5, "available": False}
    if clf is not None:
        X_flat = build_classifier_features(X_norm)
        n_exp  = getattr(clf, "n_features_in_", None)
        if n_exp:
            if X_flat.shape[1] > n_exp:   X_flat = X_flat[:, :n_exp]
            elif X_flat.shape[1] < n_exp: X_flat = np.pad(X_flat, ((0, 0), (0, n_exp - X_flat.shape[1])))
        proba = clf.predict_proba(X_flat)[0]
        clf_result = {
            "label":        int(clf.predict(X_flat)[0]),
            "confidence":   float(proba[1]),
            "prob_rain":    float(proba[1]),
            "prob_no_rain": float(proba[0]),
            "available":    True,
        }

    X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(X_t).squeeze(0).cpu().numpy()

    pred_log = tgt_scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    pred_mm  = np.maximum(np.expm1(pred_log), 0.0)

    if clf_result["available"] and clf_result["label"] == 0:
        pred_mm = np.zeros_like(pred_mm)

    return pred_mm, forecast_hours, clf_result, window_df


# ══════════════════════════════════════════════════════════════════════════════
#  ПІДРАХУНОК ГОДИН З ОПАДАМИ
#  Рахуємо кількість годин де прогнозоване значення >= RAIN_THRESHOLD (0.1 мм/год).
#  Це той самий поріг що використовує класифікатор при навчанні.
#  Результат — ціле число від 0 до 24.
# ══════════════════════════════════════════════════════════════════════════════
def count_rainy_hours(pred_mm: np.ndarray) -> int:
    return int((pred_mm >= RAIN_THRESHOLD).sum())


# ══════════════════════════════════════════════════════════════════════════════
#  ТЕКСТОВИЙ ПРОГНОЗ
# ══════════════════════════════════════════════════════════════════════════════
def generate_text_forecast(pred_mm, forecast_hours, clf_result, city_ua):
    total   = float(pred_mm.sum())
    max_h   = float(pred_mm.max())
    rainy   = count_rainy_hours(pred_mm)
    peak_i  = int(np.argmax(pred_mm))
    peak_t  = forecast_hours[peak_i].strftime("%H:%M")
    parts   = []

    # ── Стан класифікатора ────────────────────────────────────────────────────
    if clf_result.get("available"):
        c = clf_result["confidence"]
        if clf_result["label"] == 1:
            parts.append(
                f"🌧 Класифікатор оцінює ймовірність опадів на **{int(c*100)}%** "
                f"протягом наступних 24 годин у **{city_ua}**."
            )
        else:
            parts.append(
                f"☀️ Класифікатор оцінює ймовірність відсутності опадів на **{int((1-c)*100)}%** "
                f"**{int((1 - c)*100)}%** у **{city_ua}**."
            )

    # ── Загальний стан прогнозу ───────────────────────────────────────────────
    if total < 0.1:
        # Класифікатор може казати "буде дощ", але трансформер дає сліди —
        # показуємо чесно що кількісно опадів не очікується
        parts.append(
            "🌤 Transformer-модель прогнозує лише **сліди опадів або їх відсутність** "
            f"(загальна сума менша за 0.1 мм). Помітних опадів не очікується."
        )
    elif rainy == 0:
        parts.append(
            f"🌦 Загальна сума опадів: **{total:.2f} мм**, проте інтенсивність "
            f"у кожній окремій годині нижча за поріг {RAIN_THRESHOLD} мм/год. "
            f"Опади розподілені рівномірно і дуже слабкі."
        )
    else:
        label = intensity_label(max_h)
        parts.append(
            f"📊 Загальна сума опадів за 24 год: **{total:.2f} мм**. "
            f"Максимальна інтенсивність — **{max_h:.3f} мм/год** ({label}) о **{peak_t}**. "
            f"Годин з опадами ≥ {RAIN_THRESHOLD} мм/год: **{rainy} із 24**."
        )

    # ── По відрізках доби ─────────────────────────────────────────────────────
    if rainy > 0:
        for hours_p, vals_p, p_label in [
            (forecast_hours[:8],   pred_mm[:8],   "Перші 8 годин"),
            (forecast_hours[8:16], pred_mm[8:16], "Середина прогнозу"),
            (forecast_hours[16:],  pred_mm[16:],  "Завершення"),
        ]:
            rain_h = [(h, v) for h, v in zip(hours_p, vals_p) if v >= RAIN_THRESHOLD]
            if rain_h:
                peak_v = max(v for _, v in rain_h)
                peak_h = [h for h, v in rain_h if v == peak_v][0]
                times  = [h.strftime("%H:%M") for h, _ in rain_h]
                parts.append(
                    f"🕐 **{p_label}**: опади о {', '.join(times[:4])}"
                    f"{'...' if len(times) > 4 else ''}. "
                    f"Пік — {peak_v:.3f} мм/год о {peak_h.strftime('%H:%M')}."
                )

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "Kyiv"
if "script_result" not in st.session_state:
    st.session_state["script_result"] = None   # (title, ok, output, script_type)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:.75rem 0 .5rem;'>
      <div style='font-size:1.3rem;font-weight:700;color:#1e293b;'>🌧 UkrainePrecip</div>
      <div style='font-size:11px;color:#94a3b8;letter-spacing:.1em;text-transform:uppercase;margin-top:3px;'>AI · Прогноз опадів</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    selected_city = st.selectbox(
        "🏙 Місто",
        options=CITIES,
        index=CITIES.index(st.session_state["selected_city"]),
        format_func=lambda c: CITIES_UA[c],
    )
    st.session_state["selected_city"] = selected_city
    st.divider()

    # ── Оновлення даних ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:#374151;margin-bottom:6px;'>"
        "⚙️ Оновлення даних</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:11px;color:#94a3b8;margin-bottom:8px;'>"
        "Запускати послідовно: спочатку 1, потім 2, потім 3</div>",
        unsafe_allow_html=True,
    )

    if st.button("📡 Завантажити нові дані", use_container_width=True):
        with st.spinner("Завантаження даних з WeatherAPI..."):
            ok, out = run_script(SCRIPT_UPDATE_DATA, "Оновлення даних")
        st.session_state["script_result"] = ("Завантаження нових даних", ok, out, "update")
        st.rerun()

    if st.button("🔧 Обробити дані", use_container_width=True):
        with st.spinner("Препроцесинг даних..."):
            ok, out = run_script(
                SCRIPT_PREPROCESS, "Препроцесинг",
                extra_args=[DATA_RAW_DIR, DATA_DIR],
            )
        st.session_state["script_result"] = ("Препроцесинг даних", ok, out, "preprocess")
        st.rerun()

    if st.button("📐 Масштабування", use_container_width=True):
        with st.spinner("Перебудова scalers..."):
            ok, out = run_script(SCRIPT_REBUILD_SC, "Rebuild scalers")
        st.session_state["script_result"] = ("Перебудова scalers", ok, out, "scalers")
        st.rerun()

    st.divider()

    if st.button("🔄 Оновити прогноз", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:12px;color:#64748b;line-height:1.9;'>
    <b>Модель</b><br>Transformer + XGBoost<br><br>
    <b>Горизонт прогнозу</b><br>24 години<br><br>
    <b>Поріг опадів</b><br>0.1 мм/год<br><br>
    <b>Міста</b><br>Київ · Львів · Одеса · Дніпро<br>Харків · Ужгород · Чернігів
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ВИВІД РЕЗУЛЬТАТУ СКРИПТУ
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["script_result"]:
    title, ok, output, stype = st.session_state["script_result"]
    render_script_result(title, ok, output, stype)
    if st.button("✕ Закрити повідомлення"):
        st.session_state["script_result"] = None
        st.rerun()
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="page-title">Прогноз опадів</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Ukraine · AI · Transformer + XGBoost · 24h</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ВИБІР МІСТА + КАРТА
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🗺 Оберіть місто</div>', unsafe_allow_html=True)

cols_city = st.columns(len(CITIES))
for i, city in enumerate(CITIES):
    with cols_city[i]:
        is_active = (city == st.session_state["selected_city"])
        label = f"● {CITIES_UA[city]}" if is_active else CITIES_UA[city]
        if st.button(label, key=f"city_btn_{city}", use_container_width=True):
            st.session_state["selected_city"] = city
            st.rerun()

selected_city = st.session_state["selected_city"]

is_sel = [c == selected_city for c in CITIES]
fig_map = go.Figure()
fig_map.add_trace(go.Scattergeo(
    lat=[CITY_COORDS[c][0] for c in CITIES],
    lon=[CITY_COORDS[c][1] for c in CITIES],
    mode="markers+text",
    marker=dict(
        size=[20 if s else 12 for s in is_sel],
        color=["#2563eb" if s else "#bfdbfe" for s in is_sel],
        line=dict(color=["#1d4ed8" if s else "#93c5fd" for s in is_sel],
                  width=[2 if s else 1 for s in is_sel]),
    ),
    text=[CITIES_UA[c] for c in CITIES],
    textposition="top center",
    textfont=dict(size=[13 if s else 11 for s in is_sel],
                  color=["#1d4ed8" if s else "#64748b" for s in is_sel]),
    hovertemplate="<b>%{text}</b><extra></extra>",
    name="",
))
fig_map.update_geos(
    scope="europe", center=dict(lat=49.0, lon=31.5), projection_scale=5.5,
    showland=True, landcolor="#f8fafc", showocean=True, oceancolor="#e0f2fe",
    showlakes=True, lakecolor="#bae6fd", showcountries=True, countrycolor="#cbd5e1",
    showcoastlines=True, coastlinecolor="#94a3b8", bgcolor="rgba(0,0,0,0)",
    showframe=False, showrivers=True, rivercolor="#bae6fd",
)
fig_map.update_layout(
    height=290, margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
)
st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  ПЕРЕВІРКА ШЛЯХІВ
# ══════════════════════════════════════════════════════════════════════════════
if not (os.path.exists(MODEL_PATH) and os.path.exists(DATA_DIR)):
    st.warning("⚠️ Шляхи до моделей або даних не знайдено. Перевірте `BASE_DIR`.")
    st.info(f"Очікуваний шлях: `{MODEL_PATH}`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  ПРОГНОЗ
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"Обчислення прогнозу для {CITIES_UA[selected_city]}..."):
    try:
        pred_mm, forecast_hours, clf_result, window_df = run_forecast(selected_city)
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Помилка: {e}")
        st.stop()

if pred_mm is None:
    st.error("❌ Target scaler не знайдено — прогноз неможливий")
    st.stop()

city_ua       = CITIES_UA[selected_city]
total         = float(pred_mm.sum())
max_h         = float(pred_mm.max())          # точне значення без округлення
rainy         = count_rainy_hours(pred_mm)    # ціле число, поріг 0.1 мм/год
peak_i        = int(np.argmax(pred_mm))
hours_str     = [dt.strftime("%H:%M") for dt in forecast_hours]
forecast_date = forecast_hours[0].strftime("%d.%m.%Y")

# Логування
save_log({
    "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
    "city":           selected_city,
    "city_ua":        city_ua,
    "total_mm":       round(total, 3),
    "max_mm":         round(max_h, 3),
    "rainy_hours":    rainy,
    "clf_label":      clf_result.get("label"),
    "clf_confidence": round(clf_result.get("confidence", 0), 3),
})

# ── Заголовок + бейдж ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style='font-size:1.75rem;font-weight:700;color:#1e293b;margin-bottom:.4rem;'>
  {city_ua}
  <span style='font-size:1rem;color:#94a3b8;font-weight:400;'> · прогноз на {forecast_date}</span>
</div>
""", unsafe_allow_html=True)

# Бейдж: якщо total < 0.1 — "Сліди опадів" незалежно від класифікатора
if total < 0.1:
    st.markdown(
        '<span class="weather-badge badge-traces">🌤 Сліди опадів або відсутні</span>',
        unsafe_allow_html=True,
    )
elif clf_result["available"]:
    c = clf_result["confidence"]
    if clf_result["label"] == 1:
        st.markdown(
            f'<span class="weather-badge badge-rain">🌧 Очікуються опади · {int(c*100)}%</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="weather-badge badge-clear">☀️ Без опадів · {int((1-c)*100)}%</span>',
            unsafe_allow_html=True,
        )
else:
    st.info("⚠ Класифікатор недоступний")

# ── Метрики ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-box">
    <div class="metric-icon">💧</div>
    <div class="metric-label">Сума опадів</div>
    <div class="metric-value">{total:.2f}<span class="metric-unit"> мм</span></div>
  </div>
  <div class="metric-box">
    <div class="metric-icon">⚡</div>
    <div class="metric-label">Максимум/год</div>
    <div class="metric-value">{max_h:.3f}<span class="metric-unit"> мм</span></div>
  </div>
  <div class="metric-box">
    <div class="metric-icon">🕐</div>
    <div class="metric-label">Годин з опадами</div>
    <div class="metric-value">{rainy}<span class="metric-unit"> / 24</span></div>
  </div>
  <div class="metric-box">
    <div class="metric-icon">📅</div>
    <div class="metric-label">Дата прогнозу</div>
    <div class="metric-value" style="font-size:1.1rem;">{forecast_date}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  ВКЛАДКИ
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Графіки",
    "📝 Текстовий прогноз",
    "🗃 Таблиця даних",
    "📜 Журнал прогнозів",
])


# ─── TAB 1 · ГРАФІКИ ──────────────────────────────────────────────────────────
with tab1:
    colors = [intensity_color(v) for v in pred_mm]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=hours_str, y=pred_mm,
        marker=dict(color=colors, opacity=0.9, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>%{y:.3f} мм/год<extra></extra>",
    ))
    fig1.add_hline(
        y=RAIN_THRESHOLD, line_dash="dot", line_color="#94a3b8",
        annotation_text=f"поріг {RAIN_THRESHOLD} мм/год",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#94a3b8"),
    )
    fig1.update_layout(
        title=dict(text=f"Опади по годинах — {city_ua} ({forecast_date})",
                   font=dict(size=16, color="#1e293b")),
        xaxis=dict(title="Час", tickangle=-45, tickfont=dict(size=10, color="#64748b"),
                   gridcolor="#f1f5f9"),
        yaxis=dict(title="мм/год", tickfont=dict(size=10, color="#64748b"),
                   gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=55, b=70), height=360, showlegend=False,
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    col_a, col_b = st.columns([2, 1])
    with col_a:
        cumsum = np.cumsum(pred_mm)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hours_str, y=cumsum, fill="tozeroy",
            line=dict(color="#2563eb", width=2.5),
            fillcolor="rgba(37,99,235,0.08)",
            hovertemplate="<b>%{x}</b><br>Накопичено: %{y:.3f} мм<extra></extra>",
        ))
        fig2.update_layout(
            title=dict(text="Кумулятивна сума опадів", font=dict(size=14, color="#1e293b")),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#64748b"), gridcolor="#f1f5f9"),
            yaxis=dict(title="мм (накопичено)", tickfont=dict(size=9, color="#64748b"),
                       gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=50, r=10, t=45, b=65), height=270, showlegend=False,
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        cats = {}
        for v in pred_mm:
            lbl = intensity_label(v)
            cats[lbl] = cats.get(lbl, 0) + 1
        level_colors = {lbl: dot for _, lbl, dot, _ in INTENSITY_LEVELS}
        fig3 = go.Figure(go.Pie(
            labels=list(cats.keys()),
            values=list(cats.values()),
            hole=0.55,
            marker=dict(
                colors=[level_colors.get(l, "#e2e8f0") for l in cats.keys()],
                line=dict(color="#ffffff", width=2),
            ),
            textfont=dict(size=11, color="#1e293b"),
            hovertemplate="<b>%{label}</b><br>%{value} год<extra></extra>",
        ))
        fig3.update_layout(
            title=dict(text="Розподіл", font=dict(size=14, color="#1e293b")),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=45, b=10), height=270, showlegend=False,
            annotations=[dict(text="24<br>год", x=0.5, y=0.5, showarrow=False,
                              font=dict(size=15, color="#1e293b"))],
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Теплова карта
    st.markdown('<div class="section-title" style="margin-top:.5rem;">🌡 Теплова карта</div>',
                unsafe_allow_html=True)
    fig4 = go.Figure(go.Heatmap(
        z=pred_mm.reshape(1, -1), x=hours_str,
        colorscale=[
            [0.0,  "#f8fafc"], [0.05, "#dbeafe"], [0.15, "#93c5fd"],
            [0.35, "#3b82f6"], [0.65, "#1d4ed8"], [1.0,  "#1e3a8a"],
        ],
        showscale=True,
        colorbar=dict(title="мм/год", tickfont=dict(color="#64748b", size=10),
                      titlefont=dict(color="#64748b", size=11), outlinecolor="#e2e8f0"),
        hovertemplate="<b>%{x}</b><br>%{z:.3f} мм/год<extra></extra>",
    ))
    fig4.update_layout(
        height=110, margin=dict(l=10, r=80, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#64748b")),
        yaxis=dict(showticklabels=False),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})


# ─── TAB 2 · ТЕКСТОВИЙ ПРОГНОЗ ───────────────────────────────────────────────
with tab2:
    forecast_text = generate_text_forecast(pred_mm, forecast_hours, clf_result, city_ua)
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">📋 Аналітичний прогноз для {city_ua} на {forecast_date}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(forecast_text)

    st.divider()
    st.markdown("#### 🎨 Шкала інтенсивності опадів")
    st.markdown(
        f"<div style='font-size:12px;color:#64748b;margin-bottom:12px;'>"
        f"Година вважається «дощовою» якщо інтенсивність ≥ {RAIN_THRESHOLD} мм/год</div>",
        unsafe_allow_html=True,
    )

    # Таблиця відповідає INTENSITY_LEVELS — той самий список що використовується скрізь
    ranges = [
        "0 мм/год",
        "0 – 0.1 мм/год",
        "0.1 – 0.5 мм/год",
        "0.5 – 2.5 мм/год",
        "2.5 – 7.5 мм/год",
        "7.5 – 15 мм/год",
        "> 15 мм/год",
    ]
    pcts = [5, 10, 22, 38, 58, 78, 100]
    rows_html = ""
    for (_, name, dot_col, bar_col), rng, pct in zip(INTENSITY_LEVELS, ranges, pcts):
        rows_html += f"""
        <div class='intensity-row'>
          <div class='intensity-dot' style='background:{dot_col};border:1px solid #e2e8f0;'></div>
          <div class='intensity-name'>{name}</div>
          <div class='intensity-bar-wrap'>
            <div class='intensity-bar-fill' style='width:{pct}%;background:{bar_col};'></div>
          </div>
          <div class='intensity-range'>{rng}</div>
        </div>"""
    st.markdown(f"<div class='section-card'>{rows_html}</div>", unsafe_allow_html=True)


# ─── TAB 3 · ТАБЛИЦЯ ДАНИХ ───────────────────────────────────────────────────
with tab3:
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("#### 📋 Погодинний прогноз")
        df_out = pd.DataFrame({
            "Час":             hours_str,
            "Опади (мм/год)":  [f"{v:.3f}" for v in pred_mm],
            "Інтенсивність":   [intensity_label(v) for v in pred_mm],
            "Накопичено (мм)": [f"{v:.3f}" for v in np.cumsum(pred_mm)],
        })
        st.dataframe(df_out, use_container_width=True, hide_index=True, height=440)

    with col_t2:
        st.markdown("#### 📊 Зведена таблиця")
        df_summary = pd.DataFrame({
            "Показник": [
                "Дата прогнозу",
                "Загальна сума",
                "Максимум/год",
                "Пік о",
                f"Годин з опадами (≥{RAIN_THRESHOLD} мм)",
                "Середнє/год",
                "Медіана/год",
                "Категорія піку",
            ],
            "Значення": [
                forecast_date,
                f"{total:.3f} мм",
                f"{max_h:.3f} мм",        # точне значення
                hours_str[peak_i],
                f"{rainy} / 24",           # ціле число
                f"{float(pred_mm.mean()):.3f} мм",
                f"{float(np.median(pred_mm)):.3f} мм",
                intensity_label(max_h),
            ],
        })
        st.dataframe(df_summary, use_container_width=True, hide_index=True, height=310)
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Завантажити CSV", data=csv,
            file_name=f"forecast_{selected_city}_{forecast_hours[0].strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True,
        )


# ─── TAB 4 · ЖУРНАЛ ──────────────────────────────────────────────────────────
with tab4:
    log_data = load_log()
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("#### 📜 Журнал прогнозів")
    with col_h2:
        if st.button("🗑 Очистити", use_container_width=True):
            try:
                with open(LOG_PATH, "w") as f:
                    json.dump([], f)
                st.rerun()
            except Exception:
                pass

    if not log_data:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#94a3b8;'>
          <div style='font-size:2rem;'>📭</div>
          <div style='font-size:14px;margin-top:.5rem;'>Журнал порожній</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_log = pd.DataFrame(log_data)
        col_ls1, col_ls2, col_ls3 = st.columns(3)
        with col_ls1:
            st.metric("Всього запитів", len(df_log))
        with col_ls2:
            most_city = df_log["city_ua"].value_counts().idxmax() if len(df_log) else "—"
            st.metric("Найпопулярніше місто", most_city)
        with col_ls3:
            avg_total = df_log["total_mm"].mean() if "total_mm" in df_log.columns else 0
            st.metric("Середня сума опадів", f"{avg_total:.2f} мм")

        st.divider()

        rows_html = ""
        for entry in reversed(log_data[-50:]):
            ts      = entry.get("timestamp", "")[:19].replace("T", " ")
            c_ua    = entry.get("city_ua", entry.get("city", ""))
            total_e = entry.get("total_mm", 0)
            rainy_e = entry.get("rainy_hours", 0)
            conf_e  = entry.get("clf_confidence", 0)
            label_e = entry.get("clf_label", 1)
            badge   = "🌧" if label_e == 1 else "☀️"
            badge_cls = "log-rain" if label_e == 1 else "log-clear"
            pct = int(conf_e * 100 if label_e == 1 else (1 - conf_e) * 100)
            rows_html += f"""
            <div class='log-row'>
              <div class='log-time'>{ts}</div>
              <div class='log-city'>{c_ua}</div>
              <div class='log-val'>💧 {total_e:.2f} мм · 🕐 {rainy_e}/24 год</div>
              <div class='{badge_cls}' style='margin-left:auto;font-size:12px;'>{badge} {pct}%</div>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)

        if len(df_log) >= 3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📈 Динаміка запитів")
            fig_log = make_subplots(specs=[[{"secondary_y": True}]])
            fig_log.add_trace(go.Bar(
                x=df_log["timestamp"].str[:19], y=df_log["total_mm"],
                name="Сума опадів", marker_color="#bfdbfe", opacity=0.8,
                hovertemplate="<b>%{x}</b><br>%{y:.2f} мм<extra></extra>",
            ), secondary_y=False)
            fig_log.add_trace(go.Scatter(
                x=df_log["timestamp"].str[:19], y=df_log["rainy_hours"],
                name="Годин з дощем", line=dict(color="#2563eb", width=2),
                mode="lines+markers", marker=dict(size=5),
                hovertemplate="<b>%{x}</b><br>%{y} год<extra></extra>",
            ), secondary_y=True)
            fig_log.update_layout(
                height=220, plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=50, r=50, t=20, b=70),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#64748b"),
                           gridcolor="#f1f5f9"),
                yaxis=dict(title="мм", tickfont=dict(size=9, color="#64748b"),
                           gridcolor="#f1f5f9"),
                yaxis2=dict(title="год", tickfont=dict(size=9, color="#64748b")),
                legend=dict(font=dict(size=11, color="#374151"), bgcolor="rgba(0,0,0,0)"),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_log, use_container_width=True, config={"displayModeBar": False})

            if "city_ua" in df_log.columns and df_log["city_ua"].nunique() > 1:
                st.markdown("#### 🏙 Статистика по містах")
                city_stats = (
                    df_log.groupby("city_ua")
                    .agg(
                        Запитів=("total_mm", "count"),
                        Середня_сума=("total_mm", "mean"),
                        Макс_сума=("total_mm", "max"),
                    )
                    .reset_index()
                    .rename(columns={"city_ua": "Місто"})
                )
                city_stats["Середня_сума"] = city_stats["Середня_сума"].round(2)
                city_stats["Макс_сума"]    = city_stats["Макс_сума"].round(2)
                st.dataframe(city_stats, use_container_width=True, hide_index=True)