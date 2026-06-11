"""
Прогноз опадів — inference скрипт
Запуск: python forecast.py [--city Kyiv] [--debug]
"""
import os
import sys
import glob
import argparse
import warnings
import pickle

warnings.filterwarnings("ignore")

# ─── ШЛЯХИ ────────────────────────────────────────────────────────────────────

BASE_DIR        = "/Users/darynanagaevskaya/Desktop/Дипломна робота/precipitation_forecasting"
DATA_DIR        = os.path.join(BASE_DIR, "data_preprocessing/processed")
SCALERS_DIR     = os.path.join(DATA_DIR, "scalers")
MODEL_PATH      = os.path.join(BASE_DIR, "models/transformer/best_model.pt")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models/classifier/best_classifier.json")

CITIES = ["Kyiv", "Lviv", "Odesa", "Dnipro", "Kharkiv", "Uzhhorod", "Chernihiv"]


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАВАНТАЖЕННЯ КЛАСИФІКАТОРА (викликається ПЕРШИМ — до імпорту torch/nn)
# ══════════════════════════════════════════════════════════════════════════════

def load_classifier(classifier_path):
    from xgboost import XGBClassifier
    import json as _json

    json_path = classifier_path
    if classifier_path.endswith(".pkl"):
        json_path = classifier_path.replace(".pkl", ".json")

    if not os.path.exists(json_path):
        print(f"  ⚠ Класифікатор не знайдено: {json_path}")
        return None, None

    clf = XGBClassifier()
    clf.load_model(json_path)

    meta_path = os.path.join(os.path.dirname(json_path), "classifier_meta.json")
    cfg = None
    if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
        try:
            with open(meta_path, "r") as f:
                cfg = _json.load(f)
        except _json.JSONDecodeError:
            pass

    print(f"  ✓ XGBoost класифікатор завантажено | очікує {clf.n_features_in_} ознак")
    return clf, cfg


# ══════════════════════════════════════════════════════════════════════════════
#  TORCH — імпортується ПІСЛЯ завантаження класифікатора
# ══════════════════════════════════════════════════════════════════════════════

def _import_torch():
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    return torch, nn, np, pd


def _build_transformer_class(torch, nn):
    class PrecipitationTransformer(nn.Module):
        def __init__(self, n_features, pred_len, cfg):
            super().__init__()
            d = cfg["d_model"]
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, d), nn.LayerNorm(d), nn.ReLU(),
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, d))
            self.pos_emb   = nn.Embedding(cfg["seq_len"] + 10, d)
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
            x = self.input_proj(x)
            pos = torch.arange(T, device=x.device)
            x = x + self.pos_emb(pos).unsqueeze(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.encoder(x)
            return self.regression_head(x[:, 0])

    return PrecipitationTransformer


def load_transformer(model_path, torch, nn):
    device = torch.device("cpu")
    ckpt   = torch.load(model_path, map_location=device)
    cfg    = ckpt["config"]
    Cls    = _build_transformer_class(torch, nn)
    model  = Cls(ckpt["n_features"], cfg["pred_len"], cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  ✓ Transformer завантажено | features={ckpt['n_features']}")
    return model, cfg, device


def load_feature_scaler(scalers_dir, city):
    path = os.path.join(scalers_dir, f"{city}_feature_scaler.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  ✓ Feature scaler: {city}")
        return data["scaler"], data["columns"]
    for p in glob.glob(os.path.join(scalers_dir, "*_feature_scaler.pkl")):
        with open(p, "rb") as f:
            data = pickle.load(f)
        print(f"  ⚠ Feature scaler fallback: {os.path.basename(p)}")
        return data["scaler"], data["columns"]
    print("  ❌ Feature scaler не знайдено!")
    return None, None


def load_target_scaler(scalers_dir, city):
    path = os.path.join(scalers_dir, f"{city}_target_scaler.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            sc = pickle.load(f)
        print(f"  ✓ Target scaler: {city}")
        return sc
    for p in glob.glob(os.path.join(scalers_dir, "*_target_scaler.pkl")):
        with open(p, "rb") as f:
            sc = pickle.load(f)
        print(f"  ⚠ Target scaler fallback: {os.path.basename(p)}")
        return sc
    print(f"  ❌ Target scaler для {city} не знайдено!")
    return None


def load_city_window(data_dir, city, seq_len, pd):
    for split in ("test", "val", "train"):
        path = os.path.join(data_dir, f"all_cities_{split}.csv")
        if not os.path.exists(path):
            continue
        df  = pd.read_csv(path)
        sub = df[df["city"] == city].sort_values("datetime").reset_index(drop=True)
        if len(sub) >= seq_len:
            window = sub.tail(seq_len).copy()
            print(f"  ✓ Дані: {window['datetime'].iloc[0]} → {window['datetime'].iloc[-1]}")
            return window
    raise FileNotFoundError(f"Немає достатньо даних для {city}")


def prepare_features(window_df, feature_scaler, feature_columns, np, debug=False):
    X_raw = window_df[feature_columns].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    if feature_scaler is not None:
        X_norm = feature_scaler.transform(X_raw).astype(np.float32)
        X_norm = np.clip(X_norm, -10, 10)
    else:
        X_norm = X_raw
    if debug:
        print(f"  [DEBUG] X_norm shape={X_norm.shape} mean={X_norm.mean():.4f}")
    return X_norm


def build_classifier_features(X_norm, np):
    return np.concatenate([
        X_norm.mean(0), X_norm.std(0), X_norm.min(0), X_norm.max(0),
        X_norm[-1], X_norm[-3:].mean(0), X_norm[-6:].mean(0), X_norm[-12:].mean(0),
    ]).reshape(1, -1)


def run_classifier(clf, X_norm, np, debug=False):
    if clf is None:
        return {"label": 1, "confidence": 0.5, "available": False}
    X_flat = build_classifier_features(X_norm, np)
    n_exp  = getattr(clf, "n_features_in_", None)
    if n_exp and X_flat.shape[1] != n_exp:
        X_flat = X_flat[:, :n_exp] if X_flat.shape[1] > n_exp else np.pad(X_flat, ((0,0),(0, n_exp - X_flat.shape[1])))
    label = int(clf.predict(X_flat)[0])
    proba = float(clf.predict_proba(X_flat)[0][1])
    if debug:
        print(f"  [DEBUG] XGBoost: label={label}, P(rain)={proba:.4f}")
    return {"label": label, "confidence": proba, "available": True}


def run_transformer(model, X_norm, device, torch, debug=False):
    X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(X_t).squeeze(0).cpu().numpy()
    if debug:
        print(f"  [DEBUG] pred_norm mean={pred_norm.mean():.4f}")
    return pred_norm


def denormalize(pred_norm, target_scaler, np, debug=False):
    if target_scaler is None:
        pred_mm = np.maximum(np.expm1(pred_norm), 0.0)
    else:
        pred_log = target_scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        pred_mm  = np.maximum(np.expm1(pred_log), 0.0)
    if debug:
        print(f"  [DEBUG] після denorm: mean={pred_mm.mean():.4f}, max={pred_mm.max():.4f}")
    return pred_mm


def apply_classifier_logic(pred_mm, clf_result, np):
    if clf_result["available"] and clf_result["label"] == 0:
        return np.zeros_like(pred_mm)
    return pred_mm


def intensity_label(mm):
    if mm < 0.005:  return " Без опадів"
    elif mm < 0.01: return " Вологість"
    elif mm < 0.05: return " Мікроопади"
    elif mm < 0.2:  return " Мряка"
    elif mm < 1.0:  return " Слабкий дощ"
    elif mm < 5.0:  return " Помірний дощ"
    elif mm < 10.0: return " Сильний дощ"
    else:           return " Злива"


# ══════════════════════════════════════════════════════════════════════════════
#  ОСНОВНА ФУНКЦІЯ
# ══════════════════════════════════════════════════════════════════════════════

def forecast(city, model_path=MODEL_PATH, classifier_path=CLASSIFIER_PATH,
             data_dir=DATA_DIR, scalers_dir=SCALERS_DIR, debug=False):

    print(f"\n{'='*60}")
    print(f"  Завантаження моделей та даних...")
    print(f"{'='*60}")

    # КРОК 1: XGBoost ПЕРШИМ (до torch.nn — уникаємо конфлікту OpenMP)
    clf, _ = load_classifier(classifier_path)

    # КРОК 2: torch імпортується тільки тепер
    torch, nn, np, pd = _import_torch()
    model, cfg, device = load_transformer(model_path, torch, nn)

    seq_len  = cfg["seq_len"]
    pred_len = cfg["pred_len"]

    feature_scaler, feature_columns = load_feature_scaler(scalers_dir, city)
    target_scaler = load_target_scaler(scalers_dir, city)

    if target_scaler is None:
        print("❌ Немає target scaler — прогноз неможливий")
        return

    try:
        window_df = load_city_window(data_dir, city, seq_len, pd)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    last_dt        = pd.to_datetime(window_df["datetime"].iloc[-1])
    forecast_hours = [last_dt + pd.Timedelta(hours=i+1) for i in range(pred_len)]

    X_norm     = prepare_features(window_df, feature_scaler, feature_columns, np, debug)
    clf_result = run_classifier(clf, X_norm, np, debug)
    pred_norm  = run_transformer(model, X_norm, device, torch, debug)
    pred_mm    = denormalize(pred_norm, target_scaler, np, debug)
    pred_final = apply_classifier_logic(pred_mm, clf_result, np)

    if clf_result["available"]:
        c = clf_result["confidence"]
        cls_str = (f"🌧  Очікуються опади ({int(c*100)}%)"
                   if clf_result["label"] == 1 else
                   f"☀️  Без опадів ({int((1-c)*100)}%)")
    else:
        cls_str = "⚠  Класифікатор недоступний"

    print(f"\n{'='*60}")
    print(f"  🌧  Прогноз опадів — {city}")
    print(f"{'='*60}")
    print(f"  Вхідний період : {window_df['datetime'].iloc[0]} → {window_df['datetime'].iloc[-1]}")
    print(f"  Прогноз на     : {forecast_hours[0].strftime('%Y-%m-%d')}")
    print(f"  Класифікатор   : {cls_str}")
    print(f"{'='*60}")
    print(f"  {'Час':<8} {'Опади (мм)':>12}   Інтенсивність")
    print(f"  {'-'*52}")

    for dt, mm in zip(forecast_hours, pred_final):
        print(f"  {dt.strftime('%H:%M'):<8} {mm:>8.3f} мм   {intensity_label(mm)}")

    total = pred_final.sum()
    max_h = pred_final.max()
    rainy = (pred_final >= 0.05).sum()

    print(f"\n  {'─'*52}")
    print(f"  📊 Підсумок за 24 години:")
    print(f"     Загальна кількість опадів : {total:.2f} мм")
    print(f"     Максимум за годину        : {max_h:.2f} мм")
    print(f"     Годин з опадами           : {rainy} / {pred_len}")
    print(f"{'='*60}\n")

    return pred_final, forecast_hours


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def select_city():
    print(f"\n{'='*60}")
    print(f"  🌍  Прогноз опадів — вибір міста")
    print(f"{'='*60}")
    for i, c in enumerate(CITIES, 1):
        print(f"  {i}. {c}")
    print(f"{'='*60}")
    while True:
        try:
            idx = int(input("  Введіть номер міста (1-7): ").strip()) - 1
            if 0 <= idx < len(CITIES):
                return CITIES[idx]
            print("  ❌ Невірний номер")
        except (ValueError, KeyboardInterrupt):
            print("\n  До побачення!")
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",            default=None, choices=CITIES)
    parser.add_argument("--model_path",      default=MODEL_PATH)
    parser.add_argument("--classifier_path", default=CLASSIFIER_PATH)
    parser.add_argument("--data_dir",        default=DATA_DIR)
    parser.add_argument("--scalers_dir",     default=SCALERS_DIR)
    parser.add_argument("--debug",           action="store_true")
    args = parser.parse_args()

    city = args.city or select_city()

    forecast(
        city=city,
        model_path=args.model_path,
        classifier_path=args.classifier_path,
        data_dir=args.data_dir,
        scalers_dir=args.scalers_dir,
        debug=args.debug,
    )