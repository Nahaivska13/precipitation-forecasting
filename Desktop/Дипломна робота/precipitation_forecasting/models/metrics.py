import os
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn

# ───────────────────────────────────────────────────────────────
# Шляхи
# ───────────────────────────────────────────────────────────────

BASE_DIR = "/Users/darynanagaevskaya/Desktop/Дипломна робота/precipitation_forecasting"
DATA_DIR = os.path.join(BASE_DIR, "data_preprocessing/processed")
SCALERS_DIR = os.path.join(DATA_DIR, "scalers")
MODEL_PATH = os.path.join(BASE_DIR, "models/transformer/best_model.pt")

# ───────────────────────────────────────────────────────────────
# Модель (точно як у тренуванні!)
# ───────────────────────────────────────────────────────────────

class PrecipitationTransformer(nn.Module):
    def __init__(self, n_features, pred_len, cfg):
        super().__init__()
        d = cfg["d_model"]

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d),
            nn.LayerNorm(d),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.pos_emb   = nn.Embedding(cfg["seq_len"] + 10, d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["d_ff"],
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg["n_layers"])

        self.regression_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d, pred_len),
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.regression_head(x[:, 0])

# ───────────────────────────────────────────────────────────────
# Завантаження scaler’ів
# ───────────────────────────────────────────────────────────────

def load_scalers(city):
    feat_path = os.path.join(SCALERS_DIR, f"{city}_feature_scaler.pkl")
    tgt_path  = os.path.join(SCALERS_DIR, f"{city}_target_scaler.pkl")

    with open(feat_path, "rb") as f:
        d = pickle.load(f)
        feat_scaler = d["scaler"]
        feat_cols   = d["columns"]

    with open(tgt_path, "rb") as f:
        tgt_scaler = pickle.load(f)

    return feat_scaler, feat_cols, tgt_scaler

# ───────────────────────────────────────────────────────────────
# Формування вікон
# ───────────────────────────────────────────────────────────────

def make_windows(df, seq_len, pred_len, feat_cols):
    X_list, y_list = [], []
    for i in range(len(df) - seq_len - pred_len):
        X = df.iloc[i:i+seq_len][feat_cols].values.astype(np.float32)
        y = df.iloc[i+seq_len:i+seq_len+pred_len]["precip_mm"].values.astype(np.float32)
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list), np.array(y_list)

# ───────────────────────────────────────────────────────────────
# Основна логіка
# ───────────────────────────────────────────────────────────────

def evaluate_global():
    print("\n=== Глобальна оцінка моделі по всій Україні ===")

    # 1. Завантаження моделі
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    cfg = ckpt["config"]
    model = PrecipitationTransformer(ckpt["n_features"], cfg["pred_len"], cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 2. Завантаження тестових даних
    df = pd.read_csv(os.path.join(DATA_DIR, "all_cities_test.csv"))
    cities = df["city"].unique()

    all_true = []
    all_pred = []

    # 3. Проходимо по всіх містах
    for city in cities:
        print(f" → Обробка міста: {city}")

        feat_scaler, feat_cols, tgt_scaler = load_scalers(city)

        df_city = df[df["city"] == city].sort_values("datetime").reset_index(drop=True)

        X_raw, y_true = make_windows(df_city, cfg["seq_len"], cfg["pred_len"], feat_cols)

        if len(X_raw) == 0:
            continue

        X_norm = feat_scaler.transform(X_raw.reshape(-1, len(feat_cols))).reshape(X_raw.shape)

        # Прогноз
        for x, y in zip(X_norm, y_true):
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_norm = model(x_t).squeeze(0).numpy()

            pred_log = tgt_scaler.inverse_transform(pred_norm.reshape(-1,1)).flatten()
            pred_mm  = np.maximum(np.expm1(pred_log), 0.0)

            all_true.extend(y)
            all_pred.extend(pred_mm)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # ───────────────────────────────────────────────────────────────
    # Метрики
    # ───────────────────────────────────────────────────────────────

    mse  = mean_squared_error(all_true, all_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(all_true, all_pred)
    r2   = r2_score(all_true, all_pred)

    # Baseline
    y_zero = np.zeros_like(all_true)
    mse0  = mean_squared_error(all_true, y_zero)
    rmse0 = np.sqrt(mse0)
    mae0  = mean_absolute_error(all_true, y_zero)
    r20   = r2_score(all_true, y_zero)

    print("\n=== METRICS (Transformer, Global) ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"R²   : {r2:.4f}")

    print("\n=== BASELINE (0 мм, Global) ===")
    print(f"MAE  : {mae0:.4f}")
    print(f"RMSE : {rmse0:.4f}")
    print(f"MSE  : {mse0:.4f}")
    print(f"R²   : {r20:.4f}")

    # ───────────────────────────────────────────────────────────────
    # Метрики тільки на дощових годинах
    # ───────────────────────────────────────────────────────────────

    rain_mask = all_true > 0.1  # поріг дощу

    if rain_mask.sum() > 0:
        mae_rain  = mean_absolute_error(all_true[rain_mask], all_pred[rain_mask])
        rmse_rain = np.sqrt(mean_squared_error(all_true[rain_mask], all_pred[rain_mask]))
    else:
        mae_rain = rmse_rain = float("nan")

    print("\n=== METRICS (Rain-only, y_true > 0.1 мм) ===")
    print(f"MAE_rain  : {mae_rain:.4f}")
    print(f"RMSE_rain : {rmse_rain:.4f}")

    # ───────────────────────────────────────────────────────────────
    # Класифікаційні метрики (дощ / не дощ)
    # ───────────────────────────────────────────────────────────────

    y_true_cls = (all_true > 0.1).astype(int)
    y_pred_cls = (all_pred > 0.1).astype(int)

    TP = np.sum((y_true_cls == 1) & (y_pred_cls == 1))
    FP = np.sum((y_true_cls == 0) & (y_pred_cls == 1))
    FN = np.sum((y_true_cls == 1) & (y_pred_cls == 0))
    TN = np.sum((y_true_cls == 0) & (y_pred_cls == 0))

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n=== CLASSIFICATION METRICS (Rain / No Rain) ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # ───────────────────────────────────────────────────────────────
    # Метеорологічні метрики
    # ───────────────────────────────────────────────────────────────

    # POD — Probability of Detection
    POD = TP / (TP + FN + 1e-9)

    # FAR — False Alarm Rate
    FAR = FP / (TP + FP + 1e-9)

    # CSI — Critical Success Index
    CSI = TP / (TP + FP + FN + 1e-9)

    print("\n=== METEOROLOGICAL METRICS ===")
    print(f"POD (Probability of Detection) : {POD:.4f}")
    print(f"FAR (False Alarm Rate)         : {FAR:.4f}")
    print(f"CSI (Critical Success Index)   : {CSI:.4f}")

    # ───────────────────────────────────────────────────────────────
    # Brier Score (якість прогнозу ймовірності)
    # ───────────────────────────────────────────────────────────────

    # ймовірність дощу = нормалізований прогноз інтенсивності
    prob_pred = np.clip(all_pred / (all_pred.max() + 1e-9), 0, 1)
    brier = np.mean((prob_pred - y_true_cls)**2)

    print("\n=== BRIER SCORE ===")
    print(f"Brier Score : {brier:.4f}")


# ───────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_global()
