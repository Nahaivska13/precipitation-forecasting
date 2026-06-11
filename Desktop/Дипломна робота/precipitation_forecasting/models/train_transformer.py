"""
Transformer для прогнозування інтенсивності опадів  
Вхід  : 72 години погодних даних (26 ознак)
Вихід : precip_mm — інтенсивність опадів на 24 години вперед

Покращення v5:
  - Більша модель (d_model=128, n_heads=8, n_layers=4)
  - CLS токен замість global average pooling
  - Глибший regression head з GELU
  - Learning rate warmup
  - Менший batch_size для кращого навчання
  - Більший dropout для регуляризації
- Правильний scaler для кожного міста
"""

import os, argparse, json, pickle, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Конфігурація ─────────────────────────────────────────────────────────────
CONFIG = {
    "seq_len":       72,
    "pred_len":      24,
    "batch_size":    128,
    "d_model":       128,
    "n_heads":       8,
    "n_layers":      4,
    "d_ff":          256,
    "dropout":       0.2,
    "epochs":        100,
    "lr":            5e-5,
    "patience":      15,
    "weight_decay":  1e-4,
    "warmup_epochs": 5,
}

NON_FEATURE_COLS = {"datetime", "city", "precip_mm"}


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class WeatherDataset(Dataset):
    """
    Sliding window dataset.
    Кожен sample:
      X     : [seq_len, n_features]  — 72 години вхідних даних
      y_reg : [pred_len]             — нормалізований precip_mm (24 год)
    """
    def __init__(self, df, seq_len, pred_len):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in NON_FEATURE_COLS]

        df_feat = df[feat_cols].copy()
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        df_feat = df_feat.clip(-10, 10)

        self.features   = df_feat.values.astype(np.float32)
        self.target_reg = df["precip_mm"].values.astype(np.float32)

    def __len__(self):
        return max(0, len(self.features) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        X     = torch.tensor(self.features[idx : idx + self.seq_len])
        y_reg = torch.tensor(
            self.target_reg[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        )
        return X, y_reg


def load_datasets(data_dir, seq_len, pred_len, batch_size):
    loaders    = {}
    n_features = None
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"all_cities_{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не знайдено: {path}")
        df = (pd.read_csv(path)
                .sort_values(["city", "datetime"])
                .reset_index(drop=True))
        ds = WeatherDataset(df, seq_len, pred_len)
        if n_features is None:
            n_features = ds.features.shape[1]
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"), num_workers=0
        )
        print(f"  {split:5s} samples : {len(ds):,}")
    print(f"  Features      : {n_features}")
    return loaders, n_features


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PrecipitationTransformer(nn.Module):
    """
    Encoder-only Transformer з CLS токеном.

    Архітектура:
      Linear projection → CLS token + Learnable positional embedding
      → Transformer Encoder → CLS токен як представлення
      → Глибокий Regression head → [pred_len] значень
    """
    def __init__(self, n_features, pred_len, cfg):
        super().__init__()
        d = cfg["d_model"]

        # Проекція вхідних ознак у простір моделі
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d),
            nn.LayerNorm(d),
            nn.ReLU(),
        )

        # CLS токен — вчиться збирати глобальну інформацію
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))

        # Learnable positional embedding (+1 для CLS токена)
        self.pos_emb = nn.Embedding(cfg["seq_len"] + 10, d)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["d_ff"],
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg["n_layers"]
        )

        # Глибокий Regression head з GELU
        self.regression_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d, pred_len),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # CLS токен ініціалізуємо малими значеннями
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, seq_len, n_features]
        B, T, _ = x.shape

        # Проекція + positional embedding
        x   = self.input_proj(x)                           # [B, T, d]
        pos = torch.arange(T, device=x.device)
        x   = x + self.pos_emb(pos).unsqueeze(0)          # [B, T, d]

        # Додаємо CLS токен на початок
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, d]
        x   = torch.cat([cls, x], dim=1)                  # [B, T+1, d]

        # Encoder
        x   = self.encoder(x)                             # [B, T+1, d]

        # Беремо тільки CLS токен як глобальне представлення
        cls_out = x[:, 0]                                  # [B, d]

        return self.regression_head(cls_out)               # [B, pred_len]


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS  — MSE + MAE
# ══════════════════════════════════════════════════════════════════════════════

class RegressionLoss(nn.Module):
    """
    MSE + MAE комбінована втрата.
    MSE — штрафує великі помилки (важливо для сильних опадів).
    MAE — стабільніший до нулів (більшість годин без опадів).
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        l_mse = self.mse(pred, target)
        l_mae = self.mae(pred, target)
        total = (l_mse + l_mae) / 2
        return total, l_mse, l_mae


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total = mse_sum = mae_sum = 0.0
    n = n_skip = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y_reg in loader:
            X, y_reg = X.to(device), y_reg.to(device)

            pred = model(X)
            loss, l_mse, l_mae = criterion(pred, y_reg)

            if torch.isnan(loss):
                n_skip += 1
                if train:
                    optimizer.zero_grad()
                continue

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total   += loss.item()
            mse_sum += l_mse.item()
            mae_sum += l_mae.item()
            n       += 1

    if n_skip:
        print(f"    ⚠ Пропущено {n_skip} батчів через nan")
    n = max(n, 1)
    return total/n, mse_sum/n, mae_sum/n


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\n{'='*60}\n  Пристрій: {device}\n{'='*60}\n")

    print("Завантаження даних...")
    loaders, n_features = load_datasets(
        data_dir, cfg["seq_len"], cfg["pred_len"], cfg["batch_size"]
    )

    model     = PrecipitationTransformer(n_features, cfg["pred_len"], cfg).to(device)
    criterion = RegressionLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # ── Learning rate scheduler: warmup + ReduceLROnPlateau ───────────────
    warmup_epochs = cfg["warmup_epochs"]
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup_scheduler   = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    plateau_scheduler  = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nПараметрів моделі: {total_params:,}\n")

    best_val  = float("inf")
    patience  = 0
    history   = []

    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'MSE':>10} "
          f"{'MAE':>10} {'LR':>12}")
    print("-" * 62)

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, _, _          = run_epoch(model, loaders["train"],
                                           criterion, optimizer, device, True)
        val_loss, v_mse, v_mae = run_epoch(model, loaders["val"],
                                           criterion, optimizer, device, False)

        # Warmup перші N епох, потім ReduceLROnPlateau
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": tr_loss,
            "val_loss": val_loss, "val_mse": v_mse, "val_mae": v_mae,
        })

        print(f"{epoch:>6} {tr_loss:>10.4f} {val_loss:>10.4f} "
              f"{v_mse:>10.4f} {v_mae:>10.4f} {lr:>12.2e}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val, "config": cfg, "n_features": n_features,
            }, os.path.join(output_dir, "best_model.pt"))
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"\n⚡ Early stopping на епосі {epoch}")
                break

    # ── Фінальна оцінка на test ───────────────────────────────────────────
    print(f"\n{'='*60}\n  Фінальна оцінка на TEST\n{'='*60}")
    ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    t_loss, t_mse, t_mae = run_epoch(
        model, loaders["test"], criterion, optimizer, device, False
    )
    print(f"  Test Loss : {t_loss:.4f}")
    print(f"  MSE       : {t_mse:.4f}  ← чутливий до великих помилок")
    print(f"  MAE       : {t_mae:.4f}  ← середня абсолютна помилка (мм)")

    results = {
        "test_loss": t_loss, "test_mse": t_mse, "test_mae": t_mae,
        "best_val_loss": best_val, "epochs_trained": len(history), "config": cfg,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"), index=False
    )

    print(f"\n✓ Модель     → {output_dir}/best_model.pt")
    print(f"✓ Результати → {output_dir}/results.json")
    print(f"✓ Історія    → {output_dir}/training_history.csv")
    return model, results


# ══════════════════════════════════════════════════════════════════════════════
#  ВІЗУАЛІЗАЦІЯ
# ══════════════════════════════════════════════════════════════════════════════

def load_scaler(data_dir, city=None):
    """Завантажує правильний scaler для міста або перший знайдений."""
    scalers_dir = os.path.join(data_dir, "scalers")
    if city:
        path = os.path.join(scalers_dir, f"{city}_target_scaler.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    # Fallback — перший знайдений
    files = glob.glob(os.path.join(scalers_dir, "*_target_scaler.pkl"))
    if files:
        with open(files[0], "rb") as f:
            return pickle.load(f)
    return None


def visualize(model, data_dir, output_dir, cfg, device_str="cpu"):
    device    = torch.device(device_str)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    model.eval()

    # ── 1. Графік навчання ────────────────────────────────────────────────
    hist_path = os.path.join(output_dir, "training_history.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Transformer — Історія навчання", fontsize=14, fontweight="bold")

        axes[0].plot(hist["epoch"], hist["train_loss"],
                     label="Train", color="#2196F3")
        axes[0].plot(hist["epoch"], hist["val_loss"],
                     label="Val",   color="#FF5722")
        axes[0].set_title("Загальний Loss (MSE+MAE)/2")
        axes[0].set_xlabel("Епоха"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(hist["epoch"], hist["val_mse"],
                     label="MSE", color="#9C27B0")
        axes[1].plot(hist["epoch"], hist["val_mae"],
                     label="MAE", color="#4CAF50")
        axes[1].set_title("Val: MSE vs MAE")
        axes[1].set_xlabel("Епоха"); axes[1].set_ylabel("Loss")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "01_training_history.png"), dpi=150)
        plt.close()
        print("  ✓ 01_training_history.png")

    # ── 2. Прогноз vs реальність ──────────────────────────────────────────
    test_path = os.path.join(data_dir, "all_cities_test.csv")
    if os.path.exists(test_path):
        df_test = (pd.read_csv(test_path)
                   .sort_values(["city", "datetime"])
                   .reset_index(drop=True))
        ds = WeatherDataset(df_test, cfg["seq_len"], cfg["pred_len"])

        np.random.seed(42)
        idxs = np.random.choice(len(ds), min(3, len(ds)), replace=False)
        fig, axes = plt.subplots(len(idxs), 1, figsize=(14, 4 * len(idxs)))
        if len(idxs) == 1:
            axes = [axes]
        fig.suptitle("Transformer — Прогноз vs Реальність",
                     fontsize=14, fontweight="bold")

        all_real, all_pred = [], []

        for row, idx in enumerate(idxs):
            X, y_reg = ds[idx]

            # Визначаємо місто для правильного scaler
            city_idx = idx // (len(ds) // 7)
            cities   = df_test["city"].unique()
            city     = cities[min(city_idx, len(cities)-1)]
            tgt_scaler = load_scaler(data_dir, city)

            with torch.no_grad():
                pred = model(X.unsqueeze(0).to(device))
            pn = pred.squeeze(0).cpu().numpy()
            rn = y_reg.numpy()

            if tgt_scaler:
                pm = np.maximum(np.expm1(
                    tgt_scaler.inverse_transform(pn.reshape(-1,1)).flatten()), 0)
                rm = np.maximum(np.expm1(
                    tgt_scaler.inverse_transform(rn.reshape(-1,1)).flatten()), 0)
                yl = "Опади (мм)"
            else:
                pm, rm, yl = pn, rn, "Опади (норм.)"

            all_real.extend(rm)
            all_pred.extend(pm)

            h       = np.arange(1, cfg["pred_len"] + 1)
            mae_val = np.mean(np.abs(pm - rm))
            ax      = axes[row]

            ax.fill_between(h, 0, rm, alpha=0.3,
                            color="#2196F3", label="Реальні опади")
            ax.plot(h, rm, "o-", color="#2196F3", ms=4)
            ax.plot(h, pm, "s--", color="#FF5722", ms=4, label="Прогноз")
            ax.set_title(f"Зразок {row+1} ({city})  |  MAE = {mae_val:.2f} мм")
            ax.set_xlabel("Година прогнозу")
            ax.set_ylabel(yl)
            ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "02_forecast_vs_real.png"), dpi=150)
        plt.close()
        print("  ✓ 02_forecast_vs_real.png")

        # ── 3. Scatter ────────────────────────────────────────────────────
        ar, ap = np.array(all_real), np.array(all_pred)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(ar, ap, alpha=0.4, s=15, color="#2196F3")
        mx = max(ar.max(), ap.max()) * 1.1
        ax.plot([0, mx], [0, mx], "r--", lw=1.5, label="Ідеальний прогноз")
        ax.set_title(
            f"Реальні vs Передбачені опади\n"
            f"MAE = {np.mean(np.abs(ap-ar)):.3f} мм  |  "
            f"MSE = {np.mean((ap-ar)**2):.3f} мм²"
        )
        ax.set_xlabel("Реальні (мм)"); ax.set_ylabel("Прогноз (мм)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "03_scatter.png"), dpi=150)
        plt.close()
        print("  ✓ 03_scatter.png")

        # ── 4. Розподіл помилок ───────────────────────────────────────────
        err = ap - ar
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Розподіл помилок прогнозу", fontsize=13, fontweight="bold")

        axes[0].hist(err, bins=50, color="#FF9800", edgecolor="white")
        axes[0].axvline(0, color="red", ls="--")
        axes[0].axvline(err.mean(), color="blue", ls="-",
                        label=f"Середнє: {err.mean():.3f} мм")
        axes[0].set_title("Гістограма помилок (прогноз − реальність)")
        axes[0].set_xlabel("Помилка (мм)"); axes[0].set_ylabel("Кількість")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].boxplot(err, patch_artist=True,
                        boxprops=dict(facecolor="#FF9800", alpha=0.6))
        axes[1].axhline(0, color="red", ls="--")
        axes[1].set_title("Box plot помилок")
        axes[1].set_ylabel("Помилка (мм)"); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "04_errors.png"), dpi=150)
        plt.close()
        print("  ✓ 04_errors.png")

    print(f"\n✓ Графіки → {plots_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict(model_path, input_data, target_scaler_path, device_str="cpu"):
    """
    Вхід  : input_data [72, n_features] — нормалізовані дані останніх 72 годин
    Вихід : precip_mm [24] — прогноз опадів у мм на 24 години вперед
    """
    device = torch.device(device_str)
    ckpt   = torch.load(model_path, map_location=device)
    cfg    = ckpt["config"]
    model  = PrecipitationTransformer(ckpt["n_features"], cfg["pred_len"], cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()
    print("RAW pred:", pred[:10])
    print("MEAN pred:", pred.mean())
    with open(target_scaler_path, "rb") as f:
        sc = pickle.load(f)
    mm = np.maximum(
        np.expm1(sc.inverse_transform(pred.reshape(-1, 1)).flatten()), 0
    )
    return mm


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./processed")
    parser.add_argument("--output_dir", default="./models/transformer")
    parser.add_argument("--epochs",     type=int,   default=CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CONFIG["lr"])
    parser.add_argument("--d_model",    type=int,   default=CONFIG["d_model"])
    args = parser.parse_args()

    CONFIG.update({
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "d_model": args.d_model,
    })

    model, results = train(CONFIG, args.data_dir, args.output_dir)
    visualize(
        model, args.data_dir, args.output_dir, CONFIG,
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu",
    )