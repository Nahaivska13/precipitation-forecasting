"""
Оцінка метрик Transformer моделі для прогнозування опадів
Обчислює детальні метрики якості регресії:
  - RMSE, MAE, R², MAPE
  - Baseline порівняння (завжди передбачаємо 0)
  - Метрики для різних діапазонів опадів
  - Візуалізація результатів
"""

import os
import argparse
import json
import pickle
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  МОДЕЛЬ (копія з train файлу)
# ══════════════════════════════════════════════════════════════════════════════

NON_FEATURE_COLS = {"datetime", "city", "precip_mm"}

class WeatherDataset(Dataset):
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
        self.pos_emb = nn.Embedding(cfg["seq_len"] + 10, d)

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
        x   = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        x   = x + self.pos_emb(pos).unsqueeze(0)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.encoder(x)
        cls_out = x[:, 0]

        return self.regression_head(cls_out)


# ══════════════════════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred):
    """
    Обчислює всі метрики для регресії.
    
    Повертає:
        dict з метриками: RMSE, MAE, R², MAPE, MSE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Базові метрики
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² (коефіцієнт детермінації)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # MAPE (тільки для ненульових значень)
    mask = y_true > 0.01  # уникаємо ділення на дуже малі числа
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    # Додаткові метрики
    median_ae = np.median(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Кореляція Пірсона
    if len(y_true) > 1:
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    else:
        pearson_r, pearson_p = 0.0, 1.0
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "R²": r2,
        "MAPE": mape,
        "Median_AE": median_ae,
        "Max_Error": max_error,
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
    }


def calculate_baseline_metrics(y_true):
    """
    Обчислює метрики для baseline моделі (завжди передбачаємо 0).
    Це показує наскільки модель краща за naive підхід.
    """
    y_pred_zero = np.zeros_like(y_true)
    return calculate_metrics(y_true, y_pred_zero)


def metrics_by_intensity(y_true, y_pred, bins=None):
    """
    Розбиває метрики по інтенсивності опадів.
    
    Діапазони:
      - Без опадів: 0 мм
      - Слабкі: 0.1-2.5 мм
      - Помірні: 2.5-10 мм
      - Сильні: 10-50 мм
      - Дуже сильні: >50 мм
    """
    if bins is None:
        bins = [0, 0.1, 2.5, 10, 50, np.inf]
    
    labels = ["Немає опадів", "Слабкі (0.1-2.5мм)", 
              "Помірні (2.5-10мм)", "Сильні (10-50мм)", 
              "Дуже сильні (>50мм)"]
    
    results = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() > 0:
            metrics = calculate_metrics(y_true[mask], y_pred[mask])
            metrics["count"] = mask.sum()
            metrics["percentage"] = mask.sum() / len(y_true) * 100
            results[labels[i]] = metrics
        else:
            results[labels[i]] = {"count": 0, "percentage": 0.0}
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАВАНТАЖЕННЯ МОДЕЛІ ТА ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════

def load_scaler(data_dir, city=None):
    """Завантажує scaler для денормалізації."""
    scalers_dir = os.path.join(data_dir, "scalers")
    if city:
        path = os.path.join(scalers_dir, f"{city}_target_scaler.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    # Fallback
    files = glob.glob(os.path.join(scalers_dir, "*_target_scaler.pkl"))
    if files:
        with open(files[0], "rb") as f:
            return pickle.load(f)
    return None


def get_predictions(model, dataloader, device, scaler=None):
    """Отримує всі передбачення моделі на датасеті."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y_reg in dataloader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            y_np = y_reg.numpy()
            
            all_preds.append(pred)
            all_targets.append(y_np)
    
    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Денормалізація якщо є scaler
    if scaler is not None:
        preds = np.maximum(
            np.expm1(scaler.inverse_transform(preds.reshape(-1, 1)).flatten()), 0
        )
        targets = np.maximum(
            np.expm1(scaler.inverse_transform(targets.reshape(-1, 1)).flatten()), 0
        )
    
    return targets, preds


# ══════════════════════════════════════════════════════════════════════════════
#  ВІЗУАЛІЗАЦІЯ
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_comparison(metrics_model, metrics_baseline, output_dir):
    """Порівняння моделі з baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Порівняння Transformer vs Baseline (завжди 0)", 
                 fontsize=14, fontweight="bold")
    
    # RMSE
    axes[0].bar(["Baseline\n(завжди 0)", "Transformer"], 
                [metrics_baseline["RMSE"], metrics_model["RMSE"]], 
                color=["#FF5722", "#2196F3"])
    axes[0].set_ylabel("RMSE (мм)")
    axes[0].set_title("Root Mean Squared Error")
    axes[0].grid(alpha=0.3, axis='y')
    improvement = (1 - metrics_model["RMSE"]/metrics_baseline["RMSE"]) * 100
    axes[0].text(0.5, 0.95, f"Покращення: {improvement:.1f}%", 
                 transform=axes[0].transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # MAE
    axes[1].bar(["Baseline\n(завжди 0)", "Transformer"], 
                [metrics_baseline["MAE"], metrics_model["MAE"]], 
                color=["#FF5722", "#2196F3"])
    axes[1].set_ylabel("MAE (мм)")
    axes[1].set_title("Mean Absolute Error")
    axes[1].grid(alpha=0.3, axis='y')
    improvement = (1 - metrics_model["MAE"]/metrics_baseline["MAE"]) * 100
    axes[1].text(0.5, 0.95, f"Покращення: {improvement:.1f}%", 
                 transform=axes[1].transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # R²
    axes[2].bar(["Baseline\n(завжди 0)", "Transformer"], 
                [metrics_baseline["R²"], metrics_model["R²"]], 
                color=["#FF5722", "#2196F3"])
    axes[2].set_ylabel("R²")
    axes[2].set_title("Coefficient of Determination")
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_baseline_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ 01_baseline_comparison.png")


def plot_predictions_vs_actual(y_true, y_pred, output_dir):
    """Scatter plot передбачень vs реальність."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Прогноз vs Реальність", fontsize=14, fontweight="bold")
    
    # Всі дані
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="#2196F3")
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label="Ідеальний прогноз")
    axes[0].set_xlabel("Реальні опади (мм)")
    axes[0].set_ylabel("Прогноз (мм)")
    axes[0].set_title("Всі дані")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Zoom на діапазон 0-20мм (де більшість даних)
    mask = (y_true <= 20) & (y_pred <= 20)
    axes[1].scatter(y_true[mask], y_pred[mask], alpha=0.3, s=10, color="#2196F3")
    axes[1].plot([0, 20], [0, 20], 'r--', lw=2, label="Ідеальний прогноз")
    axes[1].set_xlabel("Реальні опади (мм)")
    axes[1].set_ylabel("Прогноз (мм)")
    axes[1].set_title("Zoom: 0-20 мм")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_predictions_scatter.png"), dpi=150)
    plt.close()
    print("  ✓ 02_predictions_scatter.png")


def plot_residuals(y_true, y_pred, output_dir):
    """Аналіз залишків (residuals)."""
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Аналіз помилок прогнозу", fontsize=14, fontweight="bold")
    
    # 1. Гістограма помилок
    axes[0, 0].hist(residuals, bins=100, color="#FF9800", edgecolor="white", alpha=0.7)
    axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].axvline(residuals.mean(), color="blue", linestyle="-", linewidth=2,
                       label=f"Середнє: {residuals.mean():.3f} мм")
    axes[0, 0].set_xlabel("Помилка = Прогноз - Реальність (мм)")
    axes[0, 0].set_ylabel("Частота")
    axes[0, 0].set_title("Розподіл помилок")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10, color="#9C27B0")
    axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("Прогноз (мм)")
    axes[0, 1].set_ylabel("Помилка (мм)")
    axes[0, 1].set_title("Residual Plot")
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Box plot помилок
    axes[1, 0].boxplot(residuals, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#FF9800", alpha=0.6))
    axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 0].set_ylabel("Помилка (мм)")
    axes[1, 0].set_title("Box Plot помилок")
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # 4. Q-Q plot (перевірка нормальності)
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot (перевірка нормальності помилок)")
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_residuals_analysis.png"), dpi=150)
    plt.close()
    print("  ✓ 03_residuals_analysis.png")


def plot_metrics_by_intensity(intensity_metrics, output_dir):
    """Метрики по діапазонах інтенсивності."""
    categories = list(intensity_metrics.keys())
    rmse_vals = [intensity_metrics[cat].get("RMSE", 0) for cat in categories]
    mae_vals = [intensity_metrics[cat].get("MAE", 0) for cat in categories]
    counts = [intensity_metrics[cat].get("count", 0) for cat in categories]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Метрики по діапазонах інтенсивності опадів", 
                 fontsize=14, fontweight="bold")
    
    x = np.arange(len(categories))
    width = 0.35
    
    # RMSE та MAE
    axes[0].bar(x - width/2, rmse_vals, width, label="RMSE", color="#2196F3", alpha=0.8)
    axes[0].bar(x + width/2, mae_vals, width, label="MAE", color="#4CAF50", alpha=0.8)
    axes[0].set_ylabel("Помилка (мм)")
    axes[0].set_title("RMSE та MAE по діапазонах")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    
    # Кількість зразків
    axes[1].bar(categories, counts, color="#FF9800", alpha=0.8)
    axes[1].set_ylabel("Кількість зразків")
    axes[1].set_title("Розподіл зразків по діапазонах")
    axes[1].set_xticklabels(categories, rotation=15, ha='right')
    axes[1].grid(alpha=0.3, axis='y')
    
    # Додаємо відсотки
    for i, (cat, count) in enumerate(zip(categories, counts)):
        pct = intensity_metrics[cat].get("percentage", 0)
        axes[1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_metrics_by_intensity.png"), dpi=150)
    plt.close()
    print("  ✓ 04_metrics_by_intensity.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ГОЛОВНА ФУНКЦІЯ EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model_path, data_dir, output_dir):
    """Повна оцінка моделі."""
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    print(f"\n{'='*70}")
    print(f"  EVALUATION - Transformer для прогнозування опадів")
    print(f"{'='*70}\n")
    print(f"  Пристрій: {device}")
    print(f"  Модель: {model_path}")
    print(f"  Дані: {data_dir}")
    print(f"  Результати → {output_dir}\n")
    
    # ── Завантаження моделі ───────────────────────────────────────────────
    print("Завантаження моделі...")
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    n_features = ckpt["n_features"]
    
    model = PrecipitationTransformer(n_features, cfg["pred_len"], cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"  ✓ Модель завантажена (епоха {ckpt['epoch']})")
    
    # ── Завантаження тестових даних ───────────────────────────────────────
    print("\nЗавантаження тестових даних...")
    test_path = "/Users/darynanagaevskaya/Desktop/Дипломна робота/precipitation_forecasting/data_preprocessing/processed/all_cities_test.csv"
    df_test = pd.read_csv(test_path).sort_values(["city", "datetime"]).reset_index(drop=True)
    
    test_ds = WeatherDataset(df_test, cfg["seq_len"], cfg["pred_len"])
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"  ✓ Тестових зразків: {len(test_ds):,}")
    
    # ── Отримання передбачень ─────────────────────────────────────────────
    print("\nОтримання передбачень...")
    scaler = load_scaler(data_dir)
    y_true, y_pred = get_predictions(model, test_loader, device, scaler)
    print(f"  ✓ Передбачень: {len(y_pred):,}")
    
    # ══════════════════════════════════════════════════════════════════════
    #  ОБЧИСЛЕННЯ МЕТРИК
    # ══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("  МЕТРИКИ ЯКОСТІ")
    print(f"{'='*70}\n")
    
    # 1. Метрики моделі
    metrics_model = calculate_metrics(y_true, y_pred)
    
    print("📊 Метрики Transformer:")
    print(f"  RMSE (Root Mean Squared Error)  : {metrics_model['RMSE']:.4f} мм")
    print(f"  MAE  (Mean Absolute Error)      : {metrics_model['MAE']:.4f} мм")
    print(f"  MSE  (Mean Squared Error)       : {metrics_model['MSE']:.4f} мм²")
    print(f"  R²   (Coefficient of Determination) : {metrics_model['R²']:.4f}")
    print(f"  MAPE (Mean Absolute % Error)    : {metrics_model['MAPE']:.2f}%")
    print(f"  Median Absolute Error           : {metrics_model['Median_AE']:.4f} мм")
    print(f"  Max Error                       : {metrics_model['Max_Error']:.4f} мм")
    print(f"  Pearson Correlation             : {metrics_model['Pearson_r']:.4f} (p={metrics_model['Pearson_p']:.2e})")
    
    # 2. Baseline метрики (завжди 0)
    print(f"\n📉 Baseline (завжди передбачаємо 0 мм):")
    metrics_baseline = calculate_baseline_metrics(y_true)
    print(f"  RMSE : {metrics_baseline['RMSE']:.4f} мм")
    print(f"  MAE  : {metrics_baseline['MAE']:.4f} мм")
    print(f"  R²   : {metrics_baseline['R²']:.4f}")
    
    # Покращення відносно baseline
    print(f"\n✨ Покращення відносно Baseline:")
    rmse_improvement = (1 - metrics_model['RMSE'] / metrics_baseline['RMSE']) * 100
    mae_improvement = (1 - metrics_model['MAE'] / metrics_baseline['MAE']) * 100
    print(f"  RMSE: {rmse_improvement:.2f}% краще")
    print(f"  MAE : {mae_improvement:.2f}% краще")
    
    # 3. Метрики по інтенсивності
    print(f"\n{'='*70}")
    print("  МЕТРИКИ ПО ДІАПАЗОНАХ ІНТЕНСИВНОСТІ")
    print(f"{'='*70}\n")
    
    intensity_metrics = metrics_by_intensity(y_true, y_pred)
    
    for category, metrics in intensity_metrics.items():
        if metrics['count'] > 0:
            print(f"{category}:")
            print(f"  Кількість: {metrics['count']:,} ({metrics['percentage']:.1f}%)")
            print(f"  RMSE: {metrics['RMSE']:.4f} мм  |  MAE: {metrics['MAE']:.4f} мм  |  R²: {metrics['R²']:.4f}")
            print()
    
    # ══════════════════════════════════════════════════════════════════════
    #  ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
    # ══════════════════════════════════════════════════════════════════════
    
    results = {
        "model_metrics": {k: float(v) for k, v in metrics_model.items()},
        "baseline_metrics": {k: float(v) for k, v in metrics_baseline.items()},
        "improvement": {
            "RMSE_improvement_%": float(rmse_improvement),
            "MAE_improvement_%": float(mae_improvement),
        },
        "metrics_by_intensity": {
            cat: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                  for k, v in m.items()}
            for cat, m in intensity_metrics.items()
        },
        "data_statistics": {
            "n_samples": int(len(y_true)),
            "mean_actual": float(y_true.mean()),
            "std_actual": float(y_true.std()),
            "mean_predicted": float(y_pred.mean()),
            "std_predicted": float(y_pred.std()),
        }
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Детальні передбачення
    predictions_df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "error": y_pred - y_true,
        "absolute_error": np.abs(y_pred - y_true),
    })
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    print(f"\n{'='*70}")
    print("  ВІЗУАЛІЗАЦІЯ")
    print(f"{'='*70}\n")
    
    # Графіки
    plot_metrics_comparison(metrics_model, metrics_baseline, output_dir)
    plot_predictions_vs_actual(y_true, y_pred, output_dir)
    plot_residuals(y_true, y_pred, output_dir)
    plot_metrics_by_intensity(intensity_metrics, output_dir)
    
    print(f"\n{'='*70}")
    print("  ✅ EVALUATION ЗАВЕРШЕНО")
    print(f"{'='*70}")
    print(f"\n  Результати:")
    print(f"    📄 evaluation_results.json - всі метрики")
    print(f"    📄 predictions.csv - детальні передбачення")
    print(f"    📊 4 графіки у {output_dir}/")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Оцінка метрик Transformer моделі для прогнозування опадів"
    )
    parser.add_argument(
        "--model_path",
        default="./models/transformer/best_model.pt",
        help="Шлях до збереженої моделі"
    )
    parser.add_argument(
        "--data_dir",
        default="./processed",
        help="Директорія з обробленими даними"
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation/transformer",
        help="Директорія для збереження результатів"
    )
    
    args = parser.parse_args()
    evaluate(args.model_path, args.data_dir, args.output_dir)