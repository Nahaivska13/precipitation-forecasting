"""
XGBoost Класифікатор опадів — буде/не буде дощ
Використовує XGBoost на усереднених ознаках з 72 годин.
Добре працює з дисбалансом класів (scale_pos_weight).
"""

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Конфігурація ─────────────────────────────────────────────────────────────
CONFIG = {
    "seq_len":        72,
    "pred_len":       24,
    "rain_threshold": 0.1,

    # XGBoost параметри
    "n_estimators":   300,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":          0.1,
    "reg_alpha":      0.1,
    "reg_lambda":     1.0,
}

NON_FEATURE_COLS = {"datetime", "city", "precip_mm"}


# ══════════════════════════════════════════════════════════════════════════════
#  ПІДГОТОВКА ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features(df, seq_len, pred_len, rain_threshold):
    """
    Перетворює часовий ряд у табличний формат для XGBoost.

    Для кожного вікна [seq_len годин] обчислюємо агрегати:
      - mean, std, min, max, last значення кожної ознаки
      - lag features останніх годин

    Таргет: чи будуть опади хоча б в одній з наступних pred_len годин.
    """
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in NON_FEATURE_COLS]

    df_feat = df[feat_cols].copy()
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_feat = df_feat.clip(-10, 10)

    features = df_feat.values.astype(np.float32)
    target   = df["precip_mm"].values
    cities   = df["city"].values

    X_list, y_list = [], []

    for i in range(len(features) - seq_len - pred_len + 1):
        # Перевіряємо що вікно в межах одного міста
        if cities[i] != cities[i + seq_len + pred_len - 1]:
            continue

        window = features[i : i + seq_len]   # [72, n_features]

        # Агрегати по вікну для кожної ознаки
        row = np.concatenate([
            window.mean(axis=0),             # середнє
            window.std(axis=0),              # стандартне відхилення
            window.min(axis=0),              # мінімум
            window.max(axis=0),              # максимум
            window[-1],                      # остання година (найсвіжіша)
            window[-3:].mean(axis=0),        # середнє останніх 3 годин
            window[-6:].mean(axis=0),        # середнє останніх 6 годин
            window[-12:].mean(axis=0),       # середнє останніх 12 годин
        ])
        X_list.append(row)

        # Таргет: чи є опади хоча б в одній годині з наступних pred_len
        future_precip = target[i + seq_len : i + seq_len + pred_len]
        y_list.append(1 if (future_precip > rain_threshold).any() else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y


def load_data(data_dir, cfg):
    """Завантажує і підготовлює дані для XGBoost."""
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"all_cities_{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не знайдено: {path}")

        print(f"  Завантаження {split}...")
        df = (pd.read_csv(path)
                .sort_values(["city", "datetime"])
                .reset_index(drop=True))

        X, y = prepare_features(
            df, cfg["seq_len"], cfg["pred_len"], cfg["rain_threshold"]
        )

        n_rain   = y.sum()
        n_norain = len(y) - n_rain
        print(f"    Зразків: {len(X):,}  |  "
              f"дощ: {n_rain} ({n_rain/len(y)*100:.1f}%)  |  "
              f"без дощу: {n_norain} ({n_norain/len(y)*100:.1f}%)")

        splits[split] = (X, y)

    return splits


# ══════════════════════════════════════════════════════════════════════════════
#  НАВЧАННЯ
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  XGBoost Класифікатор опадів")
    print(f"{'='*60}\n")

    # ── Завантаження даних ────────────────────────────────────────────────
    print("Підготовка даних...")
    splits = load_data(data_dir, cfg)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # ── scale_pos_weight — компенсує дисбаланс ────────────────────────────
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / (n_pos + 1e-8)
    print(f"\n  scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  Розмір вхідного вектора: {X_train.shape[1]} ознак\n")

    # ── XGBoost модель ────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators      = cfg["n_estimators"],
        max_depth         = cfg["max_depth"],
        learning_rate     = cfg["learning_rate"],
        subsample         = cfg["subsample"],
        colsample_bytree  = cfg["colsample_bytree"],
        min_child_weight  = cfg["min_child_weight"],
        gamma             = cfg["gamma"],
        reg_alpha         = cfg["reg_alpha"],
        reg_lambda        = cfg["reg_lambda"],
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "logloss",
        early_stopping_rounds = 20,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )

    print("Навчання XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"  Найкраща ітерація: {best_iter}")

    # ── Оцінка на val ─────────────────────────────────────────────────────
    val_preds = model.predict(X_val)
    val_f1    = f1_score(y_val, val_preds)
    val_prec  = precision_score(y_val, val_preds)
    val_rec   = recall_score(y_val, val_preds)
    val_acc   = accuracy_score(y_val, val_preds)

    print(f"\n  Val результати:")
    print(f"    Accuracy  : {val_acc:.4f}")
    print(f"    F1        : {val_f1:.4f}")
    print(f"    Precision : {val_prec:.4f}")
    print(f"    Recall    : {val_rec:.4f}")

    # ── Оцінка на test ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Фінальна оцінка на TEST")
    print(f"{'='*60}")

    test_preds = model.predict(X_test)
    test_f1    = f1_score(y_test, test_preds)
    test_prec  = precision_score(y_test, test_preds)
    test_rec   = recall_score(y_test, test_preds)
    test_acc   = accuracy_score(y_test, test_preds)

    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1 Score  : {test_f1:.4f}")
    print(f"  Precision : {test_prec:.4f}  з передбачених дощів скільки правильних")
    print(f"  Recall    : {test_rec:.4f}  з реальних дощів скільки знайдено")
    print(f"\n{classification_report(y_test, test_preds, target_names=['Без дощу','Дощ'])}")

    # ── Збереження моделі ─────────────────────────────────────────────────
    model_path = os.path.join(output_dir, "best_classifier.pkl")
    model.save_model(os.path.join(output_dir, "best_classifier.json"))


    results = {
        "test_accuracy":  float(test_acc),
        "test_f1":        float(test_f1),
        "test_precision": float(test_prec),
        "test_recall":    float(test_rec),
        "val_f1":         float(val_f1),
        "best_iteration": int(best_iter),
        "config":         cfg,
    }
    with open(os.path.join(output_dir, "classifier_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Графік важливості ознак ───────────────────────────────────────────
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    importance = model.feature_importances_
    top_n      = 20
    top_idx    = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importance[top_idx], color="#2196F3", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"feature_{i}" for i in top_idx])
    ax.set_title("XGBoost — Топ 20 важливих ознак", fontsize=13, fontweight="bold")
    ax.set_xlabel("Важливість")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=150)
    plt.close()

    print(f"\n Модель збережена → {model_path}")
    print(f" Результати       → {output_dir}/classifier_results.json")
    print(f" Графік           → {plots_dir}/feature_importance.png")

    return model, results


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./processed")
    parser.add_argument("--output_dir", default="./models/classifier")
    parser.add_argument("--n_estimators", type=int, default=CONFIG["n_estimators"])
    parser.add_argument("--max_depth",    type=int, default=CONFIG["max_depth"])
    args = parser.parse_args()

    CONFIG.update({
        "n_estimators": args.n_estimators,
        "max_depth":    args.max_depth,
    })

    train(CONFIG, args.data_dir, args.output_dir)