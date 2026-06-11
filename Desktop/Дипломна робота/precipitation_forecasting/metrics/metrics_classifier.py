"""
Оцінка метрик XGBoost класифікатора опадів
Обчислює детальні метрики якості класифікації:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, PR-AUC
  - Confusion Matrix
  - Baseline порівняння (завжди передбачаємо "немає опадів")
  - Метрики для різних порогів ймовірності
"""

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  ПІДГОТОВКА ДАНИХ (копія з train файлу)
# ══════════════════════════════════════════════════════════════════════════════

NON_FEATURE_COLS = {"datetime", "city", "precip_mm"}

def prepare_features(df, seq_len, pred_len, rain_threshold):
    """Перетворює часовий ряд у табличний формат для XGBoost."""
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
        if cities[i] != cities[i + seq_len + pred_len - 1]:
            continue

        window = features[i : i + seq_len]

        row = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            window[-1],
            window[-3:].mean(axis=0),
            window[-6:].mean(axis=0),
            window[-12:].mean(axis=0),
        ])
        X_list.append(row)

        future_precip = target[i + seq_len : i + seq_len + pred_len]
        y_list.append(1 if (future_precip > rain_threshold).any() else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

def calculate_all_metrics(y_true, y_pred, y_proba=None):
    """
    Обчислює всі метрики класифікації.
    
    Args:
        y_true: справжні мітки (0/1)
        y_pred: передбачені мітки (0/1)
        y_proba: ймовірності класу 1 (опціонально)
    
    Returns:
        dict з метриками
    """
    # Базові метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Negative Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Balanced Accuracy
    balanced_acc = (recall + specificity) / 2
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "npv": npv,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }
    }
    
    # Метрики на основі ймовірностей
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            metrics["roc_auc"] = roc_auc
            metrics["pr_auc"] = pr_auc
        except:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    
    return metrics


def calculate_baseline_metrics(y_true):
    """
    Baseline: завжди передбачаємо клас "немає опадів" (0).
    """
    y_pred_baseline = np.zeros_like(y_true)
    return calculate_all_metrics(y_true, y_pred_baseline)


def find_optimal_threshold(y_true, y_proba):
    """
    Знаходить оптимальний поріг ймовірності за різними критеріями.
    """
    thresholds = np.linspace(0, 1, 101)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        metrics = calculate_all_metrics(y_true, y_pred)
        results.append({
            "threshold": thresh,
            "f1": metrics["f1_score"],
            "balanced_acc": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
    
    df = pd.DataFrame(results)
    
    # Різні оптимальні пороги
    optimal_thresholds = {
        "max_f1": df.loc[df["f1"].idxmax(), "threshold"],
        "max_balanced_acc": df.loc[df["balanced_acc"].idxmax(), "threshold"],
        "precision_80": df[df["precision"] >= 0.8]["threshold"].min() if (df["precision"] >= 0.8).any() else 0.5,
        "recall_80": df[df["recall"] >= 0.8]["threshold"].max() if (df["recall"] >= 0.8).any() else 0.5,
    }
    
    return optimal_thresholds, df


# ══════════════════════════════════════════════════════════════════════════════
#  ВІЗУАЛІЗАЦІЯ
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(y_true, y_pred_model, y_pred_baseline, output_dir):
    """Порівняння confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrices: Модель vs Baseline", fontsize=14, fontweight="bold")
    
    # Baseline
    cm_baseline = confusion_matrix(y_true, y_pred_baseline)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Немає опадів', 'Опади'],
                yticklabels=['Немає опадів', 'Опади'],
                ax=axes[0], cbar_kws={'label': 'Кількість'})
    axes[0].set_title('Baseline (завжди "немає опадів")')
    axes[0].set_ylabel('Реальність')
    axes[0].set_xlabel('Прогноз')
    
    # Модель
    cm_model = confusion_matrix(y_true, y_pred_model)
    sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Немає опадів', 'Опади'],
                yticklabels=['Немає опадів', 'Опади'],
                ax=axes[1], cbar_kws={'label': 'Кількість'})
    axes[1].set_title('XGBoost Classifier')
    axes[1].set_ylabel('Реальність')
    axes[1].set_xlabel('Прогноз')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_confusion_matrices.png"), dpi=150)
    plt.close()
    print("  ✓ 01_confusion_matrices.png")


def plot_metrics_comparison(metrics_model, metrics_baseline, output_dir):
    """Порівняння метрик моделі та baseline."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Порівняння XGBoost vs Baseline", fontsize=14, fontweight="bold")
    
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("f1_score", "F1-Score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        values = [metrics_baseline[metric_key], metrics_model[metric_key]]
        bars = ax.bar(["Baseline", "XGBoost"], values, color=["#FF5722", "#2196F3"])
        
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}")
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, axis='y')
        
        # Додаємо значення на стовпчики
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Покращення
        if values[0] > 0:
            improvement = (values[1] - values[0]) / values[0] * 100
            ax.text(0.5, 0.95, f"Покращення: {improvement:+.1f}%",
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_metrics_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ 02_metrics_comparison.png")


def plot_roc_and_pr_curves(y_true, y_proba, metrics, output_dir):
    """ROC та Precision-Recall криві."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC та Precision-Recall криві", fontsize=14, fontweight="bold")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = metrics.get("roc_auc", 0)
    
    axes[0].plot(fpr, tpr, color="#2196F3", lw=2, 
                 label=f'ROC curve (AUC = {auc_score:.3f})')
    axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Random classifier')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = metrics.get("pr_auc", 0)
    
    axes[1].plot(recall, precision, color="#4CAF50", lw=2,
                 label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Baseline precision (частка позитивних зразків)
    baseline_precision = y_true.sum() / len(y_true)
    axes[1].axhline(baseline_precision, color='r', linestyle='--', lw=2,
                    label=f'Baseline (random) = {baseline_precision:.3f}')
    
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_roc_pr_curves.png"), dpi=150)
    plt.close()
    print("  ✓ 03_roc_pr_curves.png")


def plot_threshold_analysis(threshold_df, optimal_thresholds, output_dir):
    """Аналіз різних порогів класифікації."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Аналіз порогів ймовірності", fontsize=14, fontweight="bold")
    
    # F1, Precision, Recall vs Threshold
    axes[0].plot(threshold_df["threshold"], threshold_df["f1"], 
                 label="F1-Score", color="#2196F3", lw=2)
    axes[0].plot(threshold_df["threshold"], threshold_df["precision"], 
                 label="Precision", color="#4CAF50", lw=2)
    axes[0].plot(threshold_df["threshold"], threshold_df["recall"], 
                 label="Recall", color="#FF9800", lw=2)
    
    # Оптимальні пороги
    for name, thresh in optimal_thresholds.items():
        if not np.isnan(thresh):
            axes[0].axvline(thresh, color='red', linestyle='--', alpha=0.5)
            axes[0].text(thresh, 0.95, name.replace('_', ' '), 
                        rotation=90, va='top', fontsize=8)
    
    axes[0].set_xlabel("Поріг ймовірності")
    axes[0].set_ylabel("Значення метрики")
    axes[0].set_title("Метрики vs Поріг")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Balanced Accuracy vs Threshold
    axes[1].plot(threshold_df["threshold"], threshold_df["balanced_acc"],
                 color="#9C27B0", lw=2, label="Balanced Accuracy")
    axes[1].axvline(optimal_thresholds["max_balanced_acc"], 
                    color='red', linestyle='--', lw=2,
                    label=f'Optimal = {optimal_thresholds["max_balanced_acc"]:.3f}')
    axes[1].set_xlabel("Поріг ймовірності")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_title("Balanced Accuracy vs Поріг")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_threshold_analysis.png"), dpi=150)
    plt.close()
    print("  ✓ 04_threshold_analysis.png")


def plot_probability_distribution(y_true, y_proba, output_dir):
    """Розподіл передбачених ймовірностей."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Окремо для класів
    proba_no_rain = y_proba[y_true == 0]
    proba_rain = y_proba[y_true == 1]
    
    ax.hist(proba_no_rain, bins=50, alpha=0.6, label='Реально: Немає опадів', 
            color='#2196F3', edgecolor='white')
    ax.hist(proba_rain, bins=50, alpha=0.6, label='Реально: Опади',
            color='#FF5722', edgecolor='white')
    
    ax.axvline(0.5, color='black', linestyle='--', lw=2, label='Поріг = 0.5')
    ax.set_xlabel('Передбачена ймовірність опадів')
    ax.set_ylabel('Частота')
    ax.set_title('Розподіл передбачених ймовірностей по класах')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_probability_distribution.png"), dpi=150)
    plt.close()
    print("  ✓ 05_probability_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ГОЛОВНА ФУНКЦІЯ EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model_path, data_dir, output_dir):
    """Повна оцінка класифікатора."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  EVALUATION - XGBoost класифікатор опадів")
    print(f"{'='*70}\n")
    print(f"  Модель: {model_path}")
    print(f"  Дані: {data_dir}")
    print(f"  Результати → {output_dir}\n")
    
    # ── Завантаження моделі ───────────────────────────────────────────────
    print("Завантаження моделі...")
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    
    model = ckpt["model"]
    cfg = ckpt["config"]
    print(f"  ✓ Модель завантажена")
    print(f"  ✓ Кількість ознак: {ckpt['n_features_in']}")
    
    # ── Завантаження тестових даних ───────────────────────────────────────
    print("\nЗавантаження тестових даних...")
    test_path = "/Users/darynanagaevskaya/Desktop/Дипломна робота/precipitation_forecasting/data_preprocessing/processed/all_cities_test.csv"
    df_test = pd.read_csv(test_path).sort_values(["city", "datetime"]).reset_index(drop=True)
    
    X_test, y_test = prepare_features(
        df_test, cfg["seq_len"], cfg["pred_len"], cfg["rain_threshold"]
    )
    
    print(f"  ✓ Тестових зразків: {len(X_test):,}")
    print(f"  ✓ Опади: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.1f}%)")
    print(f"  ✓ Без опадів: {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.1f}%)")
    
    # ── Передбачення ──────────────────────────────────────────────────────
    print("\nОтримання передбачень...")
    y_proba = model.predict_proba(X_test)[:, 1]  # ймовірність класу 1
    y_pred = model.predict(X_test)
    print(f"  ✓ Передбачень: {len(y_pred):,}")
    
    # ══════════════════════════════════════════════════════════════════════
    #  ОБЧИСЛЕННЯ МЕТРИК
    # ══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("  МЕТРИКИ ЯКОСТІ")
    print(f"{'='*70}\n")
    
    # 1. Метрики моделі
    metrics_model = calculate_all_metrics(y_test, y_pred, y_proba)
    
    print("📊 Метрики XGBoost:")
    print(f"  Accuracy              : {metrics_model['accuracy']:.4f}")
    print(f"  Balanced Accuracy     : {metrics_model['balanced_accuracy']:.4f}")
    print(f"  Precision             : {metrics_model['precision']:.4f}  (з передбачених опадів скільки правильних)")
    print(f"  Recall (Sensitivity)  : {metrics_model['recall']:.4f}  (з реальних опадів скільки знайдено)")
    print(f"  Specificity           : {metrics_model['specificity']:.4f}  (з реальних 'без опадів' скільки знайдено)")
    print(f"  F1-Score              : {metrics_model['f1_score']:.4f}")
    print(f"  ROC-AUC               : {metrics_model['roc_auc']:.4f}")
    print(f"  PR-AUC                : {metrics_model['pr_auc']:.4f}")
    print(f"  Matthews Corr. Coef.  : {metrics_model['mcc']:.4f}")
    print(f"  Cohen's Kappa         : {metrics_model['cohen_kappa']:.4f}")
    
    print(f"\n  Confusion Matrix:")
    cm = metrics_model['confusion_matrix']
    print(f"    True Positives  (TP): {cm['TP']:,}  - правильно знайдені опади")
    print(f"    True Negatives  (TN): {cm['TN']:,}  - правильно знайдені 'без опадів'")
    print(f"    False Positives (FP): {cm['FP']:,}  - хибна тривога (опади, яких не було)")
    print(f"    False Negatives (FN): {cm['FN']:,}  - пропущені опади")
    
    # 2. Baseline метрики
    print(f"\n📉 Baseline (завжди передбачаємо 'немає опадів'):")
    y_pred_baseline = np.zeros_like(y_test)
    metrics_baseline = calculate_baseline_metrics(y_test)
    print(f"  Accuracy   : {metrics_baseline['accuracy']:.4f}")
    print(f"  Precision  : {metrics_baseline['precision']:.4f}")
    print(f"  Recall     : {metrics_baseline['recall']:.4f}")
    print(f"  F1-Score   : {metrics_baseline['f1_score']:.4f}")
    
    # Покращення
    print(f"\n✨ Покращення відносно Baseline:")
    for metric in ["accuracy", "f1_score", "precision", "recall"]:
        if metrics_baseline[metric] > 0:
            improvement = (metrics_model[metric] - metrics_baseline[metric]) / metrics_baseline[metric] * 100
            print(f"  {metric.capitalize():15s}: {improvement:+.2f}%")
        else:
            print(f"  {metric.capitalize():15s}: +∞% (baseline=0)")
    
    # 3. Аналіз порогів
    print(f"\n{'='*70}")
    print("  АНАЛІЗ ОПТИМАЛЬНИХ ПОРОГІВ")
    print(f"{'='*70}\n")
    
    optimal_thresholds, threshold_df = find_optimal_threshold(y_test, y_proba)
    
    print("Оптимальні пороги за різними критеріями:")
    for name, thresh in optimal_thresholds.items():
        if not np.isnan(thresh):
            y_pred_opt = (y_proba >= thresh).astype(int)
            metrics_opt = calculate_all_metrics(y_test, y_pred_opt)
            print(f"\n  {name}:")
            print(f"    Поріг: {thresh:.3f}")
            print(f"    F1: {metrics_opt['f1_score']:.4f}  |  "
                  f"Precision: {metrics_opt['precision']:.4f}  |  "
                  f"Recall: {metrics_opt['recall']:.4f}")
    
    # ══════════════════════════════════════════════════════════════════════
    #  ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
    # ══════════════════════════════════════════════════════════════════════
    
    results = {
        "model_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                         for k, v in metrics_model.items()},
        "baseline_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                            for k, v in metrics_baseline.items()},
        "optimal_thresholds": {k: float(v) for k, v in optimal_thresholds.items()
                              if not np.isnan(v)},
        "classification_report": classification_report(
            y_test, y_pred, 
            target_names=['Немає опадів', 'Опади'],
            output_dict=True
        ),
        "data_statistics": {
            "n_samples": int(len(y_test)),
            "n_positive": int(y_test.sum()),
            "n_negative": int((y_test == 0).sum()),
            "positive_rate": float(y_test.sum() / len(y_test)),
        }
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Детальні передбачення
    predictions_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
        "probability": y_proba,
        "correct": (y_test == y_pred).astype(int),
    })
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    # Пороги
    threshold_df.to_csv(os.path.join(output_dir, "threshold_analysis.csv"), index=False)
    
    print(f"\n{'='*70}")
    print("  ВІЗУАЛІЗАЦІЯ")
    print(f"{'='*70}\n")
    
    # Графіки
    plot_confusion_matrices(y_test, y_pred, y_pred_baseline, output_dir)
    plot_metrics_comparison(metrics_model, metrics_baseline, output_dir)
    plot_roc_and_pr_curves(y_test, y_proba, metrics_model, output_dir)
    plot_threshold_analysis(threshold_df, optimal_thresholds, output_dir)
    plot_probability_distribution(y_test, y_proba, output_dir)
    
    print(f"\n{'='*70}")
    print("  ✅ EVALUATION ЗАВЕРШЕНО")
    print(f"{'='*70}")
    print(f"\n  Результати:")
    print(f"    📄 evaluation_results.json - всі метрики")
    print(f"    📄 predictions.csv - детальні передбачення")
    print(f"    📄 threshold_analysis.csv - аналіз порогів")
    print(f"    📊 5 графіків у {output_dir}/")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Оцінка метрик XGBoost класифікатора опадів"
    )
    parser.add_argument(
        "--model_path",
        default="./models/classifier/best_classifier.pkl",
        help="Шлях до збереженої моделі"
    )
    parser.add_argument(
        "--data_dir",
        default="./processed",
        help="Директорія з обробленими даними"
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation/classifier",
        help="Директорія для збереження результатів"
    )
    
    args = parser.parse_args()
    evaluate(args.model_path, args.data_dir, args.output_dir)