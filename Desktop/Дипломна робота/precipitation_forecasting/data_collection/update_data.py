"""
Оновлення датасету новими даними з WeatherAPI
Завантажує нові погодинні дані від останньої дати в датасеті до сьогодні
і додає їх до існуючого CSV файлу.
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ast
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─── API ключ ────────────────────────────────────────────────────────────────
API_KEY = "3fccbad99750436dbeb160621260903"

# ─── Конфігурація ─────────────────────────────────────────────────────────────
BASE_DIR    = "/Users/darynanagaevskaya/Desktop/Дипломна робота/precipitation_forecasting"
RAW_DIR     = os.path.join(BASE_DIR, "data_collection/data_raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data_preprocessing/processed")

CITIES = {
    "Kyiv":      {"lat": 50.45, "lon": 30.52, "query": "Kyiv"},
    "Lviv":      {"lat": 49.84, "lon": 24.03, "query": "Lviv"},
    "Odesa":     {"lat": 46.48, "lon": 30.73, "query": "Odesa"},
    "Dnipro":    {"lat": 48.45, "lon": 35.05, "query": "Dnipro"},
    "Kharkiv":   {"lat": 50.00, "lon": 36.23, "query": "Kharkiv"},
    "Uzhhorod":  {"lat": 48.62, "lon": 22.29, "query": "Uzhhorod"},
    "Chernihiv": {"lat": 51.50, "lon": 31.30, "query": "Chernihiv"},
}

# Колонки які потрібні з API
NEEDED_COLS = [
    "time_epoch", "time", "temp_c", "is_day", "condition",
    "wind_kph", "wind_degree", "pressure_mb", "precip_mm",
    "snow_cm", "humidity", "cloud", "feelslike_c",
    "windchill_c", "heatindex_c", "dewpoint_c", "vis_km",
    "gust_kph", "uv",
]


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАВАНТАЖЕННЯ ДАНИХ З API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_day(api_key, query, date_str):
    """
    Завантажує погодинні дані за один день з WeatherAPI History.
    date_str формат: 'YYYY-MM-DD'
    Повертає list of dicts або None при помилці.
    """
    url = "http://api.weatherapi.com/v1/history.json"
    params = {
        "key": api_key,
        "q":   query,
        "dt":  date_str,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"    API помилка {resp.status_code} для {date_str}")
            return None
        data = resp.json()
        hours = data["forecast"]["forecastday"][0]["hour"]
        return hours
    except Exception as e:
        print(f"     Помилка запиту: {e}")
        return None


def hours_to_dataframe(hours, date_str):
    """Перетворює список годин з API у DataFrame."""
    rows = []
    for h in hours:
        row = {col: h.get(col, None) for col in NEEDED_COLS}
        row["date"] = date_str
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  ОТРИМАННЯ ОСТАННЬОЇ ДАТИ З ДАТАСЕТУ
# ══════════════════════════════════════════════════════════════════════════════

def get_last_date(city):
    """Повертає останню дату в сирому CSV файлі міста."""
    raw_path = os.path.join(RAW_DIR, f"{city.lower()}_hourly_weather_5years.csv")
    if not os.path.exists(raw_path):
        print(f"  ⚠ Файл не знайдено: {raw_path}")
        return None
    df = pd.read_csv(raw_path)
    df["time"] = pd.to_datetime(df["time"])
    last_date  = df["time"].max().date()
    return last_date


# ══════════════════════════════════════════════════════════════════════════════
#  ОНОВЛЕННЯ СИРОГО CSV
# ══════════════════════════════════════════════════════════════════════════════

def update_raw_csv(city, api_key):
    """
    Завантажує нові дані з API і додає до сирого CSV.
    Повертає кількість нових рядків або 0.
    """
    raw_path  = os.path.join(RAW_DIR, f"{city.lower()}_hourly_weather_5years.csv")
    city_info = CITIES[city]

    # Остання дата в датасеті
    last_date = get_last_date(city)
    if last_date is None:
        return 0

    # Дати які треба завантажити
    today      = datetime.now().date()
    start_date = last_date + timedelta(days=1)

    if start_date > today:
        print(f"  ✓ [{city}] Дані вже актуальні (остання дата: {last_date})")
        return 0

    print(f"  [{city}] Завантажуємо дані з {start_date} до {today}...")

    # Завантажуємо по одному дню
    all_new_rows = []
    current = start_date
    while current <= today:
        date_str = current.strftime("%Y-%m-%d")
        hours    = fetch_day(api_key, city_info["query"], date_str)
        if hours:
            df_day = hours_to_dataframe(hours, date_str)
            all_new_rows.append(df_day)
            print(f"    ✓ {date_str} — {len(df_day)} годин")
        current += timedelta(days=1)

    if not all_new_rows:
        print(f"  ⚠ [{city}] Нових даних не отримано")
        return 0

    # Об'єднуємо нові дані
    df_new = pd.concat(all_new_rows, ignore_index=True)

    # Додаємо до існуючого CSV
    df_old = pd.read_csv(raw_path)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    # Видаляємо дублікати по часу
    df_combined = df_combined.drop_duplicates(subset=["time_epoch"])
    df_combined = df_combined.sort_values("time_epoch").reset_index(drop=True)

    # Зберігаємо
    df_combined.to_csv(raw_path, index=False)

    n_new = len(df_new)
    print(f"  ✓ [{city}] Додано {n_new} нових рядків → {raw_path}")
    return n_new


# ══════════════════════════════════════════════════════════════════════════════
#  ОНОВЛЕННЯ PROCESSED ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════

def rerun_preprocessing():
    """Запускає preprocessing скрипт для оновлення processed CSV."""
    preprocess_script = os.path.join(
        BASE_DIR, "data_preprocessing/preprocess_weather_data.py"
    )
    if not os.path.exists(preprocess_script):
        print("⚠ Preprocessing скрипт не знайдено")
        return

    import subprocess
    print("\nЗапускаємо preprocessing...")
    result = subprocess.run(
        ["python", preprocess_script, RAW_DIR, PROCESSED_DIR],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✓ Preprocessing завершено успішно")
    else:
        print(f"⚠ Помилка preprocessing:\n{result.stderr}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key", default=API_KEY,
        help="API ключ від WeatherAPI"
    )
    parser.add_argument(
        "--city", default=None,
        choices=list(CITIES.keys()),
        help="Оновити лише одне місто (за замовчуванням — всі)"
    )
    parser.add_argument(
        "--no_preprocess", action="store_true",
        help="Не запускати preprocessing після оновлення"
    )
    args = parser.parse_args()

    # Визначаємо API ключ
    api_key = args.api_key

    # Визначаємо які міста оновлювати
    cities_to_update = [args.city] if args.city else list(CITIES.keys())

    print(f"\n{'='*55}")
    print(f"  Оновлення датасету")
    print(f"  Міста: {', '.join(cities_to_update)}")
    print(f"{'='*55}\n")

    total_new = 0
    for city in cities_to_update:
        new_rows = update_raw_csv(city, api_key)
        total_new += new_rows

    print(f"\n{'='*55}")
    print(f"  Всього нових рядків: {total_new}")
    print(f"{'='*55}")

    # Запускаємо preprocessing якщо є нові дані
    if total_new > 0 and not args.no_preprocess:
        rerun_preprocessing()
        print(f"\n✓ Датасет оновлено!")
        print(f"  Тепер можна запустити прогноз:")
        print(f"  python models/forecast.py")
    elif total_new == 0:
        print("\n✓ Датасет вже актуальний — нових даних немає")