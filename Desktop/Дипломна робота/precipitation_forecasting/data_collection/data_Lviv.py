import requests
import pandas as pd
from datetime import datetime, timedelta
import time

api_key = "3fccbad99750436dbeb160621260903"
city = "Lviv"
output_file = "lviv_hourly_weather_5years.csv"

end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

all_data = []
current_date = start_date

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")

    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date_str}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        hourly = data["forecast"]["forecastday"][0]["hour"]

        df = pd.DataFrame(hourly)
        df["date"] = date_str

        all_data.append(df)
        print(f"Завантажено {date_str}")
    else:
        print(f"Помилка {response.status_code} для {date_str}")

    current_date += timedelta(days=1)
    time.sleep(1)

final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv(output_file, index=False)

print(f"Усі дані збережено у {output_file}")