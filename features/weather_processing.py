import pandas as pd
from typing import List

def filter_weather(weather_data: pd.DataFrame, weather_cols: List[str],
                   date_of_onset: str) -> pd.DataFrame:
    weather_data = weather_data[weather_cols]
    weather_data = weather_data[weather_data["date"] > date_of_onset]
    return weather_data