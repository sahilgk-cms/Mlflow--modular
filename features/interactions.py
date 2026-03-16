import pandas as pd


def add_weather_interactions(df):

    # Create binary variable for precipitation > 500
    df['High_Precip'] = (df['total_precipitation_sum_mm'] >= 170).astype(int)

    # Create binary variable for relative humidity > 67
    df['High_Humidity'] = (df['relative_humidity_percent'] > 75).astype(int)

    # Define interaction variable
    df['High_Precip_Humidity'] = df['High_Precip'] * df['High_Humidity']

    # Create binary variable for relative humidity > 67
    df['temperature_max'] = (df['temperature_2m_max_celsius'] >= 28).astype(int)

    # Define interaction variable
    df['High_temp_Humidity'] = df['temperature_max'] * df['High_Humidity']
    df['High_temp_Humidity_preci'] = df['temperature_max'] * df['High_Humidity'] * df['temperature_max']

    return df