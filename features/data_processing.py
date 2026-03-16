import pandas as pd
from typing import List, Dict, Tuple
from features.schema import RENAME_MAPPING, WEATHER_COLS, MIN_DATE, LULC_COLS, RAIN_COLS, TEMPORAL_COLS


def clean_weather_data(df: pd.DataFrame, rename_mapping = RENAME_MAPPING,
                       weather_cols = WEATHER_COLS, min_date = MIN_DATE) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['date', 'sub_district'])
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date']>min_date]
    df = df[weather_cols]
    df = df.dropna()
    df = df.rename(columns=rename_mapping)
    df['year'] = df['date'].dt.year
    return df


def group_by_district(df: pd.DataFrame, lulc_cols = LULC_COLS) -> pd.DataFrame:
    df = df.groupby(['district'])[lulc_cols].sum().reset_index()
    return df

def get_temporal_and_cases_df(df: pd.DataFrame, temporal_cols = TEMPORAL_COLS,
                              rain_cols = RAIN_COLS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_temporal = (
        df.groupby(['date', 'dist_name'])[temporal_cols]
        .mean()
        .reset_index()
    )

    df_cases = (
        df.groupby(['date', 'dist_name'])[rain_cols + ['Case_Count']]
        .sum()
        .reset_index()
    )
    return df_temporal, df_cases

def fill_lagged_values(df: pd.DataFrame) -> pd.DataFrame:
    lagged_cols = [
        col for col in df.columns
        if 'lag_' in col or 'roll_' in col
    ]

    df = df.sort_values(['dist_name', 'week_start'])

    for col in lagged_cols:
        df[col] = (
            df
            .groupby('dist_name')[col]
            .transform(lambda x: x.bfill().ffill())
        )
    return df