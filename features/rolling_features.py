import pandas as pd
from typing import List
from features.schema import ROLLING_WINDOW, DATE_COL_FOR_LAG, GROUP_COL

def create_rolling_features(df: pd.DataFrame, features: List[str], group_col = GROUP_COL,
                             date_col = DATE_COL_FOR_LAG, windows = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Creates rolling mean and std features for specified features per group.

    Args:
        df (pd.DataFrame): DataFrame with time-ordered data.
        group_col (str): Group identifier (e.g., 'taluk_name').
        date_col (str): Time column (e.g., 'week_start').
        features (list): Columns to apply rolling stats on.
        windows (list): List of window sizes (in weeks).

    Returns:
        pd.DataFrame: With additional rolling mean/std columns.
    """
    df = df.sort_values([group_col, date_col])
    for feat in features:
        for win in windows:
            df[f"{feat}_roll_mean_{win}w"] = (
                df.groupby(group_col)[feat].transform(lambda x: x.rolling(window=win, min_periods=win).mean())
            )
            df[f"{feat}_roll_std_{win}w"] = (
                df.groupby(group_col)[feat].transform(lambda x: x.rolling(window=win, min_periods=win).std())
            )
    return df