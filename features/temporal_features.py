import numpy as np
import pandas as pd


def add_month_sin_cos(df: pd.DataFrame, date_col: str = 'week_start', 
                      inplace: bool = False) -> pd.DataFrame:
    """
    Add cyclical month features to df: 'month_sin', 'month_cos'.
    Does NOT drop the original date_col.
    If inplace=False returns a new DataFrame; else modifies df and returns it.
    """
    if not inplace:
        df = df.copy()
    # ensure datetime (non-destructive if already datetime)
    df[date_col] = pd.to_datetime(df[date_col])
    month = df[date_col].dt.month.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    return df