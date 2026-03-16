
import pandas as pd
from typing import List
from features.schema import DATE_COL_FOR_LAG, GROUP_COL, CASE_COL, SHIFT_BY

def shift_cases_forward(df: pd.DataFrame, case_col = CASE_COL, 
                        group_col = GROUP_COL,  shift_by = SHIFT_BY) -> pd.DataFrame:
    """
    Shifts case counts forward by `shift_by` weeks for each group.
    Used to align weather(t) → cases(t+1).

    Args:
        df (pd.DataFrame): Aggregated data.
        group_col (str): Grouping column (e.g. 'taluk_name').
        case_col (str): Column with weekly case counts.
        shift_by (int): Number of weeks to shift forward.

    Returns:
        pd.DataFrame: DataFrame with shifted target column.
    """
    df = df.sort_values([group_col, 'week_start'])
    df[f'{case_col}_next_week'] = (
        df.groupby(group_col)[case_col].shift(-shift_by)
    )
    return df


def create_lag_features(df: pd.DataFrame, features: List[str], lags: List[int],
                         group_col = GROUP_COL, date_col = DATE_COL_FOR_LAG) -> pd.DataFrame:
    """
    Create lag features for the specified columns per group.

    Args:
        df (pd.DataFrame): Time-indexed data.
        group_col (str): Group column (e.g. 'taluk_name').
        date_col (str): Time column (e.g. 'week_start').
        features (list): Feature columns to lag.
        lags (list): List of lag values in weeks.

    Returns:
        pd.DataFrame: With lagged columns added.
    """
    df = df.sort_values([group_col, date_col])
    for feat in features:
        for lag in lags:
            df[f"{feat}_lag_{lag}"] = df.groupby(group_col)[feat].shift(lag)
    return df



