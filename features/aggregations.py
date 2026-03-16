import pandas as pd
from typing import List
from features.schema import DATE_COL_FOR_AGG, GROUP_COL



def aggregate_weekly(df: pd.DataFrame, features: List[str],
                      date_col = DATE_COL_FOR_AGG,  group_col = GROUP_COL) -> pd.DataFrame:
    """
    Aggregates temporal features to weekly resolution per group.

    Args:
        df (pd.DataFrame): Input data.
        date_col (str): Column with datetime values.
        group_col (str): Column to group by (e.g. 'taluk_name').
        features (list): Temporal features to average.

    Returns:
        pd.DataFrame: Aggregated weekly values per group.
    """
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
    df['week_start'] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit='D')

    weekly = (
        df.groupby([group_col, 'week_start'])[features]
        .median()
        .reset_index()
        .sort_values([group_col, 'week_start'])
    )
    return weekly

