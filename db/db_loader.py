import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from typing import List, Tuple



def load_weather_cases(engine: sqlalchemy.engine.base.Engine,
                       state: str, diagnosis: str) -> pd.DataFrame:
    query = text("""
        SELECT *
        FROM dev_silver.weather_merged_sub_district
        WHERE state = :state
        AND confirmed_diagnosis = :diagnosis
    """)

    return pd.read_sql_query(
        query,
        engine,
        params={"state": state, "diagnosis": diagnosis}
    )

def load_lulc(engine: sqlalchemy.engine.base.Engine, state: str) -> pd.DataFrame:

    query = text("""
        SELECT *
        FROM dev_silver.lulc
        WHERE state = :state
    """)

    return pd.read_sql_query(query, engine, params={"state": state})


def load_training_data(engine: sqlalchemy.engine.base.Engine, disease:str) -> pd.DataFrame:
    query = text("""
    SELECT *
    FROM dev_gold.ap_final_1
    WHERE disease = :diagnosis
    """)

    df = pd.read_sql_query(
        query,
        engine,
        params={"diagnosis": disease}
    )

    df['week_start'] = pd.to_datetime(df['week_start'])

    return df


def append_df_to_db( engine:sqlalchemy.engine.base.Engine, df: pd.DataFrame,
                     table_name: str, schema_name: str):

    if df.empty:
        raise ValueError("df is empty; nothing to append.")

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        df.to_sql(
            table_name,
            conn,
            schema=schema_name,
            if_exists="append",
            index=False,
            method="multi",
        )
