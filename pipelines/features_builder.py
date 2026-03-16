from db.db_loader import load_weather_cases, load_lulc
from features.data_processing import clean_weather_data, group_by_district, get_temporal_and_cases_df, fill_lagged_values
from features.aggregations import aggregate_weekly
from features.lag_features import shift_cases_forward, create_lag_features
from features.rolling_features import create_rolling_features
from features.interactions import add_weather_interactions
from features.temporal_features import add_month_sin_cos
from db.db_loader import append_df_to_db
from features.schema import TEMPORAL_COLS, RAIN_COLS, LAG1, LAG2, CASE_COL
import sqlalchemy

def build_features(engine: sqlalchemy.engine.base.Engine, state: str, diagnosis: str, 
                   output_table: str, output_schema: str):

    weather_data = load_weather_cases(engine, state, diagnosis)
    weather_data = clean_weather_data(df=weather_data)

    df_lulc = load_lulc(engine, state)
    lulc_dist = group_by_district(df=df_lulc)

    df_temporal, df_cases = get_temporal_and_cases_df(df = weather_data)
    
    df_temporal = aggregate_weekly(df_temporal, features = TEMPORAL_COLS)
    
    df_cases = aggregate_weekly(df_cases, features= RAIN_COLS + [CASE_COL])

    df_temporal = df_temporal.merge(df_cases, on=['dist_name', 'week_start'], how='left')

    df_temporal = shift_cases_forward(df=df_temporal)

    df_temporal = create_lag_features(df=df_temporal, features=TEMPORAL_COLS+RAIN_COLS, lags=LAG1)
    
    df_temporal = create_lag_features(df=df_temporal, features=[CASE_COL], lags=LAG2)

    df_temporal = create_rolling_features(df=df_temporal, features=TEMPORAL_COLS + RAIN_COLS)
    
    df_temporal = create_rolling_features(df=df_temporal, features=['Case_Count_lag_2'])
    
    df_final = df_temporal.merge(lulc_dist, left_on='dist_name', right_on='district', how='left')

    df_final = add_weather_interactions(df_final)

    df_final = add_month_sin_cos(df_final)

    df_final = fill_lagged_values(df_final)

    append_df_to_db(engine=engine, df=df_final,
                    table_name=output_table, schema_name=output_schema)