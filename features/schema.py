WEATHER_COLS = ['date', 'sub_district', 'district', 'state', 'dewpoint_temperature',
        'maximum_temperature', 'mean_temperature', 'minimum_temperature',
        'relative_humidity', 'total_precipitation', 'confirmed_diagnosis',
        'no_of_cases']

MIN_DATE = '2022-12-31'

RENAME_MAPPING = {
    'minimum_temperature': 'temperature_2m_min_celsius',
    'maximum_temperature': 'temperature_2m_max_celsius',
    'mean_temperature': 'temperature_2m_mean_celsius',
    'dewpoint_temperature': 'temperature_2m_dewpoint_celsius',
    'relative_humidity': 'relative_humidity_percent',
    'total_precipitation': 'total_precipitation_sum_mm',
    'no_of_cases': 'Case_Count',
    'district' : 'dist_name'

}

LULC_COLS = ['water', 'trees',
       'flooded_vegetation', 'crops', 'built_area', 'bare_ground', 'snow_ice',
       'clouds', 'rangeland']

DATE_COL_FOR_AGG = "date"
GROUP_COL = "dist_name"
DATE_COL_FOR_LAG = "week_start"


TEMPORAL_COLS = ['temperature_2m_min_celsius', 'temperature_2m_max_celsius',
       'temperature_2m_mean_celsius', 'temperature_2m_dewpoint_celsius',
       'relative_humidity_percent']

RAIN_COLS = ['total_precipitation_sum_mm']

CASE_COL = "Case_Count"

SHIFT_BY = 2

LAG1 = [1,3,4,5]
LAG2 = [2,4,5,6]

STATIC_COLS = ['dist_name', "agro_zones",  "rural_pop_density_per_sqkm", 'water', 'trees',
       'flooded_vegetation', 'crops', 'built_area', 'bare_ground', 'snow_ice',
       'clouds', 'rangeland',"urban_pop_density_per_sqkm", "rural_population", "urban_population", "total_population"]


STATIC_COLS_2 = ['water', 'trees',
       'flooded_vegetation', 'crops', 'built_area', 'bare_ground', 'snow_ice',
       'clouds', 'rangeland', "rural_population", "urban_population", "total_population"]

STATIC_COLS_3 = ['rural_pop_density_per_sqkm', 'urban_pop_density_per_sqkm']


ROLLING_WINDOW = [10, 20, 30, 40, 50]