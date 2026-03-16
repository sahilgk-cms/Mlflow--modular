from utils.hardware import detect_gpu


EXPERIMENT_NAME = "Andhra_Pradesh_project"


# TODO: Change this ip to wherever its hosted
MLFLOW_URI = "http://host.docker.internal:5000/"

ROLLING_WINDOW_MONTHS = None  # Keep last 24 months, set to None for all data
N_TRIALS = 2
N_CV_SPLITS = 5

PREPROCESSOR_NAME = "tabular_v1"

MODEL_NAME = "CatBoostRegressor"



TRAINING_RUN_TYPE = "training"
EVALUATION_RUN_TYPE = "evaluation"
CV_TYPE = "TimeSeriesSplit"
#OPTIMIZER_TYPE = "optuna"
OPTIMIZER_TYPE = "grid"
#OPTIMIZER_TYPE = "random"


TRAIN_PATH = "train_dataset.parquet"
TEST_PATH = "test_dataset.parquet"
FEATURE_IMPORTANCE_PATH = "feature_importance.parquet"
PREDICTIONS_PATH = "predictions.parquet"
SHAP_VALUES_PATH = "shap_values.parquet"
SHAP_SUMMARY_PATH =  "shap_summary.png"

GPU_INFO = detect_gpu()
USE_GPU = GPU_INFO["available"]

