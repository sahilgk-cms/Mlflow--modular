from pipelines.features_builder import build_features
from pipelines.data_builder import build_data
from db.engine import get_engine
from datetime import datetime
from preprocessing.factory import PreprocessorFactory
from utils.mlflow_helpers import start_mlflow_experiment
from utils.artifact_logger import log_parquet
from utils.mlflow_helpers import register_model_with_data_tags
from utils.explainability import log_shap_summary
from training.trainer import TimeSeriesTrainer
from pipelines.train_pipeline import run_training_pipeline
from pipelines.evaluation_pipeline import run_evaluation_pipeline
import mlflow
from config.env import DB_NAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_USER
from config.features_config import STATE, DIAGNOSIS, OUTPUT_SCHEMA, OUTPUT_TABLE
from config.data_config import DISEASE
from config.ml_config import USE_GPU, PREDICTIONS_PATH, SHAP_SUMMARY_PATH, OPTIMIZER_TYPE, SHAP_VALUES_PATH, EVALUATION_RUN_TYPE, FEATURE_IMPORTANCE_PATH, TRAINING_RUN_TYPE, PREPROCESSOR_NAME, MLFLOW_URI, EXPERIMENT_NAME, TRAIN_PATH, TEST_PATH, CV_TYPE, MODEL_NAME, N_TRIALS, N_CV_SPLITS

def main():
    engine = get_engine(db_user=DB_USER, db_password=DB_PASSWORD, db_host=DB_HOST,
                        db_port=DB_PORT, db_name=DB_NAME)
    
    # build_features(engine=engine, state=STATE, diagnosis=DIAGNOSIS, 
    #                output_table=OUTPUT_TABLE, output_schema=OUTPUT_SCHEMA)
    
    output = build_data(engine=engine, disease=DISEASE)
    X_train = output["features"]["X_train"]
    y_train = output["features"]["y_train"]
    X_test = output["features"]["X_test"]
    y_test = output["features"]["y_test"]

    pre = PreprocessorFactory.create(PREPROCESSOR_NAME)
    pre.fit(X_train)
    X_train_preprocessed = pre.transform(X_train)
    X_test_preprocessed = pre.transform(X_test)

    feature_names = pre.get_feature_names()
    cat_feature_indices = pre.get_cat_feature_indices()

    experiment = start_mlflow_experiment(mlflow_uri=MLFLOW_URI,
                                         experiment_name=EXPERIMENT_NAME)
    today_date = datetime.now().strftime("%Y/%m/%d")

    with mlflow.start_run(run_name = f"{EXPERIMENT_NAME}_pipeline_root_{today_date}") as pipeline_root:
        pipeline_root_run_id = pipeline_root.info.run_id
        mlflow.set_tags(
            {
                "preprocessor_name": PREPROCESSOR_NAME,
                "train_data_hash": output["hash"]["train_data_hash"],
                "test_data_hash": output["hash"]["test_data_hash"],
                "train_date_min": output["metadata"]["train_metadata"]["train_date_min"],
                "train_date_max": output["metadata"]["train_metadata"]["train_date_max"],
                "test_date_min": output["metadata"]["test_metadata"]["test_date_min"],
                "test_date_max": output["metadata"]["test_metadata"]["test_date_max"]

            }
        )

        log_parquet(df = output["data"]["train_df"], filename=TRAIN_PATH, artifact_path="data")
        log_parquet(df=output["data"]["test_df"], filename=TEST_PATH, artifact_path="data")


        final_model, best_cv_score, best_params, training_run_id = run_training_pipeline(
                                                                    X_train=X_train_preprocessed,
                                                                    y_train=y_train,
                                                                    model_name=MODEL_NAME,
                                                                    cv_type=CV_TYPE,
                                                                    optimizer_type=OPTIMIZER_TYPE,
                                                                    trainer_cls=TimeSeriesTrainer,
                                                                    use_gpu=USE_GPU,
                                                                    n_trials=N_TRIALS,
                                                                    n_cv_split=N_CV_SPLITS,
                                                                    run_type=TRAINING_RUN_TYPE,
                                                                    pipeline_root_run_id=pipeline_root_run_id,
                                                                    cat_feature_indices=cat_feature_indices)

        if final_model.has_feature_importance():
            importance_df = final_model.get_feature_importance(feature_names=feature_names)
            importance_df = importance_df[importance_df["feature"] != "Case_Count"]
            log_parquet(df=importance_df, filename=FEATURE_IMPORTANCE_PATH,
                         artifact_path="feature_importance")

        metric_results = run_evaluation_pipeline(X_test=X_test_preprocessed,
                                y_test = y_test,
                                X_test_meta=output["test_meta"],
                                model = final_model,
                                best_cv_score=best_cv_score,
                                predictions_path=PREDICTIONS_PATH,
                                model_name=MODEL_NAME,
                                run_type=EVALUATION_RUN_TYPE,
                                pipeline_root_run_id=pipeline_root_run_id)
        
        REGISTERED_MODEL_NAME = f"{EXPERIMENT_NAME}_{MODEL_NAME}"
        register_model_with_data_tags(
            training_run_id=training_run_id,
            model_name=REGISTERED_MODEL_NAME,
            train_data_hash=output["hash"]["train_data_hash"],
            test_data_hash=output["hash"]["test_data_hash"],
            pipeline_root_run_id=pipeline_root_run_id,
            preprocessor_name=PREPROCESSOR_NAME,
            optimizer_type=OPTIMIZER_TYPE,
            eval_metric_results=metric_results
        )


        shap_summary_path, shap_df= log_shap_summary(model_wrapper=final_model,
                                                     X_sample=X_test_preprocessed,
                                                     feature_names=feature_names,
                                                     shap_summary_path=SHAP_SUMMARY_PATH)

        log_parquet(df=shap_df, filename=SHAP_VALUES_PATH, artifact_path="explainability")
        mlflow.log_artifact(shap_summary_path, artifact_path="explainability")