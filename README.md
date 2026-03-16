# Mlflow--modular

## Project structure

```bash
Mlflow--modular
├── config
│  ├── data_config.py
│  ├── env.py
│  ├── features_config.py
│  ├── filepaths.py
│  ├── ml_config.py
│  ├── search_spaces.yml
│  └── __init__.py
├── data
│  ├── data_hash.py
│  ├── metadata.py
│  ├── schema.py
│  ├── split_features_target.py
│  ├── train_test_split.py
│  └── __init__.py
├── db
│  ├── db_loader.py
│  └── engine.py
│  └── __init__.py
├── features
│  ├── aggregations.py
│  ├── data_processing.py
│  ├── interactions.py
│  ├── lag_features.py
│  ├── rolling_features.py
│  ├── schema.py
│  ├── temporal_features.py
│  ├── weather_processing.py
│  └── __init__.py
├── metrics
│  ├── factory.py
│  ├── metrics.py
│  └── __init__.py
├── models
│  ├── base.py
│  ├── catboost_model.py
│  ├── factory.py
│  └── __init__.py
├── optimizer
│  ├── base.py
│  ├── factory.py
│  ├── grid_optimizer.py
│  ├── optuna_optimizer.py
│  ├── random_optimizer.py
│  └── __init__.py
├── pipelines
│  ├── data_builder.py
│  ├── evaluation_pipeline.py
│  ├── features_builder.py
│  ├── prediction_builder.py
│  ├── train_pipeline.py
│  └── __init__.py
├── preprocessing
│  ├── base.py
│  ├── factory.py
│  ├── preprocessor_v1.py
│  └── __init__.py
├── search_space
│  ├── search_space.py
│  └── __init__.py
├── training
│  ├── cv_factory.py
│  ├── trainer.py
│  └── __init__.py
├── utils
│  ├── artifact_logger.py
│  ├── explainability.py
│  ├── hardware.py
│  ├── mlflow_helpers.py
│  └── __init__.py
├── docker-compose.yml
├── DockerFile
├── main.py
├── pyproject.toml
├── README.md
├── run_pipeline.sh
└── uv.lock
```

## Running the pipeline
- First ensure Docker Daemon is running. 
- Go to the project folder on Linux/WSL/Bash. Make the shell script executable
```
chmod +x run_pipeline.sh
```
- Run the shell script.
  ```
  ./run_pipeline.sh
  ```
