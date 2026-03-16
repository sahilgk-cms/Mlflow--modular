import mlflow
import pandas as pd
from metrics.factory import MetricFactory
from training.cv_factory import CVFactory
from search_space.search_space import get_search_space
from optimizer.factory import OptimizerFactory
from models.factory import ModelFactory

def run_training_pipeline(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        cv_type: str,
        optimizer_type: str,
        trainer_cls,
        use_gpu: bool,
        n_trials: int,
        n_cv_split: int,
        run_type: str,
        pipeline_root_run_id,
        cat_feature_indices
    ):
    tscv = CVFactory.create(cv_type, n_splits = n_cv_split)

    optimization_metric = MetricFactory.get_optimize_metric(model_name)


    param_space = get_search_space(model_name, optimizer_type)

    optimizer_cls = OptimizerFactory.create(optimizer_type)

    model_cls, model_kwargs = ModelFactory.get_model(
        model_name=model_name,
        cat_feature_indices=cat_feature_indices
    )

    with mlflow.start_run(run_name=f"{model_name}_{optimizer_type}_{run_type}", nested=True):

        mlflow.set_tags({
            "model_name": model_name,
            "run_name": run_type,
            "cv_type": cv_type,
            "pipeline_root_run_id": pipeline_root_run_id,
            "optimizer_type": optimizer_type,
            "optimization_metric": optimization_metric.name,
        })

        mlflow.log_params({
            "n_trials": n_trials,
            "n_cv_split": n_cv_split,
           # "rolling_window_months": ro,
            "gpu_used": use_gpu,
        })

        # Trainer knows CV + metric
        trainer = trainer_cls(
            model_cls=model_cls,
            cv=tscv,
            metric=optimization_metric,
        )

        optimizer = optimizer_cls(
            trainer=trainer,
            param_space=param_space,
            n_trials=n_trials,
            direction=optimization_metric.direction,
        )

        study = optimizer.optimize(X=X_train, y=y_train, **model_kwargs)

        best_score = study["best_score"]
        best_params = study["best_params"]

        mlflow.log_metric(f"best_cv_{optimization_metric.name}", best_score)
        mlflow.log_params(best_params)

        # Train final model on full training set
        final_model = model_cls.from_params(best_params, **model_kwargs)
        final_model.fit(X_train, y_train)

        final_model.log_to_mlflow()

        training_run_id = mlflow.active_run().info.run_id

        return final_model, best_score, best_params, training_run_id
