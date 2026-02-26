# -*- coding: utf-8 -*-

import pickle
from typing import Any

import numpy as np
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import UIPropertiesMetadata
from sinapsis_data_analysis.helpers.model_metrics import ModelMetrics
from sinapsis_data_analysis.templates.training.ml_base_training import MLBaseAttributes, MLBaseTraining
from sinapsis_sktime.helpers.tags import Tags
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


class SKTimeModelsBase(MLBaseTraining):
    """Base class for all sktime models templates

    This class extends MLBase to work with time series data,
    providing common functionality for different time series learning tasks

    """

    TASK_TYPE: str

    AttributesBaseModel = MLBaseAttributes

    UIProperties = UIPropertiesMetadata(
        category="SKTime", tags=[Tags.DYNAMIC, Tags.INFERENCE, Tags.MODELS, Tags.PREDICTION]
    )

    @staticmethod
    def calculate_time_series_metrics(y_true: Any, y_pred: Any) -> ModelMetrics:
        """Calculate metrics specific to time series models

        Args:
            y_true (Any): The ground truth values
            y_pred (Any): The predicted values

        Returns:
            ModelMetrics: A metrics object containing time series specific metrics.
        """
        metrics = ModelMetrics()
        metrics.mape = float(mean_absolute_percentage_error(y_true, y_pred))
        return metrics

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """
        Uses the specified TASK_TYPE to determine which metrics to calculate
        for the model's predictions

        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values

        Returns:
            ModelMetrics: A metrics object containing the appropriate metrics
                for the task type
        """
        if self.TASK_TYPE == "forecasting":
            return self.calculate_time_series_metrics(y_true, y_pred)
        elif self.TASK_TYPE == "classification":
            return super().calculate_classification_metrics(y_true, y_pred)
        elif self.TASK_TYPE == "regression":
            return super().calculate_regression_metrics(y_true, y_pred)

    def _save_model_implementation(self, full_path: str) -> None:
        """Save the trained forecasting model using pickle"""
        with open(full_path, "wb") as f:
            pickle.dump(self.trained_model, f)
        self.logger.info(f"Forecasting model saved at: {full_path}")

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Processes each time series packet in the container, trains the model,
        makes predictions, and stores the results

        Args:
            container (DataContainer): The container with time series data

        Returns:
            DataContainer: The container with added predictions and metrics
        """
        if not container.time_series:
            self.logger.warning("No time series data found in container")
            return container

        processed_data = self.process_dataset(container.time_series)

        results = self.handle_model_training(processed_data)
        for time_packet in container.time_series:
            time_packet.predictions = results.predictions
            time_packet.source = self.instance_name
        self._set_generic_data(container, results.metrics.model_dump())

        self.save_model()
        return container
