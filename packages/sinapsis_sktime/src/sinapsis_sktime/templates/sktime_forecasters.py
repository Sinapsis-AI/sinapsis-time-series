# -*- coding: utf-8 -*-
from typing import Any

from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import UIPropertiesMetadata
from sinapsis_core.template_base.dynamic_template import WrapperEntryConfig
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS
from sinapsis_data_analysis.helpers.model_metrics import (
    ModelPredictionResults,
)
from sinapsis_sktime.helpers.tags import Tags
from sinapsis_sktime.templates.sktime_models_base import (
    SKTimeModelsBase,
)
from sktime.forecasting import adapters, arch, arima, exp_smoothing, naive, theta, trend
from sktime.forecasting.base import ForecastingHorizon

EXCLUDED_MODELS = [
    "ForecastHorizonCurveFitForecaster",
    "HCrystalBallAdapter",
]


class SKTimeForecasterAttributes(SKTimeModelsBase.AttributesBaseModel):
    """Attributes for sktime forecaster templates

    Attributes:
        n_steps_ahead (int): Number of steps ahead to forecast. Defaults to 37.
    """

    n_steps_ahead: int = 37


class SKTimeArimaForecaster(SKTimeModelsBase):
    """
    This template specifically handles sktime series forecasting tasks
    where the goal is to train a model and predict future values of the input time series

    Attributes:
        WrapperEntry (WrapperEntryConfig): Configuration for dynamic template generation
        TASK_TYPE (str): The type of task (forecasting)
        CATEGORY (str): The category of the template
        AttributesBaseModel: Reference to the attributes' model class
    """

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=arima,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
    )

    TASK_TYPE = "forecasting"

    AttributesBaseModel = SKTimeForecasterAttributes

    UIProperties = UIPropertiesMetadata(
        category="SKTime", tags=[Tags.DYNAMIC, Tags.INFERENCE, Tags.MODELS, Tags.PREDICTION, Tags.FORECASTING]
    )

    def train_model(self, x_train: Any, y_train: Any) -> None:
        """
        Fits the model to the training data and stores the trained model

        Args:
            x_train (Any): Index of the values for the training data
            y_train (Any): The training data for the forecaster
        """
        _ = x_train
        self.trained_model = self.model.fit(y_train)

    def generate_predictions(self, x_test: Any, y_test: Any) -> ModelPredictionResults:
        """
        Creates a ForecastingHorizon from the test data's index and
        uses it to make predictions, then calculates metrics

        Args:
            x_test (Any): index of the test data values
            y_test (Any): The test data with the desired prediction indices

        Returns:
            ModelPredictionResults: Object containing predictions and metrics
        """
        _ = x_test
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        predictions = self.trained_model.predict(fh)

        metrics = self.calculate_metrics(y_test, predictions)

        return ModelPredictionResults(predictions=predictions, metrics=metrics)

    def handle_model_training(self, processed_data: tuple) -> ModelPredictionResults:
        """
        Extracts the relevant parts of the processed data, trains the model,
        and generates predictions

        Args:
            processed_data (tuple): Tuple containing training and test data

        Returns:
            ModelPredictionResults: Object containing predictions and metrics
        """
        x_train, y_train, x_test, y_test = processed_data
        self.train_model(x_train, y_train)
        return self.generate_predictions(x_test, y_test)


class SKTimeAdaptersForecaster(SKTimeArimaForecaster):
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=adapters,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        exclude_module_atts=["HCrystalBallAdapter"],
        force_init_as_method=False,
    )


class SKTimeArchForecaster(SKTimeArimaForecaster):
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=arch,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
    )


class SKTimeExpSmoothingForecaster(SKTimeArimaForecaster):
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=exp_smoothing,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
    )


class SKTimeNaiveForecaster(SKTimeArimaForecaster):
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=naive,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
    )


class SKTimeThetaForecaster(SKTimeArimaForecaster):
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=theta,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
    )


class SKTimeTrendForecaster(SKTimeArimaForecaster):
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=trend,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        exclude_module_atts=["CurveFitForecaster"],
        force_init_as_method=False,
    )


def __getattr__(name: str) -> Template:
    """
    Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in SKTimeAdaptersForecaster.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeAdaptersForecaster)
    if name in SKTimeArimaForecaster.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeArimaForecaster)

    if name in SKTimeArchForecaster.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeArchForecaster)

    if name in SKTimeExpSmoothingForecaster.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeExpSmoothingForecaster)

    if name in SKTimeNaiveForecaster.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeNaiveForecaster)
    if name in SKTimeThetaForecaster.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeThetaForecaster)

    if name in SKTimeTrendForecaster.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeTrendForecaster)

    raise AttributeError(f"template `{name}` not found in {__name__}")


__all__ = (
    SKTimeAdaptersForecaster.WrapperEntry.module_att_names
    + SKTimeArchForecaster.WrapperEntry.module_att_names
    + SKTimeArimaForecaster.WrapperEntry.module_att_names
    + SKTimeExpSmoothingForecaster.WrapperEntry.module_att_names
    + SKTimeNaiveForecaster.WrapperEntry.module_att_names
    + SKTimeThetaForecaster.WrapperEntry.module_att_names
    + SKTimeTrendForecaster.WrapperEntry.module_att_names
)

if SINAPSIS_BUILD_DOCS:
    dynamic_templates: list[Template] = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
