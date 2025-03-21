# -*- coding: utf-8 -*-

from darts import TimeSeries
from darts import models as darts_models
from sinapsis_core.data_containers.data_packet import DataContainer, TimeSeriesPacket
from sinapsis_core.template_base.dynamic_template import (
    BaseDynamicWrapperTemplate,
    WrapperEntryConfig,
)
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.template_base.template import Template, TemplateAttributes
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS

EXCLUDED_MODELS = [
    "AutoARIMA",
    "StatsForecastAutoARIMA",
    "StatsForecastAutoCES",
    "StatsForecastAutoETS",
    "StatsForecastAutoTheta",
    "StatsForecastAutoTBATS",
]


class TimeSeriesModel(BaseDynamicWrapperTemplate):
    """Base Template for Darts Time Series forecasting models
    Usage example:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: XGBModelWrapper
      class_name: XGBModelWrapper
      template_input: InputTemplate
      attributes:
        forecast_horizon: 100
        xgbmodel_init:
            lags: 30
            lags_past_covariates: 30
            output_chunk_length: 100
            random_state: 42
            n_estimators: 200
            learning_rate: 0.1
            max_depth: 6
    For a full list of options use the sinapsis cli: sinapsis info --all-template-names
    If you want to see all available models, please visit: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html
    """

    CATEGORY = "TimeSeries"
    WrapperEntry = WrapperEntryConfig(wrapped_object=darts_models, exclude_module_atts=EXCLUDED_MODELS)

    class AttributesBaseModel(TemplateAttributes):
        """Base attributes for TimeSeriesModel template.

        Attributes:
            - forecast_horizon (int): Number of future time steps the model should predict.
        """

        forecast_horizon: int = 10

    @staticmethod
    def _store_forecast_results(
        time_series_packet: TimeSeriesPacket,
        target_series: TimeSeries,
        predictions: TimeSeries,
    ) -> None:
        """Stores the processed target series and generated predictions in the time series packet.

        Args:
            time_series_packet (TimeSeriesPacket): The time series packet to update.
            target_series (TimeSeries): The processed target series (historical data).
            predictions (TimeSeries): The predicted values for the future time steps.
        """
        time_series_packet.content = target_series
        time_series_packet.predictions = predictions

    def _fit_model(
        self,
        target_series: TimeSeries,
        past_covariates: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
    ) -> None:
        """Fit the selected model using historical time series data.

        Args:
            target_series (TimeSeries): Historical time series data (target variables).
            past_covariates (TimeSeries, optional): Past covariates if applicable.
            future_covariates (TimeSeries, optional): Future covariates if applicable.
        """
        fit_params = {"series": target_series}

        if past_covariates is not None:
            fit_params["past_covariates"] = past_covariates
        if future_covariates is not None:
            fit_params["future_covariates"] = future_covariates

        self.wrapped_callable.fit(**fit_params)

    def _generate_predictions(
        self,
        past_covariates: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
    ) -> TimeSeries:
        """Uses the fitted model to produce time series predictions.

        Args:
            past_covariates (TimeSeries, optional): Past covariates if applicable.
            future_covariates (TimeSeries, optional): Future covariates if applicable.

        Returns:
            TimeSeries: Time series with predicted/forecasted values.
        """
        predict_params = {"n": self.attributes.forecast_horizon}

        if past_covariates is not None:
            predict_params["past_covariates"] = past_covariates
        if future_covariates is not None:
            predict_params["future_covariates"] = future_covariates

        return self.wrapped_callable.predict(**predict_params)

    def execute(self, container: DataContainer) -> DataContainer:
        """Processes each time series packet, trains the model, and generates predictions.

        Args:
            container (DataContainer): Input data container with time series packets of historical data.

        Returns:
            DataContainer: Input data container updated with the time series predictions produced by the
                forecasting model.
        """
        for time_series_packet in container.time_series:
            target_series = time_series_packet.content
            past_covariates = time_series_packet.past_covariates
            future_covariates = time_series_packet.future_covariates
            self._fit_model(target_series, past_covariates, future_covariates)
            predictions = self._generate_predictions(past_covariates, future_covariates)
            self._store_forecast_results(time_series_packet, target_series, predictions)

        return container


def __getattr__(name: str) -> Template:
    """Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in TimeSeriesModel.WrapperEntry.module_att_names:
        return make_dynamic_template(name, TimeSeriesModel)
    raise AttributeError(f"template `{name}` not found in {__name__}")


__all__ = TimeSeriesModel.WrapperEntry.module_att_names

if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
