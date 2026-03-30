# -*- coding: utf-8 -*-

import re

import pandas as pd
from darts import TimeSeries
from darts import metrics as darts_metrics
from sinapsis_core.data_containers.data_packet import DataContainer, DataFramePacket
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributes, UIPropertiesMetadata
from sinapsis_core.template_base.dynamic_template import (
    BaseDynamicWrapperTemplate,
    WrapperEntryConfig,
)
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.template_base.template import Template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS

from sinapsis_darts_forecasting.helpers.tags import Tags

EXCLUDED_METRICS = []
EXCLUDED_ATTRIBUTES = [
    "actual_series",
    "pred_series",
    "component_reduction",
    "series_reduction",
    "time_reduction",
]


class TimeSeriesMetrics(BaseDynamicWrapperTemplate):
    """Dynamic Template for Darts metrics computation. This template wraps any metric from the `darts.metrics` module
    and makes it available as a Sinapsis template. The wrapped metric is applied to the predictions and ground truth
    of each time series packet in the input container, and the result is stored in the `generic_data` field of the
    time series packet.

    IMPORTANT NOTE: Validation mode should be enabled during model fitting and prediction (e.g., in the
    `TimeSeriesModel` template), to ensure that the metric is computed on the correct portion of the time series
    (i.e., the forecast horizon). If validation mode is not enabled, forecasted values and ground truth values may not
    be properly aligned, which can lead to incorrect metric calculations.
    """

    UIProperties = UIPropertiesMetadata(
        category="Darts",
        output_type=OutputTypes.TIMESERIES,
        tags=[Tags.DARTS, Tags.DYNAMIC, Tags.TIME_SERIES, Tags.METRICS],
    )
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=darts_metrics,
        template_name_suffix="TimeSeriesWrapper",
        exclude_module_atts=EXCLUDED_METRICS,
        exclude_method_attributes=EXCLUDED_ATTRIBUTES,
    )

    def __init__(self, attributes: TemplateAttributes):
        super().__init__(attributes)

        # Unwrap to the original function to prevent argument collisions.
        if hasattr(self.wrapped_callable, "__self__") and self.wrapped_callable.__self__ is self:
            self.wrapped_callable = self.wrapped_callable.__func__

        self.wrapped_method_params = self.get_wrapped_method_params()

    def get_wrapped_method_params(self) -> dict:
        """Gets the parameters of the wrapped method, excluding any that are in the `exclude_method_attributes` list.

        Returns:
            dict: A dictionary of the parameters of the wrapped method, excluding any that are in the
            `exclude_method_attributes` list.
        """

        return getattr(self.attributes, self.wrapped_callable.__name__).model_dump()

    @staticmethod
    def _sanitize_component_name(component_name: str) -> str:
        """Sanitize a component name to create a valid metric key. This involves replacing non-alphanumeric characters
        with underscores, stripping leading/trailing underscores, and converting to lowercase.

        Args:
            component_name (str): The original component name to sanitize.
        Returns:
            str: The sanitized component name.
        """
        return re.sub(r"[^0-9a-zA-Z_]+", "_", component_name).strip("_").lower()

    def _compute_metrics(self, ground_truth: TimeSeries, predictions: TimeSeries) -> dict[str, float]:
        """Compute overall and per-component metrics.

        Args:
            ground_truth (TimeSeries): The ground truth time series.
            predictions (TimeSeries): The predicted time series.
        Returns:
            dict[str, float]: A dictionary containing the computed metrics.
        """
        metric_name = self.wrapped_callable.__name__
        metrics = {
            f"overall_{metric_name}": self.wrapped_callable(
                actual_series=ground_truth,
                pred_series=predictions,
                **self.wrapped_method_params,
            )
        }

        if getattr(ground_truth, "n_components", 1) > 1:
            for component in ground_truth.components:
                component_name = str(component)
                sanitized = self._sanitize_component_name(component_name)
                key = f"{metric_name}_{sanitized}" if sanitized else metric_name
                metrics[key] = self.wrapped_callable(
                    actual_series=ground_truth[component_name],
                    pred_series=predictions[component_name],
                    **self.wrapped_method_params,
                )

        return metrics

    @staticmethod
    def _metrics_to_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
        """Convert a metrics dictionary into a single-row DataFrame.

        Args:
            metrics (dict[str, float]): A dictionary of metrics to convert.
        Returns:
            pd.DataFrame: A single-row DataFrame containing the metrics.
        """
        return pd.DataFrame([metrics])

    def execute(self, container: DataContainer) -> DataContainer:
        """Run the wrapped metric on predictions and ground truth for each time series packet.

        Args:
            container (DataContainer): The input data container containing time series packets with predictions and
            ground truth.
        Returns:
            DataContainer: The input container with computed metrics added as DataFramePackets in the `data_frames`
            field.
        """

        for time_series_packet in container.time_series:
            predictions = time_series_packet.predictions
            if predictions is None:
                self.logger.warning(
                    f"No predictions found for time series packet with id {time_series_packet.id}, skipping."
                )
                continue

            ground_truth = time_series_packet.content[-len(predictions) :]
            metrics = self._compute_metrics(ground_truth, predictions)
            container.data_frames.append(DataFramePacket(content=self._metrics_to_dataframe(metrics)))

        return container


def __getattr__(name: str) -> Template:
    """Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in TimeSeriesMetrics.WrapperEntry.module_att_names:
        return make_dynamic_template(name, TimeSeriesMetrics)
    raise AttributeError(f"template `{name}` not found in {__name__}")


__all__ = TimeSeriesMetrics.WrapperEntry.module_att_names

if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
