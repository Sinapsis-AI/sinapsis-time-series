# -*- coding: utf-8 -*-

from typing import Any, Literal

import numpy as np
from darts import TimeSeries
from darts.dataprocessing import transformers
from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer, TimeSeriesPacket
from sinapsis_core.template_base.dynamic_template import (
    BaseDynamicWrapperTemplate,
    WrapperEntryConfig,
)
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.template_base.template import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS


class TimeSeriesPreprocessor(BaseDynamicWrapperTemplate):
    """A dynamic preprocessing template for applying Darts transformations to time series data.

    This template allows users to apply various transformations from `darts.dataprocessing.transformers`
    to different parts of the time series packet (`content`, `past_covariates`, `future_covariates`).

    If a transformer requires fitting (e.g., `Scaler`), it stores and retrieves transformation parameters.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: MissingValuesFillerWrapper
      class_name: MissingValuesFillerWrapper
      template_input: InputTemplate
      attributes:
        method: "transform"
        apply_to: ["content"]
        transform_kwargs:
            method: "linear"
        missingvaluesfiller_init:
            fill: "auto"
            name: "MissingValuesFiller"
            n_jobs: 1
            verbose: False
    For a full list of options use the sinapsis cli: sinapsis info --all-template-names
    If you want to see all available transformers, please visit: https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.html
    """

    CATEGORY = "TimeSeries"
    WrapperEntry = WrapperEntryConfig(wrapped_object=transformers)

    _FIT_METHODS = ("fit", "fit_transform")

    class AttributesBaseModel(TemplateAttributes):
        """Defines the attributes required for the preprocessor template.

        Attributes:
            - apply_to (list[Literal["content", "past_covariates", "future_covariates", "predictions"]]):
                Specifies which attributes in `TimeSeriesPacket` should be transformed.
            - method (Literal["fit", "transform", "fit_transform", "inverse_transform"]):
                Specifies the transformation method to apply.
            - transform_kwargs (dict[str, Any]):
                Additional keyword arguments for the selected transformation method.
            - params_key (str | None):
                If provided, transformation parameters are stored/retrieved in `TimeSeriesPacket.generic_data`.
                If `None`, no parameter storage is performed (useful for stateless transformations).
        """

        apply_to: list[Literal["content", "past_covariates", "future_covariates", "predictions"]]
        method: Literal["fit", "transform", "fit_transform", "inverse_transform"]
        transform_kwargs: dict[str, Any] = Field(default_factory=dict)
        params_key: str | None = None

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.transform_func = getattr(self.wrapped_callable, self.attributes.method)

    def _get_params_key(self, attribute: str) -> str:
        """Generates the parameter storage key for a specific attribute. If `attribute` is
        `"predictions"`, it uses `"content"`'s parameters.

        Args:
            attribute (str): The time series component
                (`content`, `past_covariates`, `future_covariates`, `predictions`).

        Returns:
            str: The generated key for storing/retrieving parameters.
        """
        return f"{attribute}_{self.attributes.params_key}"

    def _store_params(self, time_series_packet: TimeSeriesPacket, attribute: str) -> None:
        """Stores transformation parameters in `generic_data` for a specific attribute.

        Args:
            time_series_packet (TimeSeriesPacket): The packet containing time series data.
            attribute (str): The attribute being processed
                (`content`, `past_covariates`, `future_covariates`, `predictions`).
        """
        if self.attributes.params_key is not None and hasattr(self.wrapped_callable, "_fitted_params"):
            key = self._get_params_key(attribute)
            time_series_packet.generic_data[key] = self.wrapped_callable._fitted_params
            self.logger.info(f"Stored transformation parameters for '{attribute}' under key '{key}'.")

    def _load_params(self, time_series_packet: TimeSeriesPacket, attribute: str) -> Any | None:
        """Loads transformation parameters from `generic_data` for a specific attribute.

        Args:
            time_series_packet (TimeSeriesPacket): The packet containing time series data.
            attribute (str): The attribute being processed (`content`, `past_covariates`, `future_covariates`).

        Returns:
            Any | None: The stored transformation parameters if available, otherwise None.
        """
        if self.attributes.params_key is None:
            return None

        key = self._get_params_key("content") if attribute == "predictions" else self._get_params_key(attribute)
        return time_series_packet.generic_data.get(key, None)

    def _fit(self, time_series_packet: TimeSeriesPacket, attribute: str) -> TimeSeries | None:
        """Applies a fitting transformation (fit or fit_transform) to the specified attribute.

        Args:
            time_series_packet (TimeSeriesPacket): The packet containing time series data.
            attribute (str): The attribute to fit (`content`, `past_covariates`, `future_covariates`).

        Returns:
            TimeSeries | None: Transformed time series, or None if no data.
        """
        series = getattr(time_series_packet, attribute)
        if series is None:
            self.logger.warning(f"No data found in '{attribute}' to fit.")
            return None

        transformed_series = self.transform_func(series, **self.attributes.transform_kwargs)

        self._store_params(time_series_packet, attribute)

        return transformed_series

    def _apply_transformation(self, time_series_packet: TimeSeriesPacket, attribute: str) -> TimeSeries | None:
        """Applies a transformation (transform or inverse_transform) to the specified attribute.

        Args:
            time_series_packet (TimeSeriesPacket): The packet containing time series data.
            attribute (str): The attribute to transform
                (`content`, `past_covariates`, `future_covariates`, `predictions`).

        Returns:
            TimeSeries | None: Transformed time series, or None if no data.
        """
        series = getattr(time_series_packet, attribute)

        if series is None:
            self.logger.warning(f"No data found in '{attribute}' to transform.")
            return None

        params = self._load_params(time_series_packet, attribute)

        if hasattr(self.wrapped_callable, "fit"):
            dummy_series = TimeSeries.from_values(np.array([0.0, 1.0, 2.0]))
            self.wrapped_callable.fit(dummy_series)

        if params is not None:
            self.wrapped_callable._fitted_params = params

        return self.transform_func(series, **self.attributes.transform_kwargs)

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the preprocessing transformations on each time series packet in the container.

        Applies transformations based on the specified `apply_to` attributes.

        Args:
            container (DataContainer): The input data container holding one or more time series packets.

        Returns:
            DataContainer: The processed container with transformed time series data.
        """
        for time_series_packet in container.time_series:
            for attribute in self.attributes.apply_to:
                if self.attributes.method in self._FIT_METHODS:
                    transformed_series = self._fit(time_series_packet, attribute)
                else:
                    transformed_series = self._apply_transformation(time_series_packet, attribute)

                setattr(time_series_packet, attribute, transformed_series)

        return container


def __getattr__(name: str) -> Template:
    """Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in TimeSeriesPreprocessor.WrapperEntry.module_att_names:
        return make_dynamic_template(name, TimeSeriesPreprocessor)
    raise AttributeError(f"Template `{name}` not found in {__name__}")


__all__ = TimeSeriesPreprocessor.WrapperEntry.module_att_names
if SINAPSIS_BUILD_DOCS:
    dynamic_templates = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
