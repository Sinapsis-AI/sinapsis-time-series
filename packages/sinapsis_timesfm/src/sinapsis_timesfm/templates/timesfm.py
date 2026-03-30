# -*- coding: utf-8 -*-
from typing import Any, Literal, Type
from unittest.mock import patch

import numpy as np
import pandas as pd
import timesfm
import torch
from darts import TimeSeries
from pydantic import BaseModel, Field
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.template_base.template import Template
from timesfm import ForecastConfig

from sinapsis_timesfm.helpers.tags import Tags


def _format_annotation(annotation: Any) -> str:
    """Convert a type annotation to a compact readable string for docs.

    Args:
        annotation: The type annotation to format.
    Returns:
        A string representation of the annotation with common prefixes removed.
    """
    text = str(annotation)

    text_parts = text.split(".")
    if len(text_parts) > 1:
        text = text_parts[-1]

    text = text.replace("typing.", "")
    text = text.replace("<class '", "")
    text = text.replace("'>", "")
    return text


def _build_attributes_doc(attributes_model: Type[BaseModel]) -> str:
    """Build an `Attributes` doc section from a Pydantic model definition.

    Args:
        attributes_model: The Pydantic model class representing the attributes.
    Returns:
        A string containing the formatted attributes documentation.
    """
    lines = ["Attributes:"]
    for field_name, field_info in attributes_model.model_fields.items():
        if field_name in {"metadata"}:
            continue
        annotation = _format_annotation(field_info.annotation)
        description = field_info.description or "No description provided."

        if field_info.default_factory is not None:
            default_repr = f"factory={field_info.default_factory.__name__}"
        elif field_info.is_required():
            default_repr = "required"
        else:
            default_repr = repr(field_info.default)

        lines.append(f"    {field_name} ({annotation}): {description} Default: {default_repr}.")

    return "\n".join(lines)


def _append_attributes_doc(base_doc: str | None, attributes_model: Type[BaseModel]) -> str:
    """Append generated attributes documentation to an existing docstring.

    Args:
        base_doc: The existing docstring to append to.
        attributes_model: The Pydantic model class representing the attributes.
    Returns:
        A string containing the combined docstring with appended attributes documentation.
    """
    base = (base_doc or "").rstrip()
    return f"{base}\n\n{_build_attributes_doc(attributes_model)}".strip()


class ForecastConfigBM(BaseModel, ForecastConfig):
    """Pydantic wrapper for TimesFM ForecastConfig."""


class TimesFMAttributes(TemplateAttributes):
    """Configuration attributes for the TimesFM forecasting template."""

    model_name: str = Field(
        default="google/timesfm-2.5-200m-pytorch",
        description="Hugging Face model identifier for TimesFM.",
    )
    forecasting_config: ForecastConfigBM = Field(
        default_factory=ForecastConfigBM,
        description="Forecasting behavior configuration passed to TimesFM.compile().",
    )
    forecast_horizon: int = Field(
        default=12,
        description="Number of future steps to predict for each input series.",
    )
    validation_mode: bool = Field(
        default=False,
        description="If true, reserve the last `forecast_horizon` points for validation.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device used for inference, for example 'cpu' or 'cuda:0'.",
    )
    time_column_name: str = Field(
        default="Date",
        description="Name of the time column in dataframe content.",
    )
    output_format: Literal["dataframe", "darts_time_series"] = Field(
        default="darts_time_series",
        description="Format of the output forecasts. 'dataframe' returns a pandas DataFrame, while "
        "'darts_time_series' returns a Darts TimeSeries object.",
    )


class TimesFM(Template):
    """Template for time series forecasting using the TimesFM model."""

    UIProperties = UIPropertiesMetadata(
        category="TimesFM",
        output_type=OutputTypes.TIMESERIES,
        tags=[Tags.TIMESFM, Tags.FORECASTING, Tags.TIME_SERIES, Tags.DATAFRAMES, Tags.PANDAS],
    )

    AttributesBaseModel = TimesFMAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initialize the TimesFM model and compile with configured forecast settings."""
        super().__init__(attributes)

        requested_device = str(self.attributes.device).strip().lower()
        if requested_device == "cpu":
            with patch("torch.cuda.is_available", return_value=False):
                self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.attributes.model_name)
        else:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.attributes.model_name)

        target_device = torch.device(self.attributes.device)
        if self.model.model.device != target_device:
            self.model.model.device = target_device
            self.model.model.to(target_device)

        self.model.compile(self.attributes.forecasting_config)

    @staticmethod
    def prepare_inputs(df: pd.DataFrame) -> list[np.ndarray]:
        """Convert multivariate dataframe values into the list-of-series format expected by TimesFM.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the input time series data.

        Returns:
            list[np.ndarray]: A list of numpy arrays, each representing a time series. The first column of the dataframe
            is assumed to be the time column and is excluded from the output.
        """
        data_arrays = list(df.iloc[:, 1:].T.values)
        return [*data_arrays]

    def _build_forecast_time_values(self, time_values: pd.Series, horizon: int) -> pd.Index:
        """Build forecast time index from the last values in the training series.

        Args:
            time_values (pd.Series): The time column values from the training dataframe.
            horizon (int): The forecast horizon (number of future time steps to predict).
        Returns:
            pd.Index: An index representing the time values for the forecasted points.
        """
        time_col = self.attributes.time_column_name

        freq = pd.infer_freq(time_values)
        if freq is not None:
            return pd.Index(
                pd.date_range(start=time_values.iloc[-1], periods=horizon + 1, freq=freq)[1:],
                name=time_col,
            )

        return pd.RangeIndex(start=len(time_values), stop=len(time_values) + horizon, name=time_col)

    def _get_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slice the dataframe to exclude validation horizon when validation mode is active.

        Args:
            df (pd.DataFrame): The input dataframe containing the time series data.
        Returns:
            pd.DataFrame: The sliced dataframe to be used for training, excluding the validation horizon if applicable.
        """
        if self.attributes.validation_mode:
            return df.iloc[: -self.attributes.forecast_horizon]
        return df

    def _build_forecast_df(self, train_df: pd.DataFrame, point_forecast: np.ndarray) -> pd.DataFrame:
        """Build a forecast DataFrame from model output and training metadata.

        Args:
            train_df (pd.DataFrame): The training dataframe used for forecasting.
            point_forecast (np.ndarray): The model's point forecast output.
        Returns:
            pd.DataFrame: A dataframe containing the forecasted values with the appropriate time index.
        """
        series_names = [col for col in train_df.columns if col != self.attributes.time_column_name]
        point_forecast_arr = np.asarray(point_forecast)
        forecast_time = self._build_forecast_time_values(
            train_df[self.attributes.time_column_name], point_forecast_arr.shape[1]
        )
        forecast_df = pd.DataFrame(point_forecast_arr.T, columns=series_names)
        forecast_df.insert(0, self.attributes.time_column_name, forecast_time)
        return forecast_df

    @property
    def _uses_darts_output(self) -> bool:
        """Check if the output format is set to Darts TimeSeries."""
        return self.attributes.output_format == "darts_time_series"

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame | TimeSeries:
        """Convert DataFrame to the configured output format.

        Args:
            df (pd.DataFrame): The forecast dataframe to format.
        Returns:
            pd.DataFrame or TimeSeries: The forecast in the configured output format.
        """
        if self._uses_darts_output:
            return self._to_darts_time_series(df)
        return df

    def _normalize_input_to_dataframe(self, input_data: pd.DataFrame | TimeSeries) -> pd.DataFrame | None:
        """Convert supported packet content types into a pandas dataframe.

        Args:
            input_data: A pandas DataFrame or Darts TimeSeries containing the time series data.
        Returns:
            pd.DataFrame or None: A pandas DataFrame if the input data is valid, otherwise None.
        """
        if isinstance(input_data, pd.DataFrame):
            # sort rows by time column if it exists
            if self.attributes.time_column_name in input_data.columns:
                input_data = input_data.sort_values(by=self.attributes.time_column_name)
            return input_data

        if isinstance(input_data, TimeSeries):
            df = input_data.to_dataframe()
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: self.attributes.time_column_name}, inplace=True)
            return df

        self.logger.warning(
            "Time series packet content is not a pandas DataFrame or Darts TimeSeries. Skipping this packet."
        )
        return None

    def _to_darts_time_series(self, df: pd.DataFrame) -> TimeSeries:
        """Convert a DataFrame with a time column into a Darts TimeSeries.

        Args:
            df (pd.DataFrame): The dataframe to convert.
        Returns:
            TimeSeries: The converted Darts TimeSeries.
        """
        ts_df = df.set_index(self.attributes.time_column_name)
        ts_df.index = pd.to_datetime(ts_df.index)
        return TimeSeries.from_dataframe(ts_df)

    def execute(self, container: DataContainer) -> DataContainer:
        """Run TimesFM forecasting for each time series packet in the container.

        Args:
            container (DataContainer): The input data container containing time series packets with content to forecast.
        Returns:
            DataContainer: The input container with forecasted values added as predictions in each time series packet,
            and optionally converted to Darts TimeSeries if configured.
        """
        for time_series_packet in container.time_series:
            normalized_data = self._normalize_input_to_dataframe(time_series_packet.content)
            if normalized_data is None:
                continue

            train_df = self._get_training_data(normalized_data)
            point_forecast, _ = self.model.forecast(
                horizon=self.attributes.forecast_horizon,
                inputs=self.prepare_inputs(train_df),
            )

            time_series_packet.predictions = self._format_output(self._build_forecast_df(train_df, point_forecast))
            if self._uses_darts_output:
                time_series_packet.content = self._to_darts_time_series(normalized_data)

        return container


TimesFM.__doc__ = _append_attributes_doc(TimesFM.__doc__, TimesFMAttributes)
