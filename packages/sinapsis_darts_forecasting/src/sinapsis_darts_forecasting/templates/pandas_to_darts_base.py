# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Literal

import pandas as pd
from darts import TimeSeries
from pydantic import BaseModel, ConfigDict, Field
from sinapsis_core.data_containers.data_packet import DataContainer, TimeSeriesPacket
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributes, UIPropertiesMetadata
from sinapsis_core.template_base.template import Template

from sinapsis_darts_forecasting.helpers.tags import Tags


class FromPandasBaseKwargs(BaseModel):
    """Defines and validates shared parameters for creating Darts TimeSeries from Pandas objects.

    Attributes:
        fill_missing_dates (bool | None): If `True`, adds rows for missing timestamps.
            Defaults to `False`.
        freq (str | None): The frequency of the time series (e.g., 'D' for daily, 'H' for hourly).
        fillna_value (float | None): The value to use for filling any missing data points (NaNs).
        static_covariates (pd.Series | pd.DataFrame | None): External data that is constant
            over time for this series.
        metadata (dict | None): A dictionary for storing arbitrary metadata about the time series.
        make_copy (bool): Whether to create a copy of the underlying data. Defaults to `True`.
    """

    fill_missing_dates: bool = False
    freq: str | None = None
    fillna_value: float | None = None
    static_covariates: pd.Series | pd.DataFrame | None = None
    metadata: dict | None = None
    make_copy: bool = Field(
        default=True, serialization_alias="copy", description="Whether to create a copy of the underlying data."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PandasToDartsLoaderBase(Template):
    """Abstract base class for converting Pandas objects (Series or DataFrame) to Darts TimeSeries."""

    UIProperties = UIPropertiesMetadata(
        category="Darts",
        output_type=OutputTypes.TIMESERIES,
        tags=[Tags.DARTS, Tags.DATA, Tags.DATAFRAMES, Tags.PANDAS, Tags.TIME_SERIES],
    )

    class AttributesBaseModel(TemplateAttributes):
        """Defines the attributes required for the TimeSeriesFromDataframeLoader template.

        Attributes:
            apply_to (list[Literal[...]]): A list of `TimeSeriesPacket` attributes
                to convert from Pandas objects to Darts TimeSeries.
            from_pandas_kwargs (FromPandasKwargs): Allows passing extra arguments
                specific to `from_dataframe()` or `from_series`.
        """

        apply_to: list[Literal["content", "past_covariates", "future_covariates", "predictions"]]
        from_pandas_kwargs: FromPandasBaseKwargs = Field(default_factory=FromPandasBaseKwargs)

    @abstractmethod
    def _convert(self, data: pd.DataFrame | pd.Series) -> TimeSeries:
        """Converts a Pandas object to a Darts TimeSeries.

        Subclasses must implement this method depending on the input data type used.

        Args:
            data (pd.DataFrame | pd.Series): The object to convert

        Returns:
            TimeSeries: The TimeSeries instance expected by the consumer DARTS template.
        """

    def _to_timeseries(self, time_series_packet: TimeSeriesPacket, attribute: str) -> TimeSeries | None:
        """Converts a DataFrame inside a TimeSeriesPacket to a Darts TimeSeries.

        Args:
            time_series_packet (TimeSeriesPacket): The packet containing the time series data.
            attribute (str): The attribute to convert (`"content"`, `"past_covariates"`, or `"future_covariates"`).

        Returns:
            TimeSeries | None: The converted Darts TimeSeries object, or None if no data is found.
        """
        data: pd.DataFrame | pd.Series | None = getattr(time_series_packet, attribute, None)

        if data is None:
            self.logger.warning(f"No data found in '{attribute}' to convert to TimeSeries.")
            return None

        return self._convert(data)

    def execute(self, container: DataContainer) -> DataContainer:
        """Processes each time series packet and converts DataFrames to Darts TimeSeries.

        Args:
            container (DataContainer): The input data container with time series packets.

        Returns:
            DataContainer: Updated data container with converted TimeSeries objects.
        """

        for time_series_packet in container.time_series:
            for attribute in self.attributes.apply_to:
                converted_series = self._to_timeseries(time_series_packet, attribute)
                setattr(time_series_packet, attribute, converted_series)

        return container
