# -*- coding: utf-8 -*-

from typing import Any, Literal

import pandas as pd
from darts import TimeSeries
from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer, TimeSeriesPacket
from sinapsis_core.template_base.template import Template, TemplateAttributes


class TimeSeriesDataframeLoader(Template):
    """Template for converting Pandas DataFrames to darts TimeSeries objects

    Usage example:

        agent:
            name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: TimeSeriesDataframeLoader
          class_name: TimeSeriesDataframeLoader
          template_input: InputTemplate
          attributes:
            apply_to: ["content"]
            from_dataframe_kwargs:
                value_cols: "volume"
                time_col: "Date"
                fill_missing_dates: True
                freq: "D"
    """

    CATEGORY = "TimeSeries"

    class AttributesBaseModel(TemplateAttributes):
        """Defines the attributes required for the TimeSeriesDataframeLoader template.

        Attributes:
            apply_to (Literal["content", "past_covariates", "future_covariates"]):
                Specifies which attribute in `TimeSeriesPacket` should be converted.
            from_dataframe_kwargs (dict[str, Any]):
                Additional arguments to pass to `TimeSeries.from_dataframe()`.
        """

        apply_to: list[Literal["content", "past_covariates", "future_covariates", "predictions"]]
        from_dataframe_kwargs: dict[str, Any] = Field(default_factory=dict)

    def _to_timeseries(self, time_series_packet: TimeSeriesPacket, attribute: str) -> TimeSeries | None:
        """Converts a DataFrame inside a TimeSeriesPacket to a Darts TimeSeries.

        Args:
            time_series_packet (TimeSeriesPacket): The packet containing the time series data.
            attribute (str): The attribute to convert (`"content"`, `"past_covariates"`, or `"future_covariates"`).

        Returns:
            TimeSeries | None: The converted Darts TimeSeries object, or None if no data is found.
        """
        dataframe: pd.DataFrame = getattr(time_series_packet, attribute, None)

        if dataframe is None:
            self.logger.warning(f"No data found in '{attribute}' to convert to TimeSeries.")
            return None

        return TimeSeries.from_dataframe(dataframe, **self.attributes.from_dataframe_kwargs)

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
