# -*- coding: utf-8 -*-


import pandas as pd
from darts import TimeSeries
from pydantic import Field

from sinapsis_darts_forecasting.templates.pandas_to_darts_base import FromPandasBaseKwargs, PandasToDartsLoaderBase


class FromDataFrameKwargs(FromPandasBaseKwargs):
    """Defines and validates shared parameters for creating Darts TimeSeries from Pandas objects.

    Attributes:
        fill_missing_dates (bool | None): If `True`, adds rows for missing timestamps.
            Defaults to `False`.
        freq (str | int | None): The frequency of the time series (e.g., 'D' for daily, 'H' for hourly).
        fillna_value (float | None): The value to use for filling any missing data points (NaNs).
        static_covariates (pd.Series | pd.DataFrame | None): External data that is constant
            over time for this series.
        metadata (dict | None): A dictionary for storing arbitrary metadata about the time series.
        make_copy (bool): Whether to create a copy of the underlying data. Defaults to `True`.
        time_col (str | None): The column name for the time index.
        value_cols (str | list[str] | None): The column(s) for the series values.
    """

    time_col: str | None = None
    value_cols: str | list[str] | None = None


class TimeSeriesFromDataframeLoader(PandasToDartsLoaderBase):
    """Template for converting Pandas DataFrames to darts TimeSeries objects

    Usage example:

        agent:
            name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: TimeSeriesFromDataframeLoader
          class_name: TimeSeriesFromDataframeLoader
          template_input: InputTemplate
          attributes:
            apply_to: ["content"]
            from_pandas_kwargs:
                value_cols: "volume"
                time_col: "Date"
                fill_missing_dates: True
                freq: "D"
    """

    class AttributesBaseModel(PandasToDartsLoaderBase.AttributesBaseModel):
        """Defines the attributes required for the TimeSeriesFromDataframeLoader template.

        Attributes:
            apply_to (list[Literal[...]]): A list of `TimeSeriesPacket` attributes
                to convert from Pandas objects to Darts TimeSeries.
            from_pandas_kwargs (FromPandasKwargs): Allows passing extra arguments
                specific to `from_dataframe()`.
        """

        from_pandas_kwargs: FromDataFrameKwargs = Field(default_factory=FromDataFrameKwargs)

    def _convert(self, data: pd.DataFrame | pd.Series) -> TimeSeries:
        """Converts a Pandas DataFrame to a Darts TimeSeries.

        Args:
            data (pd.DataFrame | pd.Series): The object to convert

        Returns:
            TimeSeries: The TimeSeries instance expected by the consumer DARTS template.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Input data must be a pandas.DataFrame, but got {type(data)}.")
        return TimeSeries.from_dataframe(
            data, **self.attributes.from_pandas_kwargs.model_dump(exclude_none=True, by_alias=True)
        )
