# -*- coding: utf-8 -*-


import pandas as pd
from darts import TimeSeries

from sinapsis_darts_forecasting.templates.pandas_to_darts_base import PandasToDartsLoaderBase


class TimeSeriesFromSeriesLoader(PandasToDartsLoaderBase):
    """Template for converting Pandas Series to darts TimeSeries objects

    Usage example:

        agent:
            name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: TimeSeriesFromSeriesLoader
          class_name: TimeSeriesFromSeriesLoader
          template_input: InputTemplate
          attributes:
            apply_to: ["content"]
            from_pandas_kwargs:
                fill_missing_dates: True
                freq: "D"
                fillna_value: 0.0
    """

    def _convert(self, data: pd.DataFrame | pd.Series) -> TimeSeries:
        """Converts a Pandas Series to a Darts TimeSeries.

        Args:
            data (pd.DataFrame | pd.Series): The object to convert

        Returns:
            TimeSeries: The TimeSeries instance expected by the consumer DARTS template.
        """
        if not isinstance(data, pd.Series):
            raise TypeError(f"Input data must be a pandas.Series, but got {type(data)}.")
        data = data.to_timestamp()
        return TimeSeries.from_series(
            data, **self.attributes.from_pandas_kwargs.model_dump(exclude_none=True, by_alias=True)
        )
