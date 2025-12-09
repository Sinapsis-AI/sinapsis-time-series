# -*- coding: utf-8 -*-

import os
from typing import Literal

from darts import TimeSeries
from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer, TimeSeriesPacket
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributes, UIPropertiesMetadata
from sinapsis_core.template_base.template import Template

from sinapsis_darts_forecasting.helpers.tags import Tags
from sinapsis_darts_forecasting.templates.time_series_from_dataframe_loader import FromDataFrameKwargs


class FromCSVKwargs(FromDataFrameKwargs):
    """Defines and validates parameters for creating Darts TimeSeries directly from CSV files.

    Attributes:
        fill_missing_dates (bool | None): If `True`, adds rows for missing timestamps.
            Defaults to `False`.
        freq (str | None): The frequency of the time series (e.g., 'D' for daily, 'H' for hourly).
        fillna_value (float | None): The value to use for filling any missing data points (NaNs).
        static_covariates (pd.Series | pd.DataFrame | None): External data that is constant
            over time for this series.
        metadata (dict | None): A dictionary for storing arbitrary metadata about the time series.
        make_copy (bool): Whether to create a copy of the underlying data. Defaults to `False` and ignored for CSV
            loading.
        path_to_csv (str): The filename or relative path to the CSV file.
    """

    path_to_csv: str
    make_copy: bool = Field(
        default=False, serialization_alias="copy", description="Whether to create a copy of the underlying data."
    )


class TimeSeriesFromCSVLoader(Template):
    """Template for loading time series data from a CSV file into a Time Series Packet.

    Usage example:

    agent:
        name: my_loader_agent
    templates:
    - template_name: InputTemplate
        class_name: InputTemplate
        attributes: {}
    - template_name: TimeSeriesFromCSVLoader
        class_name: TimeSeriesFromCSVLoader
        template_input: InputTemplate
        attributes:
        root_dir: "/root/.cache/sinapsis"
        assign_to: "content"
        loader_params:
            path_to_csv: "sales_data.csv"
            time_col: "Date"
            value_cols: "Revenue"
            freq: "D"
    """

    UIProperties = UIPropertiesMetadata(
        category="Darts",
        output_type=OutputTypes.TIMESERIES,
        tags=[Tags.DARTS, Tags.DATA, Tags.DATAFRAMES, Tags.PANDAS, Tags.TIME_SERIES],
    )

    class AttributesBaseModel(TemplateAttributes):
        """Defines the attributes required for the CSV Loader template.

        Attributes:
            root_dir (str | None): The base directory where the CSV file is located. If provided, it is prepended to
                `path_to_csv`.
            reuse_packet (bool): If `True`, attempts to assign the loaded series to the field specified by `assign_to`
                in the first existing `TimeSeriesPacket` found in the container. If `False` (default) or if no packets
                exist, a new `TimeSeriesPacket` is created with the loaded series assigned to its `content`.
            assign_to (Literal): The specific attribute of the `TimeSeriesPacket` where the loaded data will be
                stored (`content`, `past_covariates`, etc.).
            loader_params (FromCSVKwargs): Configuration parameters for reading the CSV file, including column
                mapping and frequency settings.
        """

        root_dir: str | None = None
        reuse_packet: bool = False
        assign_to: Literal["content", "past_covariates", "future_covariates", "predictions"]
        loader_params: FromCSVKwargs

    def __init__(self, attributes: TemplateAttributes) -> None:
        super().__init__(attributes)
        self.file_path = (
            os.path.join(self.attributes.root_dir, self.attributes.loader_params.path_to_csv)
            if self.attributes.root_dir
            else self.attributes.loader_params.path_to_csv
        )

    def _load_time_series(self) -> TimeSeries:
        """Reads the CSV file and returns the Darts TimeSeries object.

        Returns:
            TimeSeries: The TimeSeries object with the loaded series.
        """
        return TimeSeries.from_csv(
            filepath_or_buffer=self.file_path,
            **self.attributes.loader_params.model_dump(
                exclude_none=True, exclude={"make_copy", "path_to_csv"}, by_alias=True
            ),
        )

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the template by creating a new packet and loading the CSV data.

        Args:
            container (DataContainer): The input data container.

        Returns:
            DataContainer: The container updated with a new `TimeSeriesPacket` containing
                the loaded CSV data.
        """
        time_series = self._load_time_series()

        if self.attributes.reuse_packet and container.time_series:
            setattr(container.time_series[0], self.attributes.assign_to, time_series)
        else:
            container.time_series.append(TimeSeriesPacket(content=time_series))

        return container
