# -*- coding: utf-8 -*-

import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import plotly.graph_objects as go
from darts import TimeSeries
from pydantic import BaseModel, Field
from sinapsis_core.data_containers.data_packet import DataContainer, FilePacket, ImagePacket, TimeSeriesPacket
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributes, UIPropertiesMetadata
from sinapsis_core.template_base.template import Template
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

from sinapsis_darts_forecasting.helpers.tags import Tags


class TimeSeriesPlotParams(BaseModel):
    target_series_label: str = "Historical"
    predictions_label: str = "Predictions"
    title: str = "Time Series Forecast Results"
    x_axis_title: str = "Time"
    low_quantile: float = 0.05
    high_quantile: float = 0.95


class TimeSeriesVisualizationAttributes(TemplateAttributes):
    """Attributes for the TimeSeriesVisualization template.

    Attributes:
        html_figure_save_path (str): The filename for the output HTML file containing the forecast plot. This can be
            a relative path or an absolute path. If the file already exists, it will be overwritten.
        root_dir (str): The root directory where the HTML figure will be saved. This can be set to a specific directory
            or left as the default cache directory defined by SINAPSIS_CACHE_DIR.
        plot_params (TimeSeriesPlotParams): A nested model containing parameters for customizing the appearance of the
            forecast plots, such as labels, titles, and quantile ranges for prediction intervals.
    """

    html_figure_save_path: str = "forecast_plot.html"
    root_dir: str = SINAPSIS_CACHE_DIR
    plot_params: TimeSeriesPlotParams = Field(default_factory=TimeSeriesPlotParams)

    @property
    def full_save_path(self) -> str:
        return os.path.join(self.root_dir, self.html_figure_save_path)


class TimeSeriesVisualization(Template):
    UIProperties = UIPropertiesMetadata(
        category="Darts",
        output_type=OutputTypes.TIMESERIES,
        tags=[Tags.DARTS, Tags.FORECASTING, Tags.TIME_SERIES, Tags.VISUALIZATION, Tags.PLOTLY, Tags.MATPLOTLIB],
    )

    AttributesBaseModel = TimeSeriesVisualizationAttributes

    def _are_series_continuous(self, target_col: TimeSeries, pred_col: TimeSeries) -> bool:
        """Return True when prediction starts exactly after target ends.

        Uses the target series frequency when available. Falls back to an equality
        check for contiguous boundaries if frequency metadata is missing.

        Args:
            target_col (TimeSeries): The historical time series data.
            pred_col (TimeSeries): The predicted time series data.
        Returns:
            bool: True if the last timestamp of the target series is immediately followed by the first timestamp of
                the prediction series, indicating that they are continuous. False otherwise.
        """
        target_index = target_col.time_index
        pred_index = pred_col.time_index

        if len(target_index) == 0 or len(pred_index) == 0:
            return False

        target_last_value = target_index[-1]
        pred_first_value = pred_index[0]

        freq = getattr(target_col, "freq", None)
        if freq is None:
            return target_last_value == pred_first_value

        return target_last_value + freq == pred_first_value

    def _add_quantile_bands_from_array(self, fig: go.Figure, time_index: pd.Index, pred_col_values: np.ndarray) -> None:
        """Add probabilistic quantile bands to a Plotly figure from raw (time, samples) array.

        Args:
            fig (go.Figure): The Plotly figure to update.
            time_index: The time index for x-axis.
            pred_col_values (np.ndarray): The predicted values with shape (time, samples).
        """
        try:
            if pred_col_values.ndim != 2:
                return

            n_samples = pred_col_values.shape[1]
            if n_samples < 2:
                return

            low_idx = max(0, int(np.floor(self.attributes.plot_params.low_quantile * (n_samples - 1))))
            high_idx = min(n_samples - 1, int(np.ceil(self.attributes.plot_params.high_quantile * (n_samples - 1))))

            if low_idx == high_idx:
                return

            low_quantile = pred_col_values[:, low_idx]
            high_quantile = pred_col_values[:, high_idx]

            fig.add_trace(
                go.Scatter(
                    x=time_index.tolist() + time_index.tolist()[::-1],
                    y=high_quantile.tolist() + low_quantile.tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(0, 176, 246, 0.15)",
                    line={"color": "rgba(255, 255, 255, 0)"},
                    showlegend=True,
                    name=(
                        f"Confidence Band ("
                        f"{int(self.attributes.plot_params.low_quantile * 100)}-"
                        f"{int(self.attributes.plot_params.high_quantile * 100)}%)"
                    ),
                    hoverinfo="skip",
                )
            )
        except (ValueError, TypeError, IndexError) as e:
            self.logger.error("Failed to add quantile bands: %s", str(e))

    def _build_matplotlib_column_figure(
        self, target_series: TimeSeries, predictions: TimeSeries, column: str
    ) -> plt.Figure:
        """Builds a Matplotlib figure for a single time series column.

        Args:
            target_series (TimeSeries): Historical time series data.
            predictions (TimeSeries): Predicted values.
            column (str): The column name to plot.

        Returns:
            plt.Figure: Matplotlib figure for the specified column.
        """
        target_col = target_series.univariate_component(column)
        pred_col = predictions.univariate_component(column)

        fig, ax = plt.subplots()
        target_col.plot(label=self.attributes.plot_params.target_series_label, ax=ax)
        pred_col.plot(
            label=self.attributes.plot_params.predictions_label,
            low_quantile=self.attributes.plot_params.low_quantile,
            high_quantile=self.attributes.plot_params.high_quantile,
            ax=ax,
        )
        ax.set_title(f"{self.attributes.plot_params.title} for {column} data.")
        ax.set_xlabel(self.attributes.plot_params.x_axis_title)
        ax.set_ylabel(column)
        ax.legend()
        return fig

    def plot_forecast(self, target_series: TimeSeries, predictions: TimeSeries) -> list[plt.Figure]:
        """Creates one matplotlib figure per column, plotting historical data and predictions.

        Args:
            target_series (TimeSeries): Historical time series data (target variables).
            predictions (TimeSeries): The predicted values for the future time steps.

        Returns:
            list[plt.Figure]: One figure per time series column.
        """
        columns = target_series.columns.tolist()
        return [self._build_matplotlib_column_figure(target_series, predictions, column) for column in columns]

    def _build_column_figure(self, target_series: TimeSeries, predictions: TimeSeries, column: str) -> go.Figure:
        """Builds a Plotly figure for a single time series column.

        Args:
            target_series (TimeSeries): Historical time series data.
            predictions (TimeSeries): Predicted values (may be probabilistic with quantiles).
            column (str): The column name to plot.

        Returns:
            go.Figure: Plotly figure for the specified column.
        """
        target_col = target_series.univariate_component(column)

        # all_values() preserves the full (time, components, samples) shape
        # for probabilistic series. values() only returns (time, components).
        pred_values_full = predictions.all_values()
        col_idx = predictions.columns.tolist().index(column) if column in predictions.columns else 0

        if pred_values_full.ndim == 3:
            pred_col_values = pred_values_full[:, col_idx, :]  # (time, samples)
            point_forecast_values = np.median(pred_col_values, axis=1)
        else:
            pred_col_values = pred_values_full[:, col_idx]  # (time,)
            point_forecast_values = pred_col_values

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=target_col.time_index,
                y=target_col.values().flatten(),
                mode="lines",
                name=self.attributes.plot_params.target_series_label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.time_index,
                y=point_forecast_values,
                mode="lines",
                name=self.attributes.plot_params.predictions_label,
            )
        )

        target_values = target_col.values().flatten()

        target_last_value = target_col.time_index[-1]
        pred_first_value = predictions.time_index[0]

        pred_col_univariate = predictions.univariate_component(column)
        if (
            len(target_values) > 0
            and len(point_forecast_values) > 0
            and self._are_series_continuous(target_col, pred_col_univariate)
        ):
            fig.add_trace(
                go.Scatter(
                    x=[target_last_value, pred_first_value],
                    y=[target_values[-1], point_forecast_values[0]],
                    mode="lines",
                    line={"dash": "solid", "color": "rgba(100, 100, 100, 0.5)"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if pred_col_values.ndim == 2:
            self._add_quantile_bands_from_array(fig, predictions.time_index, pred_col_values)

        figure_title = f"{self.attributes.plot_params.title} for {column} data."
        fig.update_layout(
            title=figure_title,
            xaxis_title=self.attributes.plot_params.x_axis_title,
            yaxis_title=column,
        )
        return fig

    def plot_forecast_to_html(
        self, target_series: TimeSeries, predictions: TimeSeries, packet_id: str
    ) -> list[FilePacket]:
        """Builds one interactive Plotly HTML file per time series column.

        For multivariate series each column gets its own figure saved as
        ``<output_path_stem>_<packet_id>_<column>.html``. For univariate series
        a single file is written as ``<output_path_stem>_<packet_id>.html``.

        Args:
            target_series (TimeSeries): Historical time series data (target variables).
            predictions (TimeSeries): The predicted values for the future time steps.
            packet_id (str): Identifier for the time series packet, used to
                generate unique output filenames.

        Returns:
            list[FilePacket]: One file packet per saved HTML file.
        """
        columns = target_series.columns.tolist()
        base, ext = os.path.splitext(self.attributes.full_save_path)

        os.makedirs(os.path.dirname(self.attributes.full_save_path), exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_packets = []
        for column in columns:
            fig = self._build_column_figure(target_series, predictions, column)
            output_file = f"{base}_{packet_id}_{column}_{current_time}{ext}"
            fig.write_html(output_file)
            file_packets.append(FilePacket(content=output_file, source=self.instance_name))
        return file_packets

    def figure_to_image(self, fig: plt.Figure) -> np.ndarray:
        """Converts a Matplotlib figure to an RGB numpy array.

        Args:
            fig (plt.Figure): The Matplotlib figure to convert.

        Returns:
            np.ndarray: The image array (H, W, 3) in RGB format.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf)
        return np.asarray(image.convert("RGB"))

    def _create_image_packet(self, fig: plt.Figure, packet_id: str, column: str) -> ImagePacket:
        """Creates an ImagePacket from a Matplotlib figure.

        Args:
            fig (plt.Figure): The Matplotlib figure to convert.
            packet_id (str): Identifier for the time series packet.
            column (str): The column name this figure represents.

        Returns:
            ImagePacket: Image packet containing the rendered figure.
        """
        np_img = self.figure_to_image(fig)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        source = f"{self.instance_name}_{packet_id}_{column}_{current_time}"
        return ImagePacket(content=np_img, source=source)

    def _visualize_packet(
        self, target_series: TimeSeries, predictions: TimeSeries, packet_id: str
    ) -> tuple[list[ImagePacket], list[FilePacket]]:
        """Generates all visualizations for a single time series packet.

        Args:
            target_series (TimeSeries): Historical time series data.
            predictions (TimeSeries): Predicted values.
            packet_id (str): Identifier for the time series packet.

        Returns:
            list[ImagePacket]: One image packet per time series column.
            list[FilePacket]: One file packet per time series column.
        """
        columns = target_series.columns.tolist()
        figures = self.plot_forecast(target_series, predictions)
        image_packets = []
        for fig, column in zip(figures, columns):
            image_packets.append(self._create_image_packet(fig, packet_id, column))
            plt.close(fig)
        file_packets = self.plot_forecast_to_html(target_series, predictions, packet_id)
        return image_packets, file_packets

    def _validate_time_series_packet(self, packet: TimeSeriesPacket) -> bool:
        """Validates that the time series packet contains the necessary data for visualization.

        Args:
            packet (TimeSeriesPacket): The time series packet to validate.
        Returns:
            bool: True if the packet is valid for visualization, False otherwise.
        """
        if packet.predictions is None:
            self.logger.warning(
                "Time series packet with id %s does not contain predictions, skipping visualization.",
                packet.id,
            )
            return False
        if not isinstance(packet.content, TimeSeries):
            self.logger.warning(
                "Content of time series packet with id %s is not a TimeSeries instance, skipping visualization.",
                packet.id,
            )
            return False
        if not isinstance(packet.predictions, TimeSeries):
            self.logger.warning(
                "Predictions of time series packet with id %s is not a TimeSeries instance, skipping visualization.",
                packet.id,
            )
            return False
        return True

    def execute(self, container: DataContainer) -> DataContainer:
        """Generates forecast visualizations for each time series packet.

        Args:
            container (DataContainer): Input data container with time series packets.

        Returns:
            DataContainer: Updated container with forecast images and files appended.
        """
        for time_series_packet in container.time_series:
            if not self._validate_time_series_packet(time_series_packet):
                continue

            image_packets, file_packets = self._visualize_packet(
                time_series_packet.content, time_series_packet.predictions, time_series_packet.id
            )
            container.images.extend(image_packets)
            container.files.extend(file_packets)

        return container
