# -*- coding: utf-8 -*-
from typing import Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.data_containers.data_packet import DataContainer, TimeSeriesPacket
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

CONFIG = (
    AGENT_CONFIG_PATH
    or "packages/sinapsis_darts_forecasting/src/sinapsis_darts_forecasting/configs/time_series_lstm.yml"
)
agent = generic_agent_builder(CONFIG)


def get_plot(dataframe: pd.DataFrame) -> plt.Figure:
    """Method to create a plot figure using the data contained in a dataframe.

    Args:
        dataframe (pd.DataFrame): Dataframe containing the data to be plotted.

    Returns:
        plt.Figure: Figure with plotted data.
    """
    fig, ax = plt.subplots()
    dataframe.plot(ax=ax, subplots=True, sharex=True)
    fig.tight_layout()

    return fig


def time_series_forecasting(
    csv_target: gr.File,
    csv_past_covariates: gr.File | None,
    csv_future_covariates: gr.File | None,
    n_values: int,
) -> Tuple[plt.Figure, plt.Figure]:
    """Perform time series forecasting using tabular data loaded from a CSV file.

    Args:
        csv_target (gr.File): Path to CSV file containing time series of the target.
        csv_past_covariates(gr.File | None): optional path to the csv file with the past covariates
        csv_future_covariates(gr.File | None): optional path to the csv with the future covariates
        n_values (int): last N numbers of datapoints from historical data to plot

    Returns:
        Tuple[plt.Figure, plt.Figure]: Figures containing plots of target and forecasted time series data.
    """
    agent.reset_state("TimeSeries")
    target_data = pd.read_csv(csv_target, encoding="utf-8")
    past_covariates_data = pd.read_csv(csv_past_covariates, encoding="utf-8") if csv_past_covariates else None
    future_covariates_data = pd.read_csv(csv_future_covariates, encoding="utf-8") if csv_future_covariates else None

    time_series_packet = TimeSeriesPacket(
        content=target_data, past_covariates=past_covariates_data, future_covariates=future_covariates_data
    )

    input_container = DataContainer(time_series=[time_series_packet])
    output_container = agent(container=input_container)

    if output_container.time_series:
        time_series_target = output_container.time_series[0].content.to_dataframe()
        time_series_forecast = output_container.time_series[0].predictions.to_dataframe()

        time_series_plot = get_plot(time_series_target.tail(n_values))
        time_series_forecast_plot = get_plot(time_series_forecast)

        return time_series_plot, time_series_forecast_plot
    return None, None


def demo(description_str: str) -> gr.Blocks:
    with gr.Blocks(css=css_header()) as app_demo:
        add_logo_and_title("Sinapsis Time Series Forecasting")
        gr.Interface(
            fn=time_series_forecasting,
            inputs=[
                gr.File(label="Load target time series (CSV)", file_types=[".csv"], height="10em"),
                gr.File(label="Load past covariates (CSV, optional)", file_types=[".csv"], height="10em"),
                gr.File(label="Load future covariates (CSV, optional)", file_types=[".csv"], height="10em"),
                gr.Number(value=100, label="Plot last N values of historical data."),
            ],
            outputs=[
                gr.Plot(label="Historical data"),
                gr.Plot(label="Forecasted data"),
            ],
            live=False,
            flagging_mode="never",
            submit_btn="Predict",
            description=description_str,
        )
    return app_demo


if __name__ == "__main__":
    description_str = """Please load CSV files containing time series data for forecasting.
                        \nRequirements:
                        \n* The target time series file is required, while past and future covariates are optional.
                        \n* The dataset must contain a time column. Ensure the correct key is set in the configuration
                        file.
                        \n* The time column must have consecutive timestamps according to a defined period (e.g.,
                        hourly, daily, monthly).
                        \n* The target values must be numeric.
                        \n* If provided, past covariates must have the **same** length as the target time series.
                        \n* Future covariates must have a length equal to the target time series length **plus** the
                        forecast horizon.
                        \n* The date values in the target series and all covariates must **match exactly**.

                        Example: Target Data for Univariate Models

                        Date       Numeric_values_1
                        2000-01-01       10
                        2000-01-02       15
                        2000-01-03        9


                        Example: Target Data for Multivariate Models

                        Date       Numeric_values_1  Numeric_values_2  ......  Numeric_values_N
                        2000-01-01       10               25          ......        17
                        2000-01-02       15               11          ......        21
                        2000-01-03        9               32          ......        35
                        """

    ts_demo = demo(description_str)

    ts_demo.launch(share=GRADIO_SHARE_APP)
