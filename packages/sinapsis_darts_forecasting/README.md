<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis Darts Forecasting
<br>
</h1>

<h4 align="center">Module for handling time series data and forecasting using Darts.</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage Example</a> •
<a href="#webapp"> 🌐 Webapp</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

**Sinapsis Darts Forecasting** provides a powerful and flexible implementation for time series forecasting using the [Darts library](https://unit8co.github.io/darts/README.html).


<h2 id="installation"> 🐍  Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-darts-forecasting --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-darts-forecasting --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<h3>Templates Supported</h3>

The **Sinapsis Darts Forecasting** provides a powerful and flexible implementation for time series forecasting using the [Darts library](https://unit8co.github.io/darts/README.html).
<details>
<summary><strong><span style="font-size: 1.25em;">TimeSeriesFromCSVLoader</span></strong></summary>

Loads one or more columns from a CSV file directly into a `TimeSeriesPacket` using `TimeSeries.from_csv()`. Supports assigning the result to any packet attribute (`content`, `past_covariates`, `future_covariates`, `predictions`) and can either create a new packet or update an existing one in the container.

The following attributes apply to `TimeSeriesFromCSVLoader`:
- **`root_dir` (str | None, optional)**: Base directory prepended to `path_to_csv`. If omitted, `path_to_csv` is used as-is.
- **`assign_to` (Literal, required)**: The `TimeSeriesPacket` attribute where the loaded series is stored. One of `"content"`, `"past_covariates"`, `"future_covariates"`, `"predictions"`.
- **`reuse_packet` (bool, optional)**: If `True`, assigns the loaded series to the first existing `TimeSeriesPacket` in the container instead of creating a new one. Useful when loading multiple columns (e.g., target + covariates) into the same packet. Defaults to `False`.
- **`loader_params` (FromCSVKwargs, required)**: CSV-specific loading parameters:
  - `path_to_csv` (str): Filename or relative path to the CSV file.
  - `time_col` (str | None): Column name to use as the time index.
  - `value_cols` (str | list[str] | None): Column(s) to use as series values.
  - `freq` (str | None): Time series frequency (e.g., `"D"` for daily, `"H"` for hourly).
  - `fill_missing_dates` (bool): If `True`, inserts rows for missing timestamps. Defaults to `False`.
  - `fillna_value` (float | None): Fill value for any resulting NaNs after date insertion.
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">TimeSeriesFromDataframeLoader</span></strong></summary>

Converts Pandas `DataFrame` objects already present in a `TimeSeriesPacket` into Darts `TimeSeries` instances using `TimeSeries.from_dataframe()`. Useful when upstream templates deliver DataFrames that need to be converted before forecasting.

The following attributes apply to `TimeSeriesFromDataframeLoader`:
- **`apply_to` (list[Literal], required)**: List of `TimeSeriesPacket` attributes to convert. Valid values: `"content"`, `"past_covariates"`, `"future_covariates"`, `"predictions"`.
- **`from_pandas_kwargs` (FromDataFrameKwargs, optional)**: Keyword arguments forwarded to `TimeSeries.from_dataframe()`:
  - `time_col` (str | None): Column to use as the time index.
  - `value_cols` (str | list[str] | None): Column(s) to use as the series values.
  - `freq` (str | None): Time series frequency (e.g., `"D"`, `"H"`).
  - `fill_missing_dates` (bool): Insert rows for missing timestamps. Defaults to `False`.
  - `fillna_value` (float | None): Value used to fill NaNs introduced by date filling.
  - `make_copy` (bool): Whether to copy the underlying NumPy array. Defaults to `True`.
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">TimeSeriesFromSeriesLoader</span></strong></summary>

Converts Pandas `Series` objects already present in a `TimeSeriesPacket` into Darts `TimeSeries` instances using `TimeSeries.from_series()`. The series index is automatically converted to timestamps via `to_timestamp()` before conversion.

The following attributes apply to `TimeSeriesFromSeriesLoader`:
- **`apply_to` (list[Literal], required)**: List of `TimeSeriesPacket` attributes to convert. Valid values: `"content"`, `"past_covariates"`, `"future_covariates"`, `"predictions"`.
- **`from_pandas_kwargs` (FromPandasBaseKwargs, optional)**: Keyword arguments forwarded to `TimeSeries.from_series()`:
  - `freq` (str | None): Time series frequency (e.g., `"D"`, `"H"`).
  - `fill_missing_dates` (bool): Insert rows for missing timestamps. Defaults to `False`.
  - `fillna_value` (float | None): Value used to fill NaNs introduced by date filling.
  - `make_copy` (bool): Whether to copy the underlying NumPy array. Defaults to `True`.
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">Darts Transformers</span></strong></summary>

Dynamic wrapper that exposes every transformer from `darts.dataprocessing.transformers` as an individual Sinapsis template (e.g., `ScalerWrapper`, `MissingValuesFillerWrapper`, `BoxCoxWrapper`). Transformations are applied to the selected `TimeSeriesPacket` attributes in-place.

For stateful transformers (e.g., `Scaler`), fitted parameters can be persisted in `TimeSeriesPacket.generic_data` and recovered automatically for inverse transforms — enabling clean scaler/unscaler pairs in a pipeline without manual state management.

The following attributes apply to all Darts Transformer templates:
- **`apply_to` (list[Literal], required)**: `TimeSeriesPacket` attributes to transform. Valid values: `"content"`, `"past_covariates"`, `"future_covariates"`, `"predictions"`.
- **`method` (Literal, required)**: Transformation method to call. One of `"fit"`, `"transform"`, `"fit_transform"`, `"inverse_transform"`.
- **`transform_kwargs` (dict[str, Any], optional)**: Extra keyword arguments forwarded to the chosen method (e.g., `{"method": "linear"}` for `MissingValuesFiller`). Defaults to `{}`.
- **`params_key` (str | None, optional)**: Key under which fitted parameters are stored in `TimeSeriesPacket.generic_data`. Required when pairing a `fit_transform` step with a later `inverse_transform`. Defaults to `None` (no storage).

Additional transformer-specific init arguments can be set via the `*_init` key (e.g., `missingvaluesfiller_init`, `scaler_init`). These are forwarded directly to the transformer constructor.

> [!NOTE]
> For a full list of available transformers, see the [Darts transformers documentation](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.html).
</details>
<details>
<summary><strong><span style="font-size: 1.25em;">Darts Models</span></strong></summary>

The following attribute apply only to templates from Darts Models:
- **`forecast_horizon` (int, optional)**: Number of future time steps the model should predict. Defaults to `10`.
- **`validation_mode` (bool, optional)**: If `True`, the model is trained on all but the last `forecast_horizon` steps of the target series and predictions are generated for that held-out window, enabling metric evaluation. Defaults to `False`.

Additional model-specific attributes can be dynamically assigned through the class initialization dictionary (`*_init` attributes). These correspond directly to the constructor arguments of the chosen Darts model and are typically used for hyperparameter configuration.

> [!NOTE]
> Not all Darts models support both `past_covariates` and `future_covariates`. Unsupported covariates are automatically ignored at runtime based on each model's `supports_past_covariates` / `supports_future_covariates` properties.
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">TimeSeriesVisualization</span></strong></summary>

Generates per-packet visualizations of the historical target series alongside the model's predictions. For each `TimeSeriesPacket` that contains predictions, the template produces:

- One **Matplotlib PNG image** per time series column, stored as `ImagePacket` objects in `container.images`.
- One **interactive Plotly HTML file** per time series column, stored as `FilePacket` objects in `container.files`.

HTML files are named `<html_figure_save_path>_<packet_id>_<column>.html` so that multiple packets and multivariate series never overwrite each other.

The following attributes apply to `TimeSeriesVisualization`:
- **`html_figure_save_path` (str, optional)**: Base filename for the output HTML files. Defaults to `"forecast_plot.html"`.
- **`root_dir` (str, optional)**: Root directory where HTML files are saved. Defaults to `SINAPSIS_CACHE_DIR`. Parent directories are created automatically if they do not exist.
- **`plot_params` (TimeSeriesPlotParams, optional)**: Nested model to customise the appearance of both Matplotlib and Plotly figures. Available sub-fields:
  - `target_series_label` (str): Legend label for the historical series. Defaults to `"Historical"`.
  - `predictions_label` (str): Legend label for the predicted series. Defaults to `"Predictions"`.
  - `title` (str): Figure title. Defaults to `"Time Series Forecast Results"`.
  - `x_axis_title` (str): X-axis label. Defaults to `"Time"`.
  - `low_quantile` (float): Lower quantile bound for probabilistic prediction intervals (Matplotlib only). Defaults to `0.05`.
  - `high_quantile` (float): Upper quantile bound for probabilistic prediction intervals (Matplotlib only). Defaults to `0.95`.
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">TimeSeriesMetrics</span></strong></summary>

Dynamic wrapper that exposes every metric from `darts.metrics` as an individual Sinapsis template (e.g., `maeTimeSeriesWrapper`, `mseTimeSeriesWrapper`, `rmseTimeSeriesWrapper`). The selected metric is applied to the predictions and ground truth of each `TimeSeriesPacket`, and the result is stored in `TimeSeriesPacket.generic_data` keyed by the metric function name.

> [!IMPORTANT]
> `validation_mode: True` must be set on the forecasting model template so that predictions cover the held-out tail of the target series and can be correctly aligned with the ground truth for metric computation.

The following attributes apply to all metric templates. They are specified as a nested dictionary under the metric function name (e.g., `mae: {}`):
- **`intersect` (bool, optional)**: When `True`, only the overlapping time interval between ground truth and predictions is used. Defaults to `True`.
- **`q` (float | list[float] | None, optional)**: Quantile(s) to evaluate for probabilistic or quantile predictions. Defaults to `None` (median for stochastic predictions).
- **`n_jobs` (int, optional)**: Number of parallel jobs when evaluating over a sequence of series. Defaults to `1`.
- **`verbose` (bool, optional)**: Whether to print progress. Defaults to `False`.

The following parameters are intentionally excluded and cannot be overridden via the template config (they are managed internally):
`actual_series`, `pred_series`, `component_reduction`, `series_reduction`, `time_reduction`.
</details>

> [!TIP]
> Use CLI command `sinapsis info --all-template-names` to show a list with all the available Template names installed with Sinapsis Darts Forecasting.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***TimeSeriesFromDataframeLoader*** use ```sinapsis info --example-template-config TimeSeriesFromDataframeLoader``` to produce the following example config:

```yaml
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
    apply_to: 'content'
    from_pandas_kwargs: {}
```

<h2 id="example"> 📚 Usage Example </h2>

Below is an example configuration for **Sinapsis Darts Forecasting** using an XGBoost model. This setup extracts pandas DataFrames from the time series packet attributes and converts them into `TimeSeries` objects, using the `Date` column as the time index. Missing dates are filled with a daily frequency, and any missing values are interpolated using a linear method. The model is then trained and used to generate predictions with a forecast horizon of 100 days, with several configurable hyperparameters.

<details>
<summary><strong><span style="font-size: 1.25em;">Example config</span></strong></summary>


```yaml
agent:
  name: XGBLSTMForecastingAgent
  description: ''

templates:

- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: TimeSeriesFromDataframeLoader
  class_name: TimeSeriesFromDataframeLoader
  template_input: InputTemplate
  attributes:
    apply_to: ["content", "past_covariates", "future_covariates"]
    from_pandas_kwargs:
      time_col: "Date"
      fill_missing_dates: True
      freq: "D"

- template_name: MissingValuesFiller
  class_name: MissingValuesFillerWrapper
  template_input: TimeSeriesFromDataframeLoader
  attributes:
    method: "transform"
    missingvaluesfiller_init: {}
    apply_to: ["content", "past_covariates", "future_covariates"]
    transform_kwargs:
      method: "linear"

- template_name: TimeSeries
  class_name: XGBModelWrapper
  template_input: MissingValuesFiller
  attributes:
    forecast_horizon: 100
    xgbmodel_init:
      lags: 30
      lags_past_covariates: 30
      output_chunk_length: 100
      random_state: 42
      n_estimators: 200
      learning_rate: 0.1
      max_depth: 6
```
</details>

This configuration defines an **agent** and a sequence of **templates** to handle the data and perform predictions.

> [!IMPORTANT]
> Attributes specified under the `*_init` keys (e.g., `missingvaluesfiller_init`, `xgbmodel_init`) correspond directly to the Darts transformation or models parameters. Ensure that values are assigned correctly according to the official [Darts documentation](https://unit8co.github.io/darts/README.html), as they affect the behavior and performance of the model or the data.

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

<h2 id="webapp">🌐 Webapp</h2>

The webapp provides an intuitive interface for data loading, preprocessing, and forecasting. The webapp supports CSV file uploads, visualization of historical data, and forecasting.

> [!NOTE]
> Kaggle offers a variety of datasets for forecasting. In [this-link](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd?select=BTC-Daily.csv) from Kaggle, you can find a Bitcoin historical dataset. You can download it to use it in the app. Past and future covariates datasets are optional for the analysis.

> [!IMPORTANT]
> Note that if you use another dataset, you need to change the attributes of the `TimeSeriesFromDataframeLoader`


> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-time-series-forecasting.git
cd sinapsis-time-series-forecasting
```
> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-time-series-forecasting image**:
```bash
docker compose -f docker/compose.yaml build
```

2. **Start the app container**:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-darts-forecasting-gradio -d
```
3. **Check the status**:
```bash
docker logs -f sinapsis-darts-forecasting-gradio
```
4. The logs will display the URL to access the webapp, e.g.:

NOTE: The url can be different, check the output of logs
```bash
Running on local URL:  http://127.0.0.1:7860
```
4. To stop the app:
```bash
docker compose -f docker/compose_apps.yaml down
```

</details>


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">💻 UV</span></strong></summary>

To run the webapp using the <code>uv</code> package manager, please:

1. **Create the virtual environment and sync the dependencies**:
```bash
uv sync --frozen
```
2. **Install the wheel**:
```bash
uv pip install sinapsis-time-series[all] --extra-index-url https://pypi.sinapsis.tech
```

3. **Run the webapp**:
```bash
uv run webapps/darts_time_series_gradio_app.py
```
4. **The terminal will display the URL to access the webapp, e.g.**:

NOTE: The url can be different, check the output of the terminal
```bash
Running on local URL:  http://127.0.0.1:7860
```

</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.



