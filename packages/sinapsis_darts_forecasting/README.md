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

<h3> Templates Supported</h3>

The **Sinapsis Darts Forecasting** provides a powerful and flexible implementation for time series forecasting using the [Darts library](https://unit8co.github.io/darts/README.html).
<details>
<summary><strong><span style="font-size: 1.25em;">TimeSeriesDataframeLoader</span></strong></summary>

The following attributes apply to TimeSeriesDataframeLoader template:
- **`apply_to` (list, required)**: Specifies which attribute in `TimeSeriesPacket` should be converted from Pandas DataFrame to Darts TimeSeries (content, past_covariates, future_covariates, predictions).
- **`from_dataframe_kwargs` (dict[str, Any], optional)**: Additional arguments to pass to `TimeSeries.from_dataframe()`.
</details>
<details>
<summary><strong><span style="font-size: 1.25em;">Darts Transformers</span></strong></summary>

The following attributes apply to all the preprocessing templates from Darts Transformers:
- **`apply_to` (list, required)**: Specifies which attributes in `TimeSeriesPacket` should be transformed (content, past_covariates, future_covariates, predictions).
- **`method` (Literal, required)**: Specifies the transformation method to apply.
- **`transform_kwargs` (dict[str, Any], optional)**: Additional keyword arguments for the selected transformation method.
- **`params_key` (str, optional)**: If provided, transformation parameters are stored/retrieved in `TimeSeriesPacket.generic_data`.

Additional transformation-specific attributes can be dynamically assigned through the class initialization dictionary (`*_init` attributes). These attributes correspond directly to the arguments used in Darts Transformers.
</details>
<details>
<summary><strong><span style="font-size: 1.25em;">Darts Models</span></strong></summary>

The following attribute apply only to templates from Darts Models:
- **`forecast_horizon` (int, optional)**: Number of future time steps the model should predict. Defaults to `10`.
Additional transformation-specific attributes can be dynamically assigned through the class initialization dictionary (`*_init` attributes). These attributes correspond directly to the arguments used in Darts Models. Typically used for hyperparameters directly assigned to the corresponding model.
</details>

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Image Transforms.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***TimeSeriesDataframeLoader*** use ```sinapsis info --example-template-config TimeSeriesDataframeLoader``` to produce the following example config:

```yaml
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
    apply_to: 'content'
    from_dataframe_kwargs: {}
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

- template_name: TimeSeriesDataframeLoader
  class_name: TimeSeriesDataframeLoader
  template_input: InputTemplate
  attributes:
    apply_to: ["content", "past_covariates", "future_covariates"]
    from_dataframe_kwargs:
      time_col: "Date"
      fill_missing_dates: True
      freq: "D"

- template_name: MissingValuesFiller
  class_name: MissingValuesFillerWrapper
  template_input: TimeSeriesDataframeLoader
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
>Attributes specified under the `*_init` keys (e.g., `missingvaluesfiller_init`, `xgbmodel_init`) correspond directly to the Darts transformation or models parameters. Ensure that values are assigned correctly according to the official [Darts documentation](https://unit8co.github.io/darts/README.html), as they affect the behavior and performance of the model or the data.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

</details>

<h2 id="webapp">🌐 Webapp</h2>

The webapp provides an intuitive interface for data loading, preprocessing, and forecasting. The webapp supports CSV file uploads, visualization of historical data, and forecasting.

> [!NOTE]
> Kaggle offers a variety of datasets for forecasting. In [this-link](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd?select=BTC-Daily.csv) from Kaggle, you can find a Bitcoin historical dataset. You can download it to use it in the app. Past and future covariates datasets are optional for the analysis.

> [!IMPORTANT]
> Note that if you use another dataset, you need to change the attributes of the `TimeSeriesDataframeLoader`


> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-time-series-forecasting.git
cd sinapsis-time-series-forecasting
```
> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

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
3. The logs will display the URL to access the webapp, e.g.:

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
uv pip install sinapsis-time-series-forecasting[all] --extra-index-url https://pypi.sinapsis.tech
```

3. **Run the webapp**:
```bash
uv run  webapps/darts_time_series_gradio_app.py
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



