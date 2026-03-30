<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis Time Series Forecasting
<br>
</h1>

<h4 align="center"> Monorepo with packages to perform time series forecasting, preprocessing, and data loading.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#packages">📦 Packages</a> •
<a href="#usage">📚 Usage example</a> •
<a href="#webapp"> 🌐 Webapp </a>  •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

<h2 id="installation">🐍 Installation</h2>

This monorepo currently consists of the following packages to handle time-series data:

* <code>sinapsis-darts-forecasting</code>
* <code>sinapsis-sktime</code>
* <code>sinapsis-timesfm</code>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-darts-forecasting --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-darts-forecasting --extra-index-url https://pypi.sinapsis.tech
```


> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-darts-forecasting[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-darts-forecasting[all] --extra-index-url https://pypi.sinapsis.tech
```


> [!TIP]
> You can also install all the packages within this project:
>
```bash
  uv pip install sinapsis-time-series-forecasting[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="packages">📦 Packages</h2>
<details id='packages'><summary><strong><span style="font-size: 1.0em;"> Packages summary</span></strong></summary>


- **Sinapsis Darts Forecasting** (`sinapsis-darts-forecasting`)
  - **CSV Loader**\
  _Load time series data from CSV files into Darts TimeSeries objects._
  - **Dataframe Loader**\
  _Convert a pandas DataFrame into a Darts TimeSeries object._
  - **Series Loader**\
  _Convert a pandas Series into a Darts TimeSeries object._
  - **Darts Transformers**\
  _Apply data transformations (scaling, missing value filling, etc.) using Darts transformer classes._
  - **Darts Models**\
  _Fit and predict using Darts baseline, statistical, machine learning, and deep learning models._
  - **Time Series Visualization**\
  _Generate Matplotlib images and interactive Plotly HTML charts comparing historical and forecasted values._
  - **Time Series Metrics**\
  _Compute Darts metrics (MAE, RMSE, MAPE, etc.) between predicted and ground-truth time series._

- **Sinapsis SKTime** (`sinapsis-sktime`)
  - **SKTime Forecasters**\
  _Train and generate predictions using sktime forecasting models (ARIMA, AutoARIMA, ExponentialSmoothing, Theta, NaiveForecaster, and more)._
  - **SKTime Classifiers**\
  _Train and classify time series using sktime classification models (distance-based, dictionary-based, feature-based, deep learning, and dummy classifiers)._

- **Sinapsis TimesFM** (`sinapsis-timesfm`)
  - **TimesFM**\
  _Perform time series forecasting using Google's TimesFM foundation model._
</details>

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Time Series Forecasting.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***XGBModelWrapper*** use ```sinapsis info --example-template-config XGBModelWrapper``` to produce the following example config:


```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: XGBModelWrapper
  class_name: XGBModelWrapper
  template_input: InputTemplate
  attributes:
    forecast_horizon: 10
    xgbmodel_init:
      lags: null
      lags_past_covariates: null
      lags_future_covariates: null
      output_chunk_length: 1
      output_chunk_shift: 0
      add_encoders: null
      likelihood: null
      quantiles: null
      random_state: null
      multi_models: true
      use_static_covariates: true
```


<h2 id="usage">📚 Usage example</h2>
Below is an example configuration for **Sinapsis Darts Forecasting** using an XGBoost model. This setup extracts pandas DataFrames from the time series packet attributes and converts them into `TimeSeries` objects, using the `Date` column as the time index. Missing dates are filled with a daily frequency, and any missing values are interpolated using a linear method. The model is then trained and used to generate predictions with a forecast horizon of 100 days, with several configurable hyperparameters.
<details id='usage'><summary><strong><span style="font-size: 1.0em;"> Example agent config</span></strong></summary>



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

To run, simply use:

```bash
sinapsis run name_of_the_config.yml
```
</details>


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
uv run webapps/darts_time_series_gradio_app.py
```
4. **The terminal will display the URL to access the webapp, e.g.**:

NOTE: The url can be different, check the output of the terminal
```bash
Running on local URL:  http://127.0.0.1:7860
```


<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
