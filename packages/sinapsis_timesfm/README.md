<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis TimesFM
<br>
</h1>

<h4 align="center">Module for time series forecasting using Google's TimesFM foundation model.</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage Example</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

**Sinapsis TimesFM** provides zero-shot time series forecasting using [Google's TimesFM](https://github.com/google-research/timesfm) foundation model. It supports both pandas DataFrame and Darts TimeSeries outputs, including point and quantile predictions.


<h2 id="installation"> 🐍  Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-timesfm --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-timesfm --extra-index-url https://pypi.sinapsis.tech
```

> [!NOTE]
> TimesFM supports multiple backends. Install the one you need:
> ```bash
> uv pip install sinapsis-timesfm[torch] --extra-index-url https://pypi.sinapsis.tech   # PyTorch
> uv pip install sinapsis-timesfm[flax] --extra-index-url https://pypi.sinapsis.tech    # Flax/JAX
> uv pip install sinapsis-timesfm[all] --extra-index-url https://pypi.sinapsis.tech     # All backends
> ```

<h2 id="features">🚀 Features</h2>

<h3>Templates Supported</h3>

<details>
<summary><strong><span style="font-size: 1.25em;">TimesFM</span></strong></summary>

Zero-shot time series forecasting template powered by Google's TimesFM 2.5 foundation model. Accepts pandas DataFrames or Darts TimeSeries as input, automatically sorts by time, and produces point forecasts with optional quantile predictions.

The following attributes apply to the `TimesFM` template:
- **`model_name` (str, optional)**: Hugging Face model identifier for TimesFM. Defaults to `"google/timesfm-2.5-200m-pytorch"`.
- **`forecasting_config` (ForecastConfigBM, optional)**: Forecasting behavior configuration passed to `TimesFM.compile()`. Defaults to `ForecastConfigBM()`.
- **`forecast_horizon` (int, optional)**: Number of future steps to predict for each input series. Defaults to `12`.
- **`validation_mode` (bool, optional)**: If `True`, reserves the last `forecast_horizon` points for validation. Defaults to `False`.
- **`time_series_output_format` (Literal, optional)**: Output format for predictions. One of `"pandas_dataframe"` or `"darts_series"`. Defaults to `"pandas_dataframe"`.
- **`device` (str, optional)**: Torch device used for inference, for example `"cpu"` or `"cuda:0"`. Defaults to `"cpu"`.
- **`time_format` (str | None, optional)**: Explicit datetime format for parsing and formatting. Defaults to `None` (auto-inferred).
- **`time_column_name` (str, optional)**: Name of the time column in the input dataframe. Defaults to `"Date"`.
</details>

> [!TIP]
> Use CLI command `sinapsis info --all-template-names` to show a list with all the available Template names installed with Sinapsis TimesFM.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***TimesFM*** use ```sinapsis info --example-template-config TimesFM``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: TimesFM
  class_name: TimesFM
  template_input: InputTemplate
  attributes:
    model_name: google/timesfm-2.5-200m-pytorch
    forecasting_config:
      max_context: 0
      max_horizon: 0
      normalize_inputs: false
      window_size: 0
      per_core_batch_size: 1
      use_continuous_quantile_head: false
      force_flip_invariance: true
      infer_is_positive: true
      fix_quantile_crossing: false
      return_backcast: false
    forecast_horizon: 12
    validation_mode: false
    time_series_output_format: pandas_dataframe
    device: cpu
    time_format: null
    time_column_name: Date
```

<h2 id="example"> 📚 Usage Example </h2>

Below is an example configuration for **Sinapsis TimesFM** that loads a CSV file into a time series packet and runs zero-shot forecasting with a 30-step horizon on GPU.

<details>
<summary><strong><span style="font-size: 1.25em;">Example config</span></strong></summary>

```yaml
agent:
  name: TimesFMForecastingAgent
  description: 'Agent for zero-shot time series forecasting using TimesFM'

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: TimeSeriesFromCSVLoader
    class_name: TimeSeriesFromCSVLoader
    template_input: InputTemplate
    attributes:
      root_dir: ./artifacts
      assign_to: "content"
      loader_params:
        path_to_csv: "bitcoin.csv"
        time_col: "Date"
        value_cols: ["open", "high", "low", "close", "Volume BTC", "Volume USD"]
        freq: "D"

  - template_name: TimesFM
    class_name: TimesFM
    template_input: TimeSeriesFromCSVLoader
    attributes:
      model_name: google/timesfm-2.5-200m-pytorch
      forecast_horizon: 30
      validation_mode: false
      time_series_output_format: pandas_dataframe
      device: cuda:0
      time_column_name: Date
```
</details>

This configuration defines an **agent** and a sequence of **templates** to load data and generate forecasts.

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.