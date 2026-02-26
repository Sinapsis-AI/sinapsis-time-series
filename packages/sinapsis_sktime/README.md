<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis SKTime
<br>
</h1>

<h4 align="center">Module for handling time series data and forecasting using sktime.</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage Example</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

**Sinapsis SKTime** provides a powerful and flexible implementation for time series forecasting using the [sktime library](https://www.sktime.org/en/stable/).


<h2 id="installation"> 🐍  Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-sktime --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-sktime --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

The **Sinapsis SKTime** provides a powerful and flexible implementation for time series forecasting using the [sktime library](https://www.sktime.org/en/stable/).

<details>
<summary><strong><span style="font-size: 1.25em;">ThetaForecaster</span></strong></summary>

The following attributes apply to ThetaForecaster template:
- **`generic_field_key_data` (str, required)**: The key of the generic field containing the time series data.

This template loads time series data from the specified generic field key and creates a TimeSeriesPacket which is then added to the DataContainer.
</details>

<details>
<summary><strong><span style="font-size: 1.25em;">SKTime Forecasters</span></strong></summary>

The following attributes apply to all SKTime forecasting templates:
- **`generic_field_for_data` (str, optional)**: The key of the generic field where datasets are stored. Defaults to "SKTimeDatasets".
- **`model_save_path` (str, required)**: Path where the trained model will be saved.
- **`n_steps_ahead` (int, optional)**: Number of steps ahead to forecast. Defaults to 37.

The package provides access to a wide range of forecasting models from sktime, including:
- ARCH
- StatsForecastARCH
- StatsForecastGARCH
- ARIMA
- AutoARIMA
- ExponentialSmoothing
- StatsModelsARIMA
- NaiveForecaster
- NaiveVariance
- ThetaForecaster
- ThetaModularForecaster
- TrendForecaster
- PolynomialTrendForecaster
</details>

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis SKTime.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***ThetaForecaster*** use ```sinapsis info --example-template-config ThetaForecasterSKTimeWrapper``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: ThetaForecasterSKTimeWrapper
  class_name: ThetaForecasterSKTimeWrapper
  template_input: InputTemplate
  attributes:
    root_dir: '`replace_me:<class ''str''>`'
    model_save_path: '`replace_me:<class ''str''>`'
    n_steps_ahead: 37
    thetaforecaster_init:
      initial_level: null
      deseasonalize: true
      sp: 1
      deseasonalize_model: multiplicative
```

<h2 id="example"> 📚 Usage Example </h2>
Below is an example configuration for **Sinapsis SKTime** using a Theta forecasting model. This setup loads time series data into a TimeSeriesPacket and applies the Theta model for forecasting.

<details>
<summary><strong><span style="font-size: 1.25em;">Example config</span></strong></summary>

```yaml
agent:
  name: ThetaForecasterAgent
  description: 'Agent for time series forecasting using Theta'

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: load_airlineWrapper
    class_name: load_airlineWrapper
    template_input: InputTemplate
    attributes:
      split_dataset: true
      train_size: 0.8
      store_as_time_series: True
      load_airline:
        {}

  - template_name: ThetaForecasterSKTimeWrapper
    class_name: ThetaForecasterSKTimeWrapper
    template_input: load_airlineWrapper
    attributes:
      generic_field_key: load_airlineWrapper
      n_steps_ahead: 12
      forecast_horizon_in_fit: true
      model_save_path: "artifacts/theta_forecaster.pkl"
      task_type: "forecasting"
      thetaforecaster_init:
        sp: 12

```
</details>

This configuration defines an **agent** and a sequence of **templates** to handle the data and perform predictions.

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