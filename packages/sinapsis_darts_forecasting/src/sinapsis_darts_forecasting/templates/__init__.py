# -*- coding: utf-8 -*-
import importlib
from typing import Callable, cast

from sinapsis.templates import _import_template_package

_root_lib_path = "sinapsis_darts_forecasting.templates"
_ADDITIONAL_TEMPLATE_MODULES = [
    f"{_root_lib_path}.time_series_model",
    f"{_root_lib_path}.time_series_preprocessing",
]


_template_lookup: dict = {
    "TimeSeriesFromCSVLoader": f"{_root_lib_path}.time_series_from_csv_loader",
    "TimeSeriesFromDataframeLoader": f"{_root_lib_path}.time_series_from_dataframe_loader",
    "TimeSeriesFromSeriesLoader": f"{_root_lib_path}.time_series_from_series_loader",
}


for t_module in _ADDITIONAL_TEMPLATE_MODULES:
    _template_lookup |= _import_template_package(t_module)


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return cast(Callable, getattr(module, name))
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
