# -*- coding: utf-8 -*-
import importlib
from typing import Callable, cast

_root_lib_path = "sinapsis_timesfm.templates"


_template_lookup: dict = {
    "TimesFM": f"{_root_lib_path}.timesfm",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return cast(Callable, getattr(module, name))
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
