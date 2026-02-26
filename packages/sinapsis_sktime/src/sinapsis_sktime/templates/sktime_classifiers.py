# -*- coding: utf-8 -*-
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import UIPropertiesMetadata
from sinapsis_core.template_base.dynamic_template import WrapperEntryConfig
from sinapsis_core.template_base.dynamic_template_factory import make_dynamic_template
from sinapsis_core.utils.env_var_keys import SINAPSIS_BUILD_DOCS
from sinapsis_sktime.helpers.tags import Tags
from sinapsis_sktime.templates.sktime_models_base import SKTimeModelsBase
from sktime.classification import compose, deep_learning, dictionary_based, distance_based, dummy, feature_based

EXCLUDED_MODELS = [
    "ClassifierPipeline",
    "ColumnEnsembleClassifier",
    "ComposableTimeSeriesForestClassifier",
    "MultiplexClassifier",
    "SignatureClassifier",
    "WeightedEnsembleClassifier",
]


class SKTimeDistanceClassifiers(SKTimeModelsBase):
    """Template for SkTime classification model training and inference for the distance_based module"""

    TASK_TYPE = "classification"
    WrapperEntry = WrapperEntryConfig(
        wrapped_object=distance_based,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
        exclude_module_atts=EXCLUDED_MODELS,
    )
    UIProperties = UIPropertiesMetadata(
        category="SKTime", tags=[Tags.DYNAMIC, Tags.INFERENCE, Tags.MODELS, Tags.PREDICTION, Tags.CLASSIFICATION]
    )


class SKTimeDictionaryClassifiers(SKTimeDistanceClassifiers):
    """Template for SkTime classification model training and inference for the dictionary_based module"""

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=dictionary_based,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
        exclude_module_atts=EXCLUDED_MODELS,
    )


class SKTimeFeatureClassifiers(SKTimeDistanceClassifiers):
    """Template for SkTime classification model training and inference for the feature_based module"""

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=feature_based,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
        exclude_module_atts=EXCLUDED_MODELS,
    )


class SKTimeDeepLearningClassifiers(SKTimeDistanceClassifiers):
    """Template for SkTime classification model training and inference for the deep_learning module"""

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=deep_learning,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
        exclude_module_atts=EXCLUDED_MODELS,
    )


class SKTimeComposeClassifiers(SKTimeDistanceClassifiers):
    """Template for SkTime classification model training and inference for the compose module"""

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=compose,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
        exclude_module_atts=EXCLUDED_MODELS,
    )


class SKTimeDummyClassifiers(SKTimeDistanceClassifiers):
    """Template for SkTime classification model training and inference for the dummy module"""

    WrapperEntry = WrapperEntryConfig(
        wrapped_object=dummy,
        template_name_suffix="SKTimeWrapper",
        signature_from_doc_string=True,
        force_init_as_method=False,
        exclude_module_atts=EXCLUDED_MODELS,
    )


def __getattr__(name: str) -> Template:
    """
    Only create a template if it's imported, this avoids creating all the base models for all templates
    and potential import errors due to not available packages.
    """
    if name in SKTimeDeepLearningClassifiers.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeDeepLearningClassifiers)

    if name in SKTimeDictionaryClassifiers.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeDictionaryClassifiers)

    if name in SKTimeDistanceClassifiers.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeDistanceClassifiers)

    if name in SKTimeDummyClassifiers.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeDummyClassifiers)

    if name in SKTimeFeatureClassifiers.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeFeatureClassifiers)

    if name in SKTimeComposeClassifiers.WrapperEntry.module_att_names:
        return make_dynamic_template(name, SKTimeComposeClassifiers)

    raise AttributeError(f"template `{name}` not found in {__name__}")


__all__ = (
    SKTimeDistanceClassifiers.WrapperEntry.module_att_names
    + SKTimeDictionaryClassifiers.WrapperEntry.module_att_names
    + SKTimeFeatureClassifiers.WrapperEntry.module_att_names
    + SKTimeDeepLearningClassifiers.WrapperEntry.module_att_names
    + SKTimeDummyClassifiers.WrapperEntry.module_att_names
    + SKTimeComposeClassifiers.WrapperEntry.module_att_names
)


if SINAPSIS_BUILD_DOCS:
    dynamic_templates: list[Template] = [__getattr__(template_name) for template_name in __all__]
    for template in dynamic_templates:
        globals()[template.__name__] = template
        del template
