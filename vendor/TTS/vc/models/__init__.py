import importlib
import logging
import re
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)


def setup_model(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "BaseVC":
    logger.info("Using model: %s", config.model)
    # fetch the right model implementation.
    if "model" in config and config["model"].lower() == "freevc":
        MyModel = importlib.import_module("TTS.vc.models.freevc").FreeVC
        model = MyModel.init_from_config(config, samples)
    return model
