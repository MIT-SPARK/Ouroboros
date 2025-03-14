import os
import tempfile
from dataclasses import dataclass
from typing import Any

from spark_config import Config, config_field, register_config


class ExampleTopLevel:
    def __init__(self, config):
        self.config = config
        self.mouse = self.config.a_virtual_subconfig.create()


@register_config("top_level", name="ExampleTopLevel", constructor=ExampleTopLevel)
@dataclass
class ExampleTopLevelConfig(Config):
    a_number: int = 0
    another_number: int = 0
    a_string: str = "a string"
    a_virtual_subconfig: Any = config_field("mouse", default="Mouse")


class Mouse:
    def __init__(self, config):
        self.color = config.color


@register_config("mouse", name="Mouse", constructor=Mouse)
@dataclass
class MouseConfig(Config):
    color: str = "gray"


if __name__ == "__main__":
    tl_cfg = ExampleTopLevelConfig()

    d = tempfile.gettempdir()
    fn = os.path.join(d, "example_config.yaml")
    print(f"Saving example config in {fn}")
    tl_cfg.save(fn)

    """
    Generates the following yaml file in config/example_config.yaml:
    a_number: 0
    a_string: a string
    a_virtual_subconfig:
      color: gray
      type: Mouse
    another_number: 0
    """

    loaded_cfg = Config.load(ExampleTopLevelConfig, fn)
    print("loaded_cfg: ", loaded_cfg)
