# coding=utf-8
import json
from typing import Dict, Any


class Config(Dict[str, str]):
    def __init__(self, file_path: str, **kwargs: Any):
        super().__init__(**kwargs)
        with open(file_path, mode="r") as file:
            self.update(json.load(file))
