from ast import keyword
import json
from ray.workflow.common import Workflow
from os import path
from typing import List

class TaskFactory:

    keywords = []

    def __init__(self, config: dict = None, config_path: str = None) -> None:
        
        # skip if no keyword required from config file
        if not self.__class__.keywords:
            return

        # setting config for factory
        if config is not None:
            self.config = config
        
    def configure(self, config: dict, purge=False) -> None:
        if purge:
            self.config = config
        else:
            self.config.update(config)

    def configure(self, keyword: str, value: any) -> None:
        self.config[keyword] = value

    def gen_task(self, after: List[Workflow]=None, *args, **kwargs) -> Workflow:
        raise NotImplementedError

    def isReady(self):
        for key in self.__class__.keywords:
            if key not in self.config:
                raise KeyError(f"{self.__class__.__name__} not ready: \"{key}\" not specified")

