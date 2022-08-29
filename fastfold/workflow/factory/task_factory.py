from ast import keyword
import json
from os import path
from typing import List
import ray
from ray.dag.function_node import FunctionNode

class TaskFactory:

    keywords = []

    def __init__(self, config: dict = None, config_path: str = None) -> None:
        
        # skip if no keyword required from config file
        if not self.__class__.keywords:
            return

        # setting config for factory
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.loadConfig(config_path)
        else:
            self.loadConfig()
        
    def configure(self, config: dict, purge=False) -> None:
        if purge:
            self.config = config
        else:
            self.config.update(config)

    def configure(self, keyword: str, value: any) -> None:
        self.config[keyword] = value

    def gen_task(self, after: List[FunctionNode]=None, *args, **kwargs) -> FunctionNode:
        raise NotImplementedError

    def isReady(self):
        for key in self.__class__.keywords:
            if key not in self.config:
                raise KeyError(f"{self.__class__.__name__} not ready: \"{key}\" not specified")

    def loadConfig(self, config_path='./config.json'):
        with open(config_path) as configFile:
            globalConfig = json.load(configFile)
            if 'tools' not in globalConfig:
                raise KeyError("\"tools\" not found in global config file")
            factoryName = self.__class__.__name__[:-7]
            if factoryName not in globalConfig['tools']:
                raise KeyError(f"\"{factoryName}\" not found in the \"tools\" section in config")
            self.config = globalConfig['tools'][factoryName]