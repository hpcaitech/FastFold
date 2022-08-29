from typing import List
import inspect

import ray
from ray.dag.function_node import FunctionNode

import fastfold.data.tools.hhsearch as ffHHSearch
from fastfold.workflow.factory import TaskFactory


class HHSearchFactory(TaskFactory):

    keywords = ['binary_path', 'databases', 'n_cpu']

    def gen_node(self, a3m_path: str, output_path: str, atab_path: str = None, after: List[FunctionNode]=None) -> FunctionNode:

        self.isReady()

        params = { k: self.config.get(k) for k in inspect.getfullargspec(ffHHSearch.HHSearch.__init__).kwonlyargs if self.config.get(k) }
        
        # setup runner with a filtered config dict
        runner = ffHHSearch.HHSearch(
            **params
        )

        # generate function node
        @ray.remote
        def hhsearch_node_func(after: List[FunctionNode]) -> None:

            with open(a3m_path, "r") as f:
                a3m = f.read()
            if atab_path:
                hhsearch_result, atab = runner.query(a3m, gen_atab=True)
            else:
                hhsearch_result = runner.query(a3m)
            with open(output_path, "w") as f:
                f.write(hhsearch_result)
            if atab_path:
                with open(atab_path, "w") as f:
                    f.write(atab)

        return hhsearch_node_func.bind(after)
