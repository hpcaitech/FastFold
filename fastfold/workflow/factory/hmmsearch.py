from typing import List
import inspect

import ray
from ray.dag.function_node import FunctionNode

from fastfold.data.tools import hmmsearch, hmmbuild
from fastfold.workflow.factory import TaskFactory
from type import Optional


class HmmSearchFactory(TaskFactory):

    keywords = ['binary_path', 'hmmbuild_binary_path', 'database_path', 'n_cpu']

    def gen_node(self, msa_sto: str, output_path: Optional[str] = None, after: List[FunctionNode]=None) -> FunctionNode:

        self.isReady()

        params = { k: self.config.get(k) for k in inspect.getfullargspec(hmmsearch.Hmmsearch.__init__).kwonlyargs if self.config.get(k) }
        
        # setup runner with a filtered config dict
        runner = hmmsearch.Hmmsearch(
            **params
        )

        # generate function node
        @ray.remote
        def hmmsearch_node_func(after: List[FunctionNode]) -> None:

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

        return hmmsearch_node_func.bind(after)
