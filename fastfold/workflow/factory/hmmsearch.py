from typing import List
import inspect

import ray
from ray.dag.function_node import FunctionNode

from fastfold.data.tools import hmmsearch, hmmbuild
from fastfold.data import parsers
from fastfold.workflow.factory import TaskFactory
from typing import Optional



class HmmSearchFactory(TaskFactory):

    keywords = ['binary_path', 'hmmbuild_binary_path', 'database_path', 'n_cpu']

    def gen_node(self, msa_sto_path: str, output_dir: Optional[str] = None, after: List[FunctionNode]=None) -> FunctionNode:

        self.isReady()

        params = { k: self.config.get(k) for k in inspect.getfullargspec(hmmsearch.Hmmsearch.__init__).kwonlyargs if self.config.get(k) }
        
        # setup runner with a filtered config dict
        runner = hmmsearch.Hmmsearch(
            **params
        )

        # generate function node
        @ray.remote
        def hmmsearch_node_func(after: List[FunctionNode]) -> None:

            with open(msa_sto_path, "r") as f:
                msa_sto = f.read()
                msa_sto = parsers.deduplicate_stockholm_msa(msa_sto)
                msa_sto = parsers.remove_empty_columns_from_stockholm_msa(
                    msa_sto
                )
                hmmsearch_result = runner.query(msa_sto, output_dir=output_dir)

        return hmmsearch_node_func.bind(after)
