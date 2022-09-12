from typing import List

import ray
from ray.dag.function_node import FunctionNode

from fastfold.workflow.factory import TaskFactory
import fastfold.data.tools.hhblits as ffHHBlits

class HHBlitsFactory(TaskFactory):

    keywords = ['binary_path', 'databases', 'n_cpu']

    def gen_node(self, fasta_path: str, output_path: str, after: List[FunctionNode]=None) -> FunctionNode:
        
        self.isReady()

        # setup runner
        runner = ffHHBlits.HHBlits(
            **self.config
        )

        # generate function node
        @ray.remote
        def hhblits_node_func(after: List[FunctionNode]) -> None:
            result = runner.query(fasta_path)
            with open(output_path, 'w') as f:
                f.write(result['a3m'])

        return hhblits_node_func.bind(after)
