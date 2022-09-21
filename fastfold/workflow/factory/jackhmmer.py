import inspect
from typing import List

import ray
from ray.dag.function_node import FunctionNode

import fastfold.data.tools.jackhmmer as ffJackHmmer
from fastfold.data import parsers
from fastfold.workflow.factory import TaskFactory


class JackHmmerFactory(TaskFactory):

    keywords = ['binary_path', 'database_path', 'n_cpu', 'uniref_max_hits']

    def gen_node(self, fasta_path: str, output_path: str, after: List[FunctionNode]=None, output_format: str="a3m") -> FunctionNode:

        self.isReady()

        params = { k: self.config.get(k) for k in inspect.getfullargspec(ffJackHmmer.Jackhmmer.__init__).kwonlyargs if self.config.get(k) }
        
        # setup runner
        runner = ffJackHmmer.Jackhmmer(
            **params
        )

        # generate function node
        @ray.remote
        def jackhmmer_node_func(after: List[FunctionNode]) -> None:
            result = runner.query(fasta_path)[0]
            if output_format == "a3m":
                uniref90_msa_a3m = parsers.convert_stockholm_to_a3m(
                    result['sto'],
                    max_sequences=self.config['uniref_max_hits']
                )
                with open(output_path, "w") as f:
                    f.write(uniref90_msa_a3m)
            elif output_format == "sto":
                template_msa = result['sto']
                with open(output_path, "w") as f:
                    f.write(template_msa)

            
        return jackhmmer_node_func.bind(after)
