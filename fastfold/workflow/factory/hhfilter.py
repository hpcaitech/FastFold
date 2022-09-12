import subprocess
import logging
from typing import List

import ray 
from ray.dag.function_node import FunctionNode

from fastfold.workflow.factory import TaskFactory

class HHfilterFactory(TaskFactory):

    keywords = ['binary_path']

    def gen_node(self, fasta_path: str, output_path: str, after: List[FunctionNode]=None) -> FunctionNode:
        
        self.isReady()

        # generate function node
        @ray.remote
        def hhfilter_node_func(after: List[FunctionNode]) -> None:
            
            cmd = [
                self.config.get('binary_path'),
            ]
            if 'id' in self.config:
                cmd += ['-id', str(self.config.get('id'))]
            if 'cov' in self.config:
                cmd += ['-cov', str(self.config.get('cov'))]
            cmd += ['-i', fasta_path, '-o', output_path]

            subprocess.run(cmd, shell=True)

        return hhfilter_node_func.bind(after)