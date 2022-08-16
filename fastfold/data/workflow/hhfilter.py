import subprocess
import logging
from ray import workflow
from typing import List
from scaFFold.core import TaskFactory
from ray.workflow.common import Workflow

class HHfilterFactory(TaskFactory):

    keywords = ['binary_path']

    def gen_task(self, fasta_path: str, output_path: str, after: List[Workflow]=None) -> Workflow:
        
        self.isReady()

        # generate step function
        @workflow.step
        def hhfilter_step(fasta_path: str, output_path: str, after: List[Workflow]) -> None:
            
            cmd = [
                self.config.get('binary_path'),
            ]
            if 'id' in self.config:
                cmd += ['-id', str(self.config.get('id'))]
            if 'cov' in self.config:
                cmd += ['-cov', str(self.config.get('cov'))]
            cmd += ['-i', fasta_path, '-o', output_path]

            logging.info(f"HHfilter start: {' '.join(cmd)}")

            subprocess.run(cmd)

        return hhfilter_step.step(fasta_path, output_path, after)
