from ray import workflow
from typing import List
from FastFold.data.workflow import TaskFactory
from ray.workflow.common import Workflow
import fastfold.data.tools.hhblits as ffHHBlits

class HHBlitsFactory(TaskFactory):

    keywords = ['binary_path', 'databases', 'n_cpu']

    def gen_task(self, fasta_path: str, output_path: str, after: List[Workflow]=None) -> Workflow:
        
        self.isReady()

        # setup runner
        runner = ffHHBlits.HHBlits(
            **self.config
        )

        # generate step function
        @workflow.step
        def hhblits_step(fasta_path: str, output_path: str, after: List[Workflow]) -> None:
            result = runner.query(fasta_path)
            with open(output_path, "w") as f:
                f.write(result["a3m"])

        return hhblits_step.step(fasta_path, output_path, after)
