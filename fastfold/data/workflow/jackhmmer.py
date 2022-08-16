from FastFold.data.workflow import TaskFactory
from ray import workflow
from ray.workflow.common import Workflow
import fastfold.data.tools.jackhmmer as ffJackHmmer
from fastfold.data import parsers
from typing import List

class JackHmmerFactory(TaskFactory):

    keywords = ['binary_path', 'database_path', 'n_cpu', 'uniref_max_hits']

    def gen_task(self, fasta_path: str, output_path: str, after: List[Workflow]=None) -> Workflow:

        self.isReady()
        
        # setup runner
        runner = ffJackHmmer.Jackhmmer(
            binary_path=self.config['binary_path'],
            database_path=self.config['database_path'],
            n_cpu=self.config['n_cpu']
        )

        # generate step function
        @workflow.step
        def jackhmmer_step(fasta_path: str, output_path: str, after: List[Workflow]) -> None:
            result = runner.query(fasta_path)[0]
            uniref90_msa_a3m = parsers.convert_stockholm_to_a3m(
                result['sto'],
                max_sequences=self.config['uniref_max_hits']
            )
            with open(output_path, "w") as f:
                f.write(uniref90_msa_a3m)
        
        return jackhmmer_step.step(fasta_path, output_path, after)
