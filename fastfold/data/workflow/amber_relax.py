from ray import workflow
from typing import List
from scaFFold.core import TaskFactory
from ray.workflow.common import Workflow

from fastfold.config import config
import fastfold.relax.relax as relax
from fastfold.common import protein

class AmberRelaxFactory(TaskFactory):

    keywords = []

    def gen_task(self, unrelaxed_pdb_path: str, output_path: str, after: List[Workflow]=None) -> Workflow:
        
        self.isReady()

        # setup runner
        amber_relaxer = relax.AmberRelaxation(
            use_gpu=True,
            **config.relax,
        )

        # generate step function
        @workflow.step(num_gpus=1)
        def amber_relax_step(unrelaxed_pdb_path: str, output_path: str, after: List[Workflow]) -> None:

            with open(unrelaxed_pdb_path, "r") as f:
                pdb_str = f.read()
            unrelaxed_protein = protein.from_pdb_string(pdb_str)
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

            with open(output_path, "w") as f:
                f.write(relaxed_pdb_str)

        return amber_relax_step.step(unrelaxed_pdb_path, output_path, after)
