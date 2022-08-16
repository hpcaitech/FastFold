from FastFold.data.workflow import TaskFactory
from ray import workflow
from ray.workflow.common import Workflow
import fastfold.data.tools.hhsearch as ffHHSearch
from typing import List

class HHSearchFactory(TaskFactory):

    keywords = ['binary_path', 'databases', 'n_cpu']

    def gen_task(self, a3m_path: str, output_path: str, after: List[Workflow]=None) -> Workflow:

        self.isReady()
        
        # setup runner
        runner = ffHHSearch.HHSearch(
            binary_path=self.config['binary_path'],
            databases=self.config['databases'],
            n_cpu=self.config['n_cpu']
        )

        # generate step function
        @workflow.step
        def hhsearch_step(a3m_path: str, output_path: str, after: List[Workflow], atab_path: str = None) -> None:

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

        return hhsearch_step.step(a3m_path, output_path, after)
