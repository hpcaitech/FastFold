from ast import Call
from typing import Callable, List
from ray.workflow.common import Workflow
from ray import workflow

def batch_run(wfs: List[Workflow], workflow_id: str) -> None:

    @workflow.step
    def batch_step(wfs) -> None:
        return

    batch_wf = batch_step.step(wfs)

    batch_wf.run(workflow_id=workflow_id)

def wf(after: List[Workflow]=None):
    def decorator(f: Callable):

        @workflow.step
        def step_func(after: List[Workflow]) -> None:
            f()
        
        return step_func.step(after)
    
    return decorator
