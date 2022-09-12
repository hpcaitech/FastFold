from ast import Call
from typing import Callable, List
import ray
from ray.dag.function_node import FunctionNode
from ray import workflow

def batch_run(workflow_id: str, dags: List[FunctionNode]) -> None:

    @ray.remote
    def batch_dag_func(dags) -> None:
        return

    batch = batch_dag_func.bind(dags)
    workflow.run(batch, workflow_id=workflow_id)
