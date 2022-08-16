from datetime import date
import imp
from ray import workflow
from typing import List
import time

import torch
import numpy as np
from scaFFold.core import TaskFactory
from ray.workflow.common import Workflow

from fastfold.distributed import init_dap
from fastfold.model.hub import AlphaFold
from fastfold.common import protein, residue_constants
from fastfold.config import model_config
from fastfold.data import data_pipeline, feature_pipeline, templates
from fastfold.utils import inject_fastnn
from fastfold.utils.import_weights import import_jax_weights_
from fastfold.utils.tensor_utils import tensor_tree_map

class AlphaFoldFactory(TaskFactory):

    keywords = ['kalign_bin_path', 'template_mmcif_dir', 'param_path', 'model_name']

    def gen_task(self, fasta_path: str, alignment_dir: str, output_path: str, after: List[Workflow]=None):

        self.isReady()

        # setup runners
        config = model_config(self.config.get('model_name'))
        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=self.config.get('template_mmcif_dir'),
            max_template_date=date.today().strftime("%Y-%m-%d"),
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=self.config.get('kalign_bin_path')
        )

        data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer)
        feature_processor = feature_pipeline.FeaturePipeline(config.data)

        # generate step function
        @workflow.step(num_gpus=1)
        def alphafold_step(fasta_path: str, alignment_dir: str, output_path: str, after: List[Workflow]) -> None:

            # setup model
            init_dap()
            model = AlphaFold(config)
            import_jax_weights_(model, self.config.get('param_path'), self.config.get('model_name'))
            model = inject_fastnn(model)
            model = model.eval()
            model = model.cuda()

            feature_dict = data_processor.process_fasta(
                fasta_path=fasta_path,
                alignment_dir=alignment_dir
            )
            processed_feature_dict = feature_processor.process_features(
                feature_dict,
                mode='predict'
            )
            with torch.no_grad():
                batch = {k: torch.as_tensor(v).cuda() for k, v in processed_feature_dict.items()}
                out = model(batch)
            batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
            plddt = out["plddt"]
            mean_plddt = np.mean(plddt)
            plddt_b_factors = np.repeat(plddt[..., None], residue_constants.atom_type_num, axis=-1)
            unrelaxed_protein = protein.from_prediction(features=batch,
                                            result=out,
                                            b_factors=plddt_b_factors)
            with open(output_path, 'w') as f:
                f.write(protein.to_pdb(unrelaxed_protein))

        return alphafold_step.step(fasta_path, alignment_dir, output_path, after)
