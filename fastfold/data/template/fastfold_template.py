import os
from multiprocessing import cpu_count
from ray import workflow
from FastFold.data.workflow import JackHmmerFactory, HHSearchFactory, HHBlitsFactory, AlphaFoldFactory, AmberRelaxFactory
from typing import Mapping, Optional, Sequence, Any

# only the first sequence in FASTA file is processed

class FastFoldAlignmentWorkFlow:
    def __init__(
        self,
        jackhmmer_binary_path: Optional[str] = None,
        hhblits_binary_path: Optional[str] = None,
        hhsearch_binary_path: Optional[str] = None,
        uniref90_database_path: Optional[str] = None,
        mgnify_database_path: Optional[str] = None,
        bfd_database_path: Optional[str] = None,
        kalign_binary_path: Optional[str] = None,
        template_mmcif_dir: Optional[str] = None,
        param_path: Optional[str] = None,
        model_name: Optional[str] = None,
        uniclust30_database_path: Optional[str] = None,
        pdb70_database_path: Optional[str] = None,
        use_small_bfd: Optional[bool] = None,
        no_cpus: Optional[int] = None,
        uniref_max_hits: int = 10000,
        mgnify_max_hits: int = 5000,
    ):
        self.db_map = {
            "jackhmmer": {
                "binary": jackhmmer_binary_path,
                "dbs": [
                    uniref90_database_path,
                    mgnify_database_path,
                    bfd_database_path if use_small_bfd else None,
                ],
            },
            "hhblits": {
                "binary": hhblits_binary_path,
                "dbs": [
                    bfd_database_path if not use_small_bfd else None,
                ],
            },
            "hhsearch": {
                "binary": hhsearch_binary_path,
                "dbs": [
                    pdb70_database_path,
                ],
            },
        }

        for name, dic in self.db_map.items():
            binary, dbs = dic["binary"], dic["dbs"]
            if(binary is None and not all([x is None for x in dbs])):
                raise ValueError(
                    f"{name} DBs provided but {name} binary is None"
                )

        if(not all([x is None for x in db_map["hhsearch"]["dbs"]])
            and uniref90_database_path is None):
            raise ValueError(
                """uniref90_database_path must be specified in order to perform
                    template search"""
            )

        self.use_small_bfd = use_small_bfd
        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits
        self.kalign_binary_path = kalign_binary_path
        self.template_mmcif_dir = template_mmcif_dir
        self.param_path = param_path
        self.model_name = model_name

        if(no_cpus is None):
            self.no_cpus = cpu_count()
        else:
            self.no_cpus = no_cpus

    def run(self, fasta_path: str, output_dir: str, alignment_dir: str=None) -> None:

        # clearing remaining ray workflow data
        try:
            workflow.cancel('FastFold')
            workflow.delete('FastFold')
        except:
            print("Workflow not found. Clean. Skipping")
            pass
        
        # prepare alignment directory for alignment outputs
        if alignment_dir is None:
            alignment_dir = os.path.join(output_dir, "alignment")
            if not os.path.exists(alignment_dir):
                os.makedirs(alignment_dir)
        
        # prepare database paths

        """STEP1: Run JackHmmer on UNIREF90"""

        # create JackHmmer workflow generator
        jh_config = {
            "binary_path": self.db_map["jackhmmer"]["binary"],
            "database_path": self.db_map["jackhmmer"]["dbs"][0],
            "n_cpu": self.no_cpus,
        }
        jh_fac = JackHmmerFactory(config = jh_config)
        # set jackhmmer output path
        uniref90_out_path = os.path.join(alignment_dir, "uniref90_hits.a3m")
        # generate the workflow with i/o path
        wf1 = jh_fac.gen_task(fasta_path, uniref90_out_path)

        """STEP2: Run HHSearch on STEP1's result with PDB70"""

        # create HHSearch workflow generator
        hhs_config = {
            "binary_path": self.db_map["hhsearch"]["binary"],
            "database_path": self.db_map["hhsearch"]["dbs"][0],
            "n_cpu": self.no_cpus,
        }
        hhs_fac = HHSearchFactory(config=hhs_config)
        # set HHSearch output path
        pdb70_out_path = os.path.join(alignment_dir, "pdb70_hits.hhr")
        # generate the workflow (STEP2 depend on STEP1)
        wf2 = hhs_fac.gen_task(uniref90_out_path, pdb70_out_path, after=[wf1])

        """STEP3: Run JackHmmer on MGNIFY"""

        # reconfigure jackhmmer factory to use MGNIFY DB instead
        jh_fac.configure('database_path', self.db_map["jackhmmer"]["dbs"][1])
        # set jackhmmer output path
        mgnify_out_path = os.path.join(alignment_dir, "mgnify_hits.a3m")
        # generate workflow for STEP3
        wf3 = jh_fac.gen_task(fasta_path, mgnify_out_path)

        """STEP4: Run HHBlits on BFD"""

        # create HHBlits workflow generator
        hhb_config = {
            "binary_path": self.db_map["hhblits"]["binary"],
            "database_path": self.db_map["hhblits"]["dbs"][0],
            "n_cpu": self.no_cpus,
        }
        hhb_fac = HHBlitsFactory(config=hhb_config)
        # set HHBlits output path
        bfd_out_path = os.path.join(alignment_dir, "bfd_uniclust_hits.a3m")
        # generate workflow for STEP4
        wf4 = hhb_fac.gen_task(fasta_path, bfd_out_path)

        """STEP5: Generate Features"""

        # create AlphaFold2 workflow generator
        af_config = {
            "kalign_bin_path": self.kalign_binary_path,
            "template_mmcif_dir": self.template_mmcif_dir,
            "param_path": self.param_path,
            "model_name": self.model_name,
        }
        af_fac = AlphaFoldFactory(config=af_config)
        # set output unrelaxed PDB path
        unrelaxed_pdb_path = os.path.join(output_dir, "unrelaxed_result.pdb")
        # generate workflow for STEP5
        wf5 = af_fac.gen_task(fasta_path, alignment_dir, unrelaxed_pdb_path, [wf2, wf3, wf4])

        # run workflow
        wf5.run(workflow_id='FastFold')
        return
