import os
import time
from multiprocessing import cpu_count
import ray
from ray import workflow
from fastfold.data.tools import hhsearch, hmmsearch
from fastfold.workflow.factory import JackHmmerFactory, HHSearchFactory, HHBlitsFactory
from fastfold.workflow import batch_run
from typing import Optional, Union

TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


class FastFoldMultimerDataWorkFlow:
    def __init__(
        self,
        jackhmmer_binary_path: Optional[str] = None,
        hhblits_binary_path: Optional[str] = None,
        uniref90_database_path: Optional[str] = None,
        mgnify_database_path: Optional[str] = None,
        bfd_database_path: Optional[str] = None,
        uniclust30_database_path: Optional[str] = None,
        uniprot_database_path: Optional[str] = None,
        template_searcher: Optional[TemplateSearcher] = None,
        use_small_bfd: Optional[bool] = None,
        no_cpus: Optional[int] = None,
        uniref_max_hits: int = 10000,
        mgnify_max_hits: int = 5000,
        uniprot_max_hits: int = 50000,
    ):
        db_map = {
            "jackhmmer": {
                "binary": jackhmmer_binary_path,
                "dbs": [
                    uniref90_database_path,
                    mgnify_database_path,
                    bfd_database_path if use_small_bfd else None,
                    uniprot_database_path,
                ],
            },
            "hhblits": {
                "binary": hhblits_binary_path,
                "dbs": [
                    bfd_database_path if not use_small_bfd else None,
                ],
            },
        }


        for name, dic in db_map.items():
            binary, dbs = dic["binary"], dic["dbs"]
            if(binary is None and not all([x is None for x in dbs])):
                raise ValueError(
                    f"{name} DBs provided but {name} binary is None"
                )

        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits
        self.uniprot_max_hits = uniprot_max_hits
        self.use_small_bfd = use_small_bfd

        if(no_cpus is None):
            no_cpus = cpu_count()

        # create JackHmmer workflow generator
        self.jackhmmer_uniref90_factory = None
        if jackhmmer_binary_path is not None and uniref90_database_path is not None:
            jh_config = {
                "binary_path": db_map["jackhmmer"]["binary"],
                "database_path": uniref90_database_path,
                "n_cpu": no_cpus,
                "uniref_max_hits": uniref_max_hits,
            }
            self.jackhmmer_uniref90_factory = JackHmmerFactory(config = jh_config)

        # create HHSearch workflow generator
        self.hhmmsearch_pdb_factory = None
        if pdb70_database_path is not None:
            hhs_config = {
                "binary_path": db_map["hhsearch"]["binary"],
                "databases": db_map["hhsearch"]["dbs"],
                "n_cpu": self.no_cpus,
            }
            self.hhsearch_pdb_factory = HHSearchFactory(config=hhs_config)


        self.jackhmmer_mgnify_factory = None
        if jackhmmer_binary_path is not None and mgnify_database_path is not None:
            jh_config = {
                "binary_path": db_map["jackhmmer"]["binary"],
                "database_path": mgnify_database_path,
                "n_cpu": no_cpus,
                "uniref_max_hits": mgnify_max_hits,
            }
            self.jackhmmer_mgnify_factory = JackHmmerFactory(config=jh_config)

        if bfd_database_path is not None:
            if not use_small_bfd:
                hhb_config = {
                    "binary_path": db_map["hhblits"]["binary"],
                    "databases": db_map["hhblits"]["dbs"],
                    "n_cpu": self.no_cpus,
                }
                self.hhblits_bfd_factory = HHBlitsFactory(config=hhb_config)
            else:
                jh_config = {
                    "binary_path": db_map["jackhmmer"]["binary"],
                    "database_path": bfd_database_path,
                    "n_cpu": no_cpus,
                }
                self.jackhmmer_small_bfd_factory = JackHmmerFactory(config=jh_config)

        self.jackhmmer_uniprot_factory = None
        if jackhmmer_binary_path is not None and uniprot_database_path is not None:
            jh_config = {
                "binary_path": db_map["jackhmmer"]["binary"],
                "database_path": uniprot_database_path,
                "n_cpu": no_cpus,
                "uniref_max_hits": uniprot_max_hits,
            }
            self.jackhmmer_uniprot_factory = JackHmmerFactory(config=jh_config)


    def run(self, fasta_path: str, output_dir: str, alignment_dir: str=None, storage_dir: str=None) -> None:
        storage_dir = "file:///tmp/ray/lcmql/workflow_data"
        if storage_dir is not None:
            if not os.path.exists(storage_dir):
                os.makedirs(storage_dir)
            ray.init(storage=storage_dir)

        localtime = time.asctime(time.localtime(time.time()))
        workflow_id = 'fastfold_data_workflow ' + str(localtime)
        # clearing remaining ray workflow data
        try:
            workflow.cancel(workflow_id)
            workflow.delete(workflow_id)
        except:
            print("Workflow not found. Clean. Skipping")
            pass

        
        # prepare alignment directory for alignment outputs
        if alignment_dir is None:
            alignment_dir = os.path.join(output_dir, "alignment")
            if not os.path.exists(alignment_dir):
                os.makedirs(alignment_dir)

        # Run JackHmmer on UNIREF90
        uniref90_out_path = os.path.join(alignment_dir, "uniref90_hits.a3m")
        # generate the workflow with i/o path
        uniref90_node = self.jackhmmer_uniref90_factory.gen_node(fasta_path, uniref90_out_path)

        #Run HHSearch on STEP1's result with PDB70"""
        pdb70_out_path = os.path.join(alignment_dir, "pdb70_hits.hhr")
        # generate the workflow (STEP2 depend on STEP1)
        hhs_node = self.hhsearch_pdb_factory.gen_node(uniref90_out_path, pdb70_out_path, after=[uniref90_node])

        # Run JackHmmer on MGNIFY
        mgnify_out_path = os.path.join(alignment_dir, "mgnify_hits.a3m")
        # generate workflow for STEP3
        mgnify_node = self.jackhmmer_mgnify_factory.gen_node(fasta_path, mgnify_out_path)

        if not self.use_small_bfd:
            # Run HHBlits on BFD
            bfd_out_path = os.path.join(alignment_dir, "bfd_uniclust_hits.a3m")
            # generate workflow for STEP4
            bfd_node = self.hhblits_bfd_factory.gen_node(fasta_path, bfd_out_path)

        else:
            # Run Jackhmmer on small_bfd
            bfd_out_path = os.path.join(alignment_dir, "bfd_uniclust_hits.a3m")
            # generate workflow for STEP4_2
            bfd_node = self.jackhmmer_small_bfd_factory.gen_node(fasta_path, bfd_out_path)

        # run workflow
        batch_run(workflow_id=workflow_id, dags=[hhs_node, mgnify_node, bfd_node]) 

        return