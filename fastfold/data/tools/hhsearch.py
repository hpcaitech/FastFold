# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library to run HHsearch from Python."""
import glob
import logging
import os
import subprocess
from typing import Sequence, Union

from fastfold.data.tools import utils


class HHSearch:
    """Python wrapper of the HHsearch binary."""

    def __init__(
        self,
        *,
        binary_path: str,
        databases: Sequence[str],
        n_cpu: int = 2,
        maxseq: int = 1_000_000,
        mact: float = 0.35,
        min_align: int = 10,
        max_align: int = 500,
        min_lines: int = 10,
        max_lines: int = 500,
        aliw: int = 100000,
        e_value: float = 0.001,
        min_prob: float = 20.0,
    ):
        """Initializes the Python HHsearch wrapper.

        Args:
          binary_path: The path to the HHsearch executable.
          databases: A sequence of HHsearch database paths. This should be the
            common prefix for the database files (i.e. up to but not including
            _hhm.ffindex etc.)
          n_cpu: The number of CPUs to use
          maxseq: The maximum number of rows in an input alignment. Note that this
            parameter is only supported in HHBlits version 3.1 and higher.
          mact: Posterior probability threshold for MAC realignment controlling greediness at alignment
            ends.
          min_align: Minimum number of alignments in alignment list. (-b)
          max_align: Maximum number of alignments in alignment list. (-B)
          min_lines: Minimum number of lines in summary hit list. (-z)
          max_lines: Maximum number of lines in summary hit list. (-Z)
          aliw: Number of columns per line in alignment list.
          e_value: E-value cutoff for inclusion in result alignment. (-e)
          min_prob: Minimum probability in summary and alignment list. (-p)

        Raises:
          RuntimeError: If HHsearch binary not found within the path.
        """
        self.binary_path = binary_path
        self.databases = databases
        self.n_cpu = n_cpu
        self.maxseq = maxseq
        self.mact = mact
        self.min_align = min_align
        self.max_align = max_align
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.aliw = aliw
        self.e_value = e_value
        self.min_prob = min_prob

        for database_path in self.databases:
            if not glob.glob(database_path + "_*"):
                logging.error(
                    "Could not find HHsearch database %s", database_path
                )
                raise ValueError(
                    f"Could not find HHsearch database {database_path}"
                )
                
    @property
    def output_format(self) -> str:
        return 'hhr'

    @property
    def input_format(self) -> str:
        return 'a3m'

    def query(self, a3m: str, gen_atab: bool = False) -> Union[str, tuple]:
        """Queries the database using HHsearch using a given a3m."""
        with utils.tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
            input_path = os.path.join(query_tmp_dir, "query.a3m")
            hhr_path = os.path.join(query_tmp_dir, "output.hhr")
            atab_path = os.path.join(query_tmp_dir, "output.atab")
            with open(input_path, "w") as f:
                f.write(a3m)

            db_cmd = []
            for db_path in self.databases:
                db_cmd.append("-d")
                db_cmd.append(db_path)
            cmd = [
                self.binary_path,
                "-i",
                input_path,
                "-o",
                hhr_path,
                "-maxseq",
                str(self.maxseq),
                "-cpu",
                str(self.n_cpu),
                "-b",
                str(self.min_align),
                "-B",
                str(self.max_align),
                "-z",
                str(self.min_lines),
                "-Z",
                str(self.max_lines),
                "-mact",
                str(self.mact),
                "-aliw",
                str(self.aliw),
                "-e",
                str(self.e_value),
                "-p",
                str(self.min_prob),
            ] + db_cmd
            if gen_atab:
                cmd += ["-atab", atab_path]

            logging.info('Launching subprocess "%s"', " ".join(cmd))
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            with utils.timing("HHsearch query"):
                stdout, stderr = process.communicate()
                retcode = process.wait()

            if retcode:
                # Stderr is truncated to prevent proto size errors in Beam.
                raise RuntimeError(
                    "HHSearch failed:\nstdout:\n%s\n\nstderr:\n%s\n"
                    % (stdout.decode("utf-8"), stderr[:100_000].decode("utf-8"))
                )

            with open(hhr_path) as f:
                hhr = f.read()
            if gen_atab:
                with open(atab_path) as f:
                    atab = f.read()
        if gen_atab:
            return hhr, atab
        else:
            return hhr
