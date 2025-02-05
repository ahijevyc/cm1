import os
from pathlib import Path
import subprocess

import f90nml


class PBS:
    def __init__(self, name: str, account: str, walltime: str, nodes: int, run_dir: Path):
        """
        Initialize PBS job configuration.

        :param name: Job name.
        :param account: Account number for the PBS job.
        :param walltime: Wall clock time for the job (e.g., '02:00:00' for 2 hours).
        :param nodes: Number of nodes to request.
        :param run_dir: Directory where the model run will be executed.
        """
        self.name = name
        self.account = account
        self.walltime = walltime
        self.nodes = nodes
        self.run_dir = run_dir if isinstance(run_dir, Path) else Path(run_dir)


class CM1Run:
    def __init__(self, namelist: f90nml.Namelist, pbs_config: PBS):
        """
        Initialize a CM1 model run.

        :param namelist: Instance of FortranNamelistController.
        :param pbs_config: Instance of PBS.
        """
        self.namelist = namelist
        self.pbs = pbs_config

    def generate_pbs_script(self, script_path="pbs.job"):
        """Generate a PBS job script for the CM1 model run."""
        script_content = f"""#!/bin/bash
# job name:
#PBS -N {self.pbs.name}
#PBS -A {self.pbs.account}
# below here, "select" is the number of 128-CPU nodes to use.
# e.g. select=4 will use 512 (=4*128) CPUs:
# (do not change settings for "ncpus" or "mpiprocs" or "ompthreads")
# For more info, see: https://arc.ucar.edu/knowledge_base/74317833
#
#PBS -l select={self.pbs.nodes}:ncpus=128:mpiprocs=128:ompthreads=1

# maximum wall-clock time (hh:mm:ss)
#PBS -l walltime={self.pbs.walltime}

# queue (works from derecho or casper)
#PBS -q main@desched1

# Write STDOUT and STDERR to run directory.
#PBS -o {self.pbs.run_dir}
#PBS -e {self.pbs.run_dir}

#-------------------------------------------

# temporary directory
export TMPDIR={os.getenv("TMPDIR")}
mkdir -p {os.getenv("TMPDIR")}

# These seem to work well for CM1 runs
export PALS_PPN=128
export PALS_DEPTH=1
export PALS_CPU_BIND=depth

cd {self.pbs.run_dir}

# run CM1
mpiexec --cpu-bind depth ./cm1.exe >& cm1.print.out
"""
        script_full_path = os.path.join(self.pbs.run_dir, script_path)
        with open(script_full_path, "w") as script_file:
            script_file.write(script_content)
        return script_full_path

    def submit_job(self):
        """Submit the PBS job."""
        script_path = self.generate_pbs_script()
        subprocess.run(["qsub", script_path], check=True)