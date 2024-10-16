from glob import glob
import os
from pathlib import Path
import subprocess
from utils import obtain_chembench_repo
from paths import output, data

if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()
    human_files = os.listdir(os.path.join(chembench_repo, "reports", "humans"))
    human_files = [
        os.path.join(chembench_repo, "reports", "humans", p)
        for p in human_files
    ]

    outpath = os.path.join(output, "human_scores")
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    datafolder = os.path.join(chembench_repo, "data")
    human_baseline_folder = os.path.join(chembench_repo, "reports", "humans")
    scriptpath = os.path.join(chembench_repo, "scripts", "collect_scores.py")

    for file in human_files:
        try:
            p = Path(file).parts[-1]
            outfile = os.path.join(outpath, p + ".json")
            subprocess.run(
                f"python {scriptpath} {file} {outfile} --datafolder={datafolder} --human_baseline_folder={human_baseline_folder}",
                shell=True,
            )
        except Exception:
            pass
