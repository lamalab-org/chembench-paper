from glob import glob
import os
from pathlib import Path
import subprocess
from utils import obtain_chembench_repo
from paths import output
from loguru import logger

if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()
    reports = glob(
        os.path.join(chembench_repo, "reports", "**", "reports", "**", "*.json")
    )

    models = list(set([Path(p).parent for p in reports]))
    logger.info(f"Found {len(models)} reports")
    logger.debug(f"First 5 reports: {models[:5]}")
    outpath = os.path.join(output, "overall_model_scores")
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    datafolder = os.path.join(chembench_repo, "data")
    human_baseline_folder = os.path.join(chembench_repo, "reports", "humans")
    scriptpath = os.path.join(chembench_repo, "scripts", "collect_scores.py")

    for file in models:
        try:
            p = Path(file).parts[-3]
            outfile = os.path.join(outpath, p + ".json")
            subprocess.run(
                f"python {scriptpath} {file} {outfile} --datafolder={datafolder}",
                shell=True,
            )
        except Exception:
            pass
