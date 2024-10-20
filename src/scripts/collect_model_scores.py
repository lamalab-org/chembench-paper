import os
import subprocess
from utils import obtain_chembench_repo
from paths import output
from loguru import logger
from paths import data
import pickle

if __name__ == "__main__":
    chembench_repo = obtain_chembench_repo()

    with open(data / "name_to_dir_map.pkl", "rb") as handle:
        name_dir_map = pickle.load(handle)

    outpath = os.path.join(output, "overall_model_scores")
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    datafolder = os.path.join(chembench_repo, "data")
    human_baseline_folder = os.path.join(chembench_repo, "reports", "humans")
    scriptpath = os.path.join(chembench_repo, "scripts", "collect_scores.py")

    for model, directory in name_dir_map.items():
        print(f"Processing {directory}")
        try:
            outfile = os.path.join(outpath, model + ".json")
            subprocess.run(
                f'python {scriptpath} "{directory}" {outfile} --datafolder={datafolder}',
                shell=True,
            )
        except Exception as e:
            logger.error(f"Error processing {directory}: {e}")
            pass
