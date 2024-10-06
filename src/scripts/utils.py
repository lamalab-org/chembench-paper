import os
from paths import data, scripts
from dotenv import load_dotenv

from scipy.constants import golden

ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2

ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

load_dotenv()
GH_PAT = os.getenv("GH_PAT")
print(GH_PAT)


def obtain_chembench_repo():
    # if in ../data the repository chem-bench is not found, clone it
    # if it is found, pull the latest version
    print("Obtaining ChemBench repository")
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(data, "chem-bench-main")):
        os.chdir(data)
        # download zip from https://github.com/lamalab-org/chem-bench/archive/refs/heads/humanbench.zip
        # using wget and then unzip it
        os.system(
            f"wget --header 'Authorization: token {GH_PAT}' https://github.com/lamalab-org/chem-bench/archive/refs/heads/main.zip"
        )
        os.system("unzip main.zip")

    else:
        os.chdir(os.path.join(data, "chem-bench-main"))
    os.chdir(cwd)
    return os.path.join(data, "chem-bench-main")
