import os
from paths import data


from scipy.constants import golden

ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2

ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden


def obtain_chembench_repo():
    # if in ../data the repository chem-bench is not found, clone it
    # if it is found, pull the latest version
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(data, "chem-bench")):
        os.chdir(data)
        os.system("git clone https://github.com/lamalab-org/chem-bench.git --depth 1")
        os.system("git checkout -b humanbench")
    else:
        os.chdir(os.path.join(data, "chem-bench"))
        os.system("git checkout -b humanbench")
        os.system("git pull")
    os.chdir(cwd)
    return os.path.join(data, "chem-bench")
