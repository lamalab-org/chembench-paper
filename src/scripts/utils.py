import os


def obtain_chembench_repo():
    # if in ../data the repository chem-bench is not found, clone it
    # if it is found, pull the latest version
    if not os.path.exists("../data/chem-bench"):
        os.system("git clone https://github.com/lamalab-org/chem-bench-paper.git")
    else:
        os.system("cd ../data/chem-bench; git pull")
    return "../data/chem-bench"
