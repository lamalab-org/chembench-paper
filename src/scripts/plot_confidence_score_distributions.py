import pandas as pd
import matplotlib.pyplot as plt
from paths import scripts
import seaborn as sns
from utils import obtain_chembench_repo
from plotutils import range_frame
import os

plt.style.use(scripts / "lamalab.org")


def plot(confidence_scores):
    chemnbench = obtain_chembench_repo
    gpt = pd.read_csv(os.path.join(chemnbench, "reports"))
