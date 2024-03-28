import matplotlib.pyplot as plt
from paths import scripts, figures
from utils import ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH

plt.style.use(scripts / "lamalab.org")

# make a dummy plots
fig, ax = plt.subplots(
    figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))

ax.plot([1, 2, 3], [1, 2, 3])

fig.savefig(figures / "llama_performance.pdf", bbox_inches="tight")
