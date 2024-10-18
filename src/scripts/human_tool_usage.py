import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import (
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_WIDTH_INCH,
    TWO_COL_WIDTH_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
)
from paths import output, figures, scripts, data, static
from plotutils import range_frame

plt.style.use(scripts / "lamalab.mplstyle")


def categorize_tool(tool):
    if pd.isna(tool) or tool in ["no tools", "No tool used", "No tools used"]:
        return "No tools"
    elif any(
        term in str(tool).lower()
        for term in ["google", "websearch", "web search", "googeln", "web"]
    ):
        return "Web search"
    elif "calculator" in str(tool).lower():
        return "Calculator"
    elif "wikipedia" in str(tool).lower() or "wiki" in str(tool).lower():
        return "Wikipedia"
    elif "chemdraw" in str(tool).lower():
        return "ChemDraw"
    elif any(term in str(tool).lower() for term in ["book", "textbook"]):
        return "Textbook"
    elif str(tool).lower() in ["pse", "periodic table"]:
        return "PSE"
    else:
        return tool


df = pd.read_csv(data / "responses_updated_cleaned_toolUseAllowed.csv")
df["toolCategory"] = df["toolsUsed"].apply(categorize_tool)

print("Unique tool categories after categorization:")
print(df["toolCategory"].unique())

df_tool_allowed = df[
    (df["toolUseAllowed"] == True) & (df["toolCategory"] != "No tools")
]
topic_tool_usage = (
    df_tool_allowed.groupby(["topic", "toolCategory"]).size().unstack(fill_value=0)
)
topic_tool_percentage = topic_tool_usage.div(topic_tool_usage.sum(axis=1), axis=0) * 100

print("\nColumns in topic_tool_percentage:")
print(topic_tool_percentage.columns)

N = 16
top_tools = topic_tool_usage.sum().nlargest(N).index.tolist()
for tool in ["Web search", "Wikipedia", "Calculator", "ChemDraw", "Textbook", "PSE"]:
    if tool not in top_tools:
        top_tools.append(tool)

print("\nTop tools:")
print(top_tools)

plt.rcParams["figure.facecolor"] = "white"
fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH))
ax.set_facecolor("white")

colors = [
    "#4ECDC4",
    "#F7DC6F",
    "#A498DB",
    "#AB8D9E",
    "#85C1E9",
    "#FDA07A",
    "#82E0AA",
    "#CBBEB5",
    "#E2908A",
    "#F39C12",
    "#98D8C8",
    "#2ECC71",
    "#E74C3C",
    "#9B59B6",
    "#1ABC9C",
    "#34495E",
]

topic_tool_percentage[top_tools].plot(
    kind="bar", stacked=True, ax=ax, color=colors[: len(top_tools)]
)
plt.xlabel("")
plt.ylabel("Percentage of Tool Usage", fontsize=12)
plt.legend(
    title="", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small"
)
plt.xticks(rotation=45, ha="right")
ax.set_xticklabels(
    [label.get_text().replace("Topics", "").strip() for label in ax.get_xticklabels()]
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

# Save as PDF
plt.savefig(
    static / "human_tool_usage_by_topic.pdf", format="pdf", dpi=300, bbox_inches="tight"
)
plt.show()

print("\nTools included in the plot:", top_tools)
