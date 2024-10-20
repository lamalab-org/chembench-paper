import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import (
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    ONE_COL_WIDTH_INCH,
    obtain_chembench_repo,
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
        for term in [
            "google",
            "websearch",
            "web search",
            "googeln",
            "web",
            "docflex.com",
            "sciencedirect.com",
        ]
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
    elif str(tool).lower() in {"pse", "periodic table"}:
        return "PTE"
    elif any(
        term in str(tool).lower()
        for term in [
            "scifinder",
            "scifinder-n",
            "scifinder n",
            "Reaxys",
            "GESTIS-Stoffdatenbank",
            "https://www.nmrdb.org/",
            "PubChem",
        ]
    ):
        return "Database"
    else:
        return "Other"


if __name__ == "__main__":
    chembench = obtain_chembench_repo()
    df = pd.read_csv(
        os.path.join(chembench, "reports/humans/responses_20240918_161121.csv")
    )
    df["toolCategory"] = df["toolsUsed"].apply(categorize_tool)

    print("Unique tool categories after categorization:")
    print(df["toolCategory"].unique())

    with open(data / "human_tool_answered_questions.txt", "r") as f:
        answered_questions = f.read().splitlines()

    df_questions = pd.read_csv(
        os.path.join(chembench, "reports", "humans", "questions_20240918_161121.csv")
    )
    df = df.merge(df_questions, left_on="questionId", right_on="id")
    df_topic = pd.read_csv(
        os.path.join(chembench, "scripts", "classified_questions.csv")
    )
    df_topic["name"] = df_topic["index"].apply(lambda x: x.split("-")[-1])

    df = df.merge(df_topic, left_on="name", right_on="name")
    print()

    df_tool_allowed = df[(df["toolCategory"] != "No tools")]

    topic_tool_usage = (
        df_tool_allowed.groupby(["topic", "toolCategory"]).size().unstack(fill_value=0)
    )
    topic_tool_percentage = (
        topic_tool_usage.div(topic_tool_usage.sum(axis=1), axis=0) * 100
    )

    print("\nColumns in topic_tool_percentage:")
    print(topic_tool_percentage.columns)

    N = 16
    top_tools = topic_tool_usage.sum().nlargest(N).index.tolist()
    for tool in [
        "Web search",
        "Wikipedia",
        "Calculator",
        "ChemDraw",
        "Textbook",
        "PTE",
    ]:
        if tool not in top_tools:
            top_tools.append(tool)

    print("\nTop tools:")
    print(top_tools)

    plt.rcParams["figure.facecolor"] = "white"
    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )
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
    plt.ylabel("Percentage of Tool Usage")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xticklabels(
        [
            label.get_text().replace("Topics", "").strip()
            for label in ax.get_xticklabels()
        ],
    )

    range_frame(
        ax,
        np.arange(len(df_tool_allowed["toolCategory"].unique())),
        topic_tool_percentage.values,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    # Save as PDF
    fig.savefig(
        figures / "human_tool_usage_by_topic.pdf", format="pdf", bbox_inches="tight"
    )

    print("\nTools included in the plot:", top_tools)
