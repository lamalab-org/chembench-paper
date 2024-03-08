from chembench.analysis import load_all_reports
from utils import obtain_chembench_repo
import os


def load_reports():
    claude2 = load_all_reports(
        "../reports/claude2/reports/7536581a-d92d-43d3-b946-39e4a7213b7f/"
    )
    claude2_zero_t = load_all_reports(
        "../reports/claude2-zero-T/reports/412657cc-7a11-4d73-80e9-03d6f05cd63e"
    )

    fb_llama_70b = load_all_reports(
        "../reports/llama-2-70b-chat/reports/3f8361fa-8fbf-4704-bccd-b0489f401d4e"
    )

    galactica_120b = load_all_reports(
        "../reports/galatica_120b/reports/82194130-de35-49b8-8375-ac89e06542ba"
    )

    gemini_pro = load_all_reports(
        "../reports/gemini-pro/reports/ebde051c-6d66-456a-a207-a1c65eceaf40"
    )
    gemini_pro_zero_t = load_all_reports(
        "../reports/gemin-pro-zero-T/reports/1e5457ad-96b5-4bc8-bd6c-bad3eb6deb7a"
    )

    gpt35turbo = load_all_reports(
        "../reports/gpt-3.5-turbo/reports/f76bf17d-3e12-47c5-b879-9cc0c78be989"
    )

    gpt4 = load_all_reports(
        "../reports/gpt-4/reports/76c5bdd4-e893-43d4-b37d-2ade66c20308"
    )
    gpt4_zero_t = load_all_reports(
        "../reports/gpt-4-zero-T/reports/8bbc0722-c86f-4388-9d84-eb8a1a75a72f"
    )
