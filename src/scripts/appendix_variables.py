import json
from paths import data, output


gemma_9B_performance = json.load(
    open(data / "chem-bench-main" / "reports" / "gemma-2-9b-it" / "gemma-2-9b-it.json")
)["fraction_correct"]

llama_405B_performance = json.load(
    open(data / "chem-bench-main" / "reports" / "llama3.1-405b-instruct" / "llama3.1-405b-instruct.json")
)["fraction_correct"]


diff_between_llama_405B_and_gemma_9B = llama_405B_performance - gemma_9B_performance
# round to the nearest integer
diff_between_llama_405B_and_gemma_9B = round(diff_between_llama_405B_and_gemma_9B*100)

# save to a text file
# make directory if it does not exist
(output / "trends_section_variables"/ "gemma_9B.txt").parent.mkdir(parents=True, exist_ok=True)
with open(output / "trends_section_variables"/ "gemma_9B.txt", "w") as f:
    f.write(f"{gemma_9B_performance:.2f}" + "\\endinput")

(output / "trends_section_variables"/ "diff_between_llama_405B_and_gemma_9B.txt").parent.mkdir(parents=True, exist_ok=True)
with open(output / "trends_section_variables"/ "diff_between_llama_405B_and_gemma_9B.txt", "w") as f:
    f.write(f"{diff_between_llama_405B_and_gemma_9B}" + "\\endinput")
