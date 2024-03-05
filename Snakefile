rule numberofquestions:
    output: 
        "src/tex/output/manually_generated.txt", 
        "src/tex/output/total_number_of_questions.txt", 
        "src/tex/output/automatically_generated.txt"
    script: 
        "src/scripts/compute_basic_statistics.py"

# rule num_experts: 
#     output: 
#         "src/tex/output/number_experts.txt"
#     script: 
#         "src/scripts/compute_human_statistics.py"