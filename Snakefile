rule numberofquestions:
    output: 
        "src/tex/output/total_number_of_questions.txt"
    script: 
        "src/scripts/compute_basic_statistics.py"