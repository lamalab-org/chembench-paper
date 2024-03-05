rule numberofquestions:
    input: 
        "src/scripts/compute_basic_statistics.py"
    output: 
        # for caching perhaps generate directory https://github.com/showyourwork/showyourwork/issues/119
        ["src/tex/output/manually_generated.txt", "src/tex/output/total_number_of_questions.txt", "src/tex/output/automatically_generated.txt"]
    script: 
        "src/scripts/compute_basic_statistics.py"
