rule test:
    input: 
        "src/scripts/compute_basic_statistics.py"
    output: 
        # for caching perhaps generate directory https://github.com/showyourwork/showyourwork/issues/119
        "src/tex/output/total_number_of_questions.txt"
    script: 
        "src/scripts/compute_basic_statistics.py"
