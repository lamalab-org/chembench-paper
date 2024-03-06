rule question_statistics:
    input: 
        "src/scripts/compute_basic_statistics.py"
    output: 
        # for caching perhaps generate directory https://github.com/showyourwork/showyourwork/issues/119
        ["src/tex/output/total_number_of_questions.txt", "src/tex/output/automatically_generated.txt", "src/tex/output/manually_generated.txt", "src/tex/output/questions.csv", "src/tex/output/questions.pkl"]
    script: 
        "src/scripts/compute_basic_statistics.py"


rule question_plots:
    input: 
        "src/tex/output/questions.pkl"
    output: 
        "src/tex/figures/question_count_barplot.pdf"
    script: 
        "src/scripts/plot_statistics.py" 