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


rule human_statistics: 
    input: 
        "src/scripts/analyze_human_data.py"
    output: 
        ["src/tex/output/number_experts.txt", "src/tex/output/total_hours.txt", "src/tex/figures/human_timing.pdf", "src/tex/figures/experience_vs_correctness.pdf","src/tex/output/spearman_experience_score.txt", "src/tex/output/spearman_experience_score_p.txt" ]
        # "src/tex/output/human_questions.csv", "src/tex/output/human_questions.pkl"
    script: 
        "src/scripts/analyze_human_data.py"


rule wordcloud: 
    input: 
        "src/tex/output/questions.pkl"
    output:
        ["src/tex/figures/wordcloud.pdf", "src/tex/figures/flesch_kincaid_reading_ease.pdf", "src/tex/output/flesch_kincaid_reading_ease.txt"]
    script:
        "src/scripts/wordcloud.py"