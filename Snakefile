# use hard-coded rules and embeddings to assign topics
rule classify_questions:
    input: 
        "src/scripts/classify_questions.py"
    cache: 
        True
    output:
        "src/tex/data/questions.pkl"
    script: 
        "src/scripts/classify_questions.py"

# output some basic statistics about our corpus of questions
rule question_statistics:
    input: 
        "src/tex/data/questions.pkl"
    output: 
        ["src/tex/output/total_number_of_questions.txt", "src/tex/output/automatically_generated.txt", "src/tex/output/manually_generated.txt",   "src/tex/output/num_h_statements.txt", "src/tex/output/num_pictograms.txt", "src/tex/output/num_tiny_questions.txt", "src/tex/output/non_mcq_questions.txt","src/tex/output/mcq_questions.txt"]
    script: 
        "src/scripts/compute_basic_statistics.py"


# collect the raw human scores and aggregate them 
rule collect_human_scores: 
    input: 
        "src/scripts/collect_human_scores.py"
    output: 
        ["src/tex/output/num_humans_with_more_than_100_scores.txt", "src/tex/output/num_humans_with_204_scores.txt", directory("src/tex/output/human_scores")]
    script: 
        "src/scripts/collect_human_scores.py"

# collect the raw model scores and aggregate them 
rule model_statistics: 
    input: 
        "src/scripts/collect_model_scores.py"
    cache: 
        True
    output:
        directory("src/tex/output/overall_model_scores")
    script: 
        "src/scripts/collect_model_scores.py"

# collect model scores on the questions that at least 4 humans answered 
rule model_human_statistics: 
    input: 
        "src/scripts/collect_model_scores_human_subset.py"
    cache: 
        True
    output:
        directory("src/tex/output/human_subset_model_scores")
    script: 
        "src/scripts/collect_model_scores_human_subset.py"

# plot the  number of questions in different topics
rule question_plots:
    input: 
        "src/tex/data/questions.pkl"
    output: 
        "src/tex/figures/question_count_barplot.pdf"
    script: 
        "src/scripts/plot_statistics.py" 

# output basic statistics about the human baseline
rule human_statistics: 
    input: 
        "src/scripts/analyze_human_data.py"
    output: 
        ["src/tex/output/number_experts.txt", "src/tex/output/total_hours.txt", "src/tex/figures/human_timing.pdf", "src/tex/figures/experience_vs_correctness.pdf","src/tex/output/spearman_experience_score.txt", "src/tex/output/spearman_experience_score_p.txt", "src/tex/output/num_human_phd.txt", "src/tex/output/num_human_master.txt", "src/tex/output/num_human_bachelor.txt", "src/tex/output/num_human_highschool.txt", "src/tex/output/num_human_postdoc.txt", "src/tex/output/num_users_with_education_info.txt" ]
    script: 
        "src/scripts/analyze_human_data.py"

# create a wordcloud based on the question bank
rule wordcloud: 
    input: 
        "src/tex/data/questions.pkl"
    output:
        ["src/tex/figures/wordcloud.pdf", "src/tex/figures/flesch_kincaid_reading_ease.pdf", "src/tex/output/flesch_kincaid_reading_ease.txt"]
    script:
        "src/scripts/wordcloud.py"


# loads reports for every model 
# scores them again (to ensure consistency)
# aligns with the questions in the current question bank (to ensure consistency)
# also adds the previously obtained topics
rule get_model_performance_dicts: 
    input: 
        "src/tex/data/questions.pkl"
    output: 
        "src/data/model_score_dicts.pkl"
    script: 
        "src/scripts/get_model_performance_dicts.py"


# plots the performance in various ways
rule analyze_model_reports: 
    input: 
        "src/data/model_score_dicts.pkl"
    output: 
        ["src/tex/output/intersection.txt", "src/tex/figures/all_questions_models_completely_correct_radar.pdf", "src/tex/figures/all_questions_models_requires_calculation_radar.pdf"]
    script:
        "src/scripts/analyze_model_reports.py"