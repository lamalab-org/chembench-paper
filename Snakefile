# in several cases, we store intermediate results in pkl files 
# while pickle files are not ideal for data sharing 
# we believe that they are very convenient for caching 
# and since we ship the environment, there should also not be too big 
# problems with opening those files


# use hard-coded rules and embeddings to assign topics
rule classify_questions:
    input: 
        "src/scripts/classify_questions.py"
    cache: 
        True
    output:
        "src/data/questions.pkl"
    script: 
        "src/scripts/classify_questions.py"

# output some basic statistics about our corpus of questions
rule question_statistics:
    input: 
        "src/data/questions.pkl"
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
        "src/data/questions.pkl"
    output: 
        ["src/tex/figures/question_count_barplot.pdf", "src/tex/output/num_topics.txt", "src/tex/figures/question_count_barplot_mcq_vs_general.pdf"]
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
        "src/data/questions.pkl"
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
        "src/data/questions.pkl"
    cache: 
        True
    output: 
        "src/data/model_score_dicts.pkl"
    script: 
        "src/scripts/get_model_performance_dicts.py"

# similar to the script above, but for the human scores
# for this, we start by treating each human that answered more than 100 
# questions as "model"
# we can subsequently groupby topic and then average those topic-aggregated scores 
# for further analysis 

rule get_human_performance_dicts: 
    input: 
        "src/data/questions.pkl"
    cache: 
        True
    output: 
        "src/data/humans_as_models_scores.pkl"
    script:
        "src/scripts/get_human_performance_dicts.py"

# plots the performance in various ways
rule analyze_model_reports: 
    input: 
        ["src/data/model_score_dicts.pkl", "src/data/humans_as_models_scores.pkl"]
    output: 
        ["src/tex/figures/all_questions_models_completely_correct_radar_overall.pdf"] #, "src/tex/figures/all_questions_models_requires_calculation_radar_overall.pdf", "src/tex/figures/all_questions_models_completely_correct_radar_human_aligned.pdf", "src/tex/figures/all_questions_models_requires_calculation_radar_human_aligned.pdf"]
    script:
        "src/scripts/analyze_model_reports.py"

# plot the overall performance 
rule plot_overview_performance: 
    input:
        rules.model_statistics.output + rules.collect_human_scores.output
    output: 
        ['src/tex/figures/overall_performance.pdf', 'src/tex/figures/human_subset_performance.pdf']
    script: 
        "src/scripts/plot_overview_performance_plot.py"


# save txt files with the average number of completely correctly
# answered questions per subset and model 
rule analyze_performance_per_source:
    input: 
       "src/data/model_score_dicts.pkl" 
    output: 
        [directory('src/tex/output/subset_scores'), directory('src/tex/output/human_subset_scores'), 'src/tex/figures/performance_per_topic.pdf', "src/tex/output/human_subset_scores/is_number_nmr_peaks.txt", "src/tex/output/subset_scores/is_number_nmr_peaks_gpt4.txt", "src/tex/output/subset_scores/is_number_of_isomers_gpt4.txt", "src/tex/output/human_subset_scores/is_number_of_isomers.txt"]
    script:
        "src/scripts/analyze_performance_per_source.py"

# plot the confidence score distributions 
rule plot_confidence_score_distributions: 
    input: 
        "src/scripts/plot_confidence_score_distributions.py"
    output: 
        "src/tex/figures/confidence_score_distributions.pdf"
    script: 
        "src/scripts/plot_confidence_score_distributions.py"


rule obtain_embeddings: 
    input: 
        "src/scripts/embed_questions.py"
    output:
        "src/data/embeddings.npy"
    cache:
        True
    script: 
        "src/scripts/embed_questions.py"


rule plot_embeddings: 
    input: 
        "src/data/embeddings.npy"
    output: 
        "src/tex/figures/question_diversity.pdf"
    script: 
        "src/scripts/plot_question_diversity.py"


# count the number of questions per directory 
rule count_json_files: 
    input: 
        "src/scripts/count_json_files.py"
    output: 
        [directory("src/tex/output/question_count_per_dir"),
        output/question_count_per_dir/json_file_counts_analytical_chemistry.txt,
        output/question_count_per_dir/json_file_counts_combustion_engineering.txt,
        output/question_count_per_dir/json_file_counts_func_mats_and_nanomats.txt,
        output/question_count_per_dir/json_file_counts_Gen_Chem_MCA.txt,
        output/question_count_per_dir/json_file_counts_ac_faessler_tum.txt,
        output/question_count_per_dir/json_file_counts_pum_tum.txt,
        output/question_count_per_dir/json_file_counts_icho.txt,
        output/question_count_per_dir/json_file_counts_materials_synthesis.txt,
        output/question_count_per_dir/json_file_counts_organic_reactivity.txt,
        output/question_count_per_dir/json_file_counts_periodic_table_properties.txt,
        output/question_count_per_dir/json_file_counts_polymer_chemistry.txt,
        output/question_count_per_dir/json_file_counts_reactive_groups.txt,
        output/question_count_per_dir/json_file_counts_oup.txt,
        output/question_count_per_dir/json_file_counts_safety.txt,
        output/question_count_per_dir/json_file_counts_materials_compatibility.txt,
        output/question_count_per_dir/json_file_counts_chem_chem_comp.txt,
        output/question_count_per_dir/json_file_counts_blac_gfk.txt,
        output/question_count_per_dir/json_file_counts_oxidation_states.txt,
        output/question_count_per_dir/json_file_counts_electron_counts.txt,
        output/question_count_per_dir/json_file_counts_number_of_isomers.txt,
        output/question_count_per_dir/json_file_counts_point_group.txt,
        output/question_count_per_dir/json_file_counts_smiles_to_name.txt,
        output/question_count_per_dir/json_file_counts_name_to_smiles.txt,
        output/question_count_per_dir/json_file_counts_name_to_smiles.txt,
        output/question_count_per_dir/json_file_counts_pictograms.txt,
        output/question_count_per_dir/json_file_counts_h_statements.txt,
        output/question_count_per_dir/json_file_counts_dai.txt]
        


# joint analysis of the confidence scores of the models and the performance on different tasks 
rule performance_confidence_correlation: 
    input: 
        ["src/data/model_score_dicts.pkl"]
    output: 
        ["src/tex/figures/confidence_vs_performance_overall.pdf", "src/tex/figures/confidence_vs_performance_human_aligned.pdf", directory("src/text/output/model_confidence_performance")]
    script: 
        "src/scripts/joint_analysis_confidence_performance.py"