# in several cases, we store intermediate results in pkl files
# while pickle files are not ideal for data sharing
# we believe that they are very convenient for caching
# and since we ship the environment, there should also not be too big
# problems with opening those files


# use hard-coded rules and embeddings to assign topics
rule classify_questions:
    input:
        "src/scripts/classify_questions.py",
    cache: True
    output:
        "src/data/questions.pkl",
    script:
        "src/scripts/classify_questions.py"


# output some basic statistics about our corpus of questions
rule question_statistics:
    input:
        "src/data/questions.pkl",
    output:
        [
            "src/tex/output/total_number_of_questions.txt",
            "src/tex/output/automatically_generated.txt",
            "src/tex/output/manually_generated.txt",
            "src/tex/output/num_h_statements.txt",
            "src/tex/output/num_pictograms.txt",
            "src/tex/output/num_tiny_questions.txt",
            "src/tex/output/non_mcq_questions.txt",
            "src/tex/output/mcq_questions.txt",
        ],
    script:
        "src/scripts/compute_basic_statistics.py"


# collect the raw human scores and aggregate them
rule collect_human_scores:
    input:
        "src/scripts/collect_human_scores.py",
    output:
        [
            "src/tex/output/num_humans_with_more_than_100_scores.txt",
            "src/tex/output/num_humans_with_204_scores.txt",
            directory("src/tex/output/human_scores"),
        ],
    script:
        "src/scripts/collect_human_scores.py"


# collect the raw model scores and aggregate them
rule model_statistics:
    input:
        "src/scripts/collect_model_scores.py",
    cache: True
    output:
        directory("src/tex/output/overall_model_scores"),
    script:
        "src/scripts/collect_model_scores.py"


# collect model scores on the questions that at least 4 humans answered
rule model_human_statistics:
    input:
        "src/scripts/collect_model_scores_human_subset.py",
    cache: True
    output:
        directory("src/tex/output/human_subset_model_scores"),
    script:
        "src/scripts/collect_model_scores_human_subset.py"


# plot the  number of questions in different topics
rule question_plots:
    input:
        "src/data/questions.pkl",
    output:
        [
            "src/tex/figures/question_count_barplot.pdf",
            "src/tex/output/num_topics.txt",
            "src/tex/figures/question_count_barplot_mcq_vs_general.pdf",
        ],
    script:
        "src/scripts/plot_statistics.py"


# output basic statistics about the human baseline
rule human_statistics:
    input:
        "src/scripts/analyze_human_data.py",
    output:
        [
            "src/tex/output/number_experts.txt",
            "src/tex/output/total_hours.txt",
            "src/tex/figures/human_timing.pdf",
            "src/tex/figures/experience_vs_correctness.pdf",
            "src/tex/output/spearman_experience_score.txt",
            "src/tex/output/spearman_experience_score_p.txt",
            "src/tex/output/num_human_phd.txt",
            "src/tex/output/num_human_master.txt",
            "src/tex/output/num_human_bachelor.txt",
            "src/tex/output/num_human_highschool.txt",
            "src/tex/output/num_human_postdoc.txt",
            "src/tex/output/num_users_with_education_info.txt",
        ],
    script:
        "src/scripts/analyze_human_data.py"


# create a wordcloud based on the question bank
rule wordcloud:
    input:
        "src/data/questions.pkl",
    output:
        [
            "src/tex/figures/wordcloud.pdf",
            "src/tex/figures/flesch_kincaid_reading_ease.pdf",
            "src/tex/output/flesch_kincaid_reading_ease.txt",
            "src/tex/output/reading_ease.pkl",
        ],
    script:
        "src/scripts/wordcloud.py"


# loads reports for every model
# scores them again (to ensure consistency)
# aligns with the questions in the current question bank (to ensure consistency)
# also adds the previously obtained topics
rule get_model_performance_dicts:
    input:
        "src/data/questions.pkl",
    cache: True
    output:
        "src/data/model_score_dicts.pkl",
    script:
        "src/scripts/get_model_performance_dicts.py"


# similar to the script above, but for the human scores
# for this, we start by treating each human that answered more than 100
# questions as "model"
# we can subsequently groupby topic and then average those topic-aggregated scores
# for further analysis


rule get_human_performance_dicts:
    input:
        "src/data/questions.pkl",
    cache: True
    output:
        "src/data/humans_as_models_scores.pkl",
    script:
        "src/scripts/get_human_performance_dicts.py"


# plots the performance in various ways
rule analyze_model_reports:
    input:
        ["src/data/model_score_dicts.pkl", "src/data/humans_as_models_scores.pkl"],
    output:
        ["src/tex/figures/all_questions_models_completely_correct_radar_overall.pdf"],  #, "src/tex/figures/all_questions_models_requires_calculation_radar_overall.pdf", "src/tex/figures/all_questions_models_completely_correct_radar_human_aligned.pdf", "src/tex/figures/all_questions_models_requires_calculation_radar_human_aligned.pdf"]
    script:
        "src/scripts/analyze_model_reports.py"


# plot the overall performance
rule plot_overview_performance:
    input:
        rules.model_statistics.output
        + rules.collect_human_scores.output
        + rules.model_human_statistics.output,
    output:
        [
            "src/tex/figures/overall_performance.pdf",
            "src/tex/figures/human_subset_performance.pdf",
        ],
    script:
        "src/scripts/plot_overview_performance_plot.py"


# save txt files with the average number of completely correctly
# answered questions per subset and model
rule analyze_performance_per_source:
    input:
        "src/data/model_score_dicts.pkl",
    output:
        [
            directory("src/tex/output/subset_scores"),
            directory("src/tex/output/human_subset_scores"),
            "src/tex/figures/performance_per_topic.pdf",
            "src/tex/output/human_subset_scores/is_number_nmr_peaks.txt",
            "src/tex/output/subset_scores/is_number_nmr_peaks_gpt4.txt",
            "src/tex/output/subset_scores/is_number_of_isomers_gpt4.txt",
                        "src/tex/output/subset_scores/is_gfk_claude3.txt",
            "src/tex/output/subset_scores/is_gfk_gpt4.txt",
            "src/tex/output/human_subset_scores/is_number_of_isomers.txt",
            "src/tex/output/human_subset_scores/is_gfk.txt"
        ],
    script:
        "src/scripts/analyze_performance_per_source.py"


# plot the confidence score distributions
rule plot_confidence_score_distributions:
    input:
        "src/scripts/plot_confidence_score_distributions.py",
    output:
        "src/tex/figures/confidence_score_distributions.pdf",
    script:
        "src/scripts/plot_confidence_score_distributions.py"


rule obtain_embeddings:
    input:
        "src/scripts/embed_questions.py",
    output:
        "src/data/embeddings.npy",
    cache: True
    script:
        "src/scripts/embed_questions.py"


rule plot_embeddings:
    input:
        "src/data/embeddings.npy",
    output:
        "src/tex/figures/question_diversity.pdf",
    script:
        "src/scripts/plot_question_diversity.py"


# count the number of questions per directory
rule count_json_files:
    input:
        "src/scripts/count_json_files.py",
    output:
        [
            directory("src/tex/output/question_count_per_dir"),
            "src/tex/output/question_count_per_dir/json_file_counts_analytical_chemistry.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_combustion_engineering.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_func_mats_and_nanomats.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_Gen_Chem_MCA.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_ac_faessler_tum.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_pum_tum.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_icho.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_materials_synthesis.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_organic_reactivity.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_periodic_table_properties.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_polymer_chemistry.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_reactive_groups.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_oup.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_safety.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_materials_compatibility.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_chem_chem_comp.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_blac_gfk.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_oxidation_states.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_electron_counts.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_number_of_isomers.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_point_group.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_smiles_to_name.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_name_to_smiles.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_pictograms.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_h_statements.txt",
            "src/tex/output/question_count_per_dir/json_file_counts_dai.txt",
        ],
    script: 
        "src/scripts/count_json_files.py"


# joint analysis of the confidence scores of the models and the performance on different tasks
rule performance_confidence_correlation:
    input:
        ["src/data/model_score_dicts.pkl"],
    output:
        [
            "src/tex/figures/confidence_vs_performance_overall.pdf",
            "src/tex/figures/confidence_vs_performance_human_aligned.pdf",
            directory("src/tex/output/model_confidence_performance"),
            "src/tex/output/model_confidence_performance/claude3_is_pictograms_average_confidence_incorrect_overall.txt",
            "src/tex/output/model_confidence_performance/gpt4_is_pictograms_average_confidence_incorrect_overall.txt",
            "src/tex/output/model_confidence_performance/gpt4_is_pictograms_num_incorrect_overall.txt",
            "src/tex/output/model_confidence_performance/gpt4_is_pictograms_num_correct_overall.txt",
            "src/tex/output/model_confidence_performance/gpt4_is_pictograms_average_confidence_correct_overall.txt",
            "src/tex/output/model_confidence_performance/claude3_is_pictograms_average_confidence_correct_overall.txt"
        ],
    script:
        "src/scripts/joint_analysis_confidence_performance.py"


# correlate reading ease with model performance
rule reading_ease_correlation:
    input:
        ["src/tex/data/model_score_dicts.pkl", "src/tex/output/reading_ease.pkl"],
    output:
        "src/tex/figures/reading_ease_vs_model_performance.pdf",
    script:
        "src/scripts/reading_ease_vs_model_performance.py"


# parallel coordinates plot
rule parallel_coordinates:
    input:
        "src/data/model_score_dicts.pkl",
    output:
        [
            "src/tex/figures/parallel_coordinates_overall.pdf",
            "src/tex/figures/parallel_coordinates_tiny.pdf",
        ],
    script:
        "src/scripts/make_parallel_coordinates_plot.py"


rule plot_human_score_distribution:
    input:
        rules.collect_human_scores.output,
    output:
        "src/tex/figures/human_score_distribution.pdf",
    script:
        "src/scripts/plot_human_score_distribution.py"


rule molecule_score_correlation:
    input:
        "src/data/model_score_dicts.pkl",
    output:
        "src/tex/figures/correlation_plot_is_number_nmr_peaks_num_atoms.pdf",
    script:
        "src/scripts/correlate_with_molecule_features.py"
