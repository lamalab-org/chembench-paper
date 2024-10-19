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
            "src/tex/output/num_h_statements.txt",
            "src/tex/output/num_pictograms.txt",
            "src/tex/output/non_mcq_questions.txt",
            "src/tex/output/mcq_questions.txt",
            "src/tex/output/automatically_generated.txt",
            "src/tex/output/manually_generated.txt",
        ],
    script:
        "src/scripts/compute_basic_statistics.py"


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
            "src/tex/output/number_of_considered_humans.txt",
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
            "src/tex/output/spearman_experience_score_with_tool_p.txt",
            "src/tex/output/spearman_experience_score_with_tool.txt",
            "src/tex/output/spearman_experience_score_without_tool_p.txt",
            "src/tex/output/spearman_experience_score_without_tool.txt",
            "src/data/human_no_tool_answered_questions.txt",
            "src/data/human_tool_answered_questions.txt",
            "src/data/human_answered_questions.txt",
            "src/tex/output/num_human_answered_questions.txt"
        ],
    script:
        "src/scripts/analyze_human_data.py"

# map the model names (based on the yaml file) to the report directory
rule map_model_name_to_report_dir:
    input:
        "src/scripts/make_name_dir_map.py"
    output:
        "src/data/name_to_dir_map.pkl"
    script:
        "src/scripts/make_name_dir_map.py"


# obtain one big dataframe with all model scores
rule model_score_dict:
    input:
        ["src/data/questions.pkl",
        "src/data/name_to_dir_map.pkl",
        "src/data/human_no_tool_answered_questions.txt",
        "src/data/human_tool_answered_questions.txt",
        "src/data/human_answered_questions.txt",]
    cache: True
    output:
        "src/data/model_score_dicts.pkl"
    script:
        "src/scripts/get_model_performance_dicts.py"

# obtain human scores summarized
rule get_human_performance_dicts:
    input:
        ["src/data/questions.pkl",
        "src/data/human_no_tool_answered_questions.txt",
        "src/data/human_tool_answered_questions.txt",
        "src/data/human_answered_questions.txt",
        ]
    output:
        ["src/data/humans_as_models_scores_tools.pkl",
        "src/data/humans_as_models_scores_no_tools.pkl",
        "src/data/humans_as_models_scores_combined.pkl",
        ]
    script:
        "src/scripts/get_human_performance_dicts.py"

# analyze the performance per source
rule performance_per_source:
    input:
        ["src/data/model_score_dicts.pkl",
        "src/data/humans_as_models_scores_combined.pkl"]
    output:
            [
            directory("src/tex/output/subset_scores"),
            directory("src/tex/output/human_subset_scores"),
            "src/tex/figures/performance_per_topic.pdf",
            "src/tex/output/human_subset_scores/is_number_nmr_peaks.txt",
            "src/tex/output/human_subset_scores/is_number_of_isomers.txt",
            "src/tex/output/human_subset_scores/is_gfk.txt",
            "src/tex/output/subset_scores/is_number_nmr_peaks_o1.txt",
            "src/tex/figures/performance_per_topic_tiny.pdf"
        ],
    script:
        "src/scripts/analyze_performance_per_source.py"

# rule reading_ease_vs_model_performance:
#     input:
#         ["src/data/model_score_dicts.pkl",
#         "src/data/questions.pkl"]
#     output:
#         ["src/tex/figures/reading_ease_vs_model_performance.pdf",
#         "src/tex/output/reading_ease.pkl"]
#     script:
#         "src/scripts/reading_ease_vs_model_performance.py"



# plot the overall performance
rule plot_overview_performance:
    input:
        "src/data/model_score_dicts.pkl"
    output:
        [
            "src/tex/figures/overall_performance.pdf",
        ],
    script:
        "src/scripts/plot_overview_performance_plot.py"


# plots the performance in various ways
rule analyze_model_reports:
    input:
        ["src/data/model_score_dicts.pkl", "src/data/humans_as_models_scores_combined.pkl"],
    output:
        ["src/tex/figures/all_questions_models_completely_correct_radar_overall.pdf"],  #, "src/tex/figures/all_questions_models_requires_calculation_radar_overall.pdf", "src/tex/figures/all_questions_models_completely_correct_radar_human_aligned.pdf", "src/tex/figures/all_questions_models_requires_calculation_radar_human_aligned.pdf"]
    script:
        "src/scripts/analyze_model_reports.py"

rule plot_human_score_distribution:
    input:
        rules.human_statistics.output
    output:
        "src/tex/figures/human_score_distribution.pdf",
    script:
        "src/scripts/plot_human_score_distribution.py"


rule molecule_score_correlation:
    input:
        "src/data/model_score_dicts.pkl",
    output:
        ["src/tex/figures/correlation_plot_is_number_nmr_peaks_num_atoms.pdf",
        "src/tex/figures/correlation_plot_is_electron_counts_num_atoms.pdf",
        "src/tex/figures/correlation_plot_is_number_nmr_peaks_complexity.pdf"],
    script:
        "src/scripts/correlate_with_molecule_features.py"


rule model_size_plot:
    input:
        "src/data/model_score_dicts.pkl"
    output:
        "src/tex/figures/model_size_plot.pdf"
    script:
        "src/scripts/performance_vs_model_size.py"


rule performance_tables:
    input:
        "src/data/model_score_dicts.pkl"
    output:
        ["src/tex/output/performance_table_human_subset.tex",
        "src/tex/output/performance_table.tex"]
    script:
        "src/scripts/make_performance_tables.py"

rule question_counts:
    input:
        "src/scripts/count_json_files.py"
    output:
        [directory("src/tex/output/question_count_per_dir"),
        "src/tex/output/question_count_per_dir/json_file_counts_reactive_groups.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_dai.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_h_statements.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_oxidation_states.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_point_group.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_pictograms.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_smiles_to_name.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_number_of_isomers.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_number_of_nmr_peaks.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_electron_counts.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_chem_chem_comp.txt",
        "src/tex/output/question_count_per_dir/json_file_counts_materials_compatibility.txt"
        ]
    script:
        "src/scripts/count_json_files.py"


rule logit_calibration:
    input:
        rules.model_score_dict.output
    output:
        "src/tex/figures/log_probs_calibration_plot_overall_filtered.pdf"
    script:
        "src/scripts/plot_logprobs.py"

rule plot_temperature_impact:
    input:
        "src/data/model_score_dicts.pkl"
    output:
        "src/tex/figures/swarm_plot_combined.pdf"
    script:
        "src/scripts/plot_temperature_diffs.py"
