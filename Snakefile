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
        "src/data/name_to_dir_map.pkl"]
    output:
        "src/data/model_score_dicts.pkl"
    script:
        "src/scripts/get_model_performance_dicts.py"

# obtain human scores summarized
rule get_human_performance_dicts:
    input:
        "src/data/questions.pkl",
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
        "src/data/humans_as_models_scores_no_tools.pkl"]
    output: 
            [
            directory("src/tex/output/subset_scores"),
            directory("src/tex/output/human_subset_scores"),
            "src/tex/figures/performance_per_topic.pdf",
            "src/tex/output/human_subset_scores/is_number_nmr_peaks.txt",
            "src/tex/output/human_subset_scores/is_number_of_isomers.txt",
            "src/tex/output/human_subset_scores/is_gfk.txt"
        ],
    script: 
        "src/scripts/analyze_performance_per_source.py"