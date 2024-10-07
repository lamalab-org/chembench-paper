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