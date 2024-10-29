import os
from utils import obtain_chembench_repo
from paths import output as output
from loguru import logger
from glob import glob

output_path = output / "question_count_per_dir"
os.makedirs(output_path, exist_ok=True)


def count_json_files_in_directory(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return len(json_files)


def process_directory(directory):
    total_json_files = 0
    for sub_dir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, sub_dir)
        if os.path.isdir(sub_dir_path) and sub_dir != "smarts":
            if sub_dir == "safety":
                total_json_files += process_safety_directory(sub_dir_path)
            elif sub_dir == "icho":
                num_json_files = count_json_files_in_directory(sub_dir_path)
                total_json_files += num_json_files
                with open(output_path / f"json_file_counts_{sub_dir}.txt", "w") as f:
                    f.write(f"{num_json_files}" + "\endinput")

                logger.info(f"Number of json files in {sub_dir}: {num_json_files}")
            else:
                num_json_files = count_json_files_in_directory(sub_dir_path)
                total_json_files += num_json_files
                with open(output_path / f"json_file_counts_{sub_dir}.txt", "w") as f:
                    f.write(f"{num_json_files}" + "\endinput")

                logger.info(f"Number of json files in {sub_dir}: {num_json_files}")

    with open(output_path / "json_file_counts.txt", "a") as f:
        f.write(f"{total_json_files}" + "\endinput")

    logger.info(f"Total number of json files: {total_json_files}")


def process_safety_directory(safety_dir):
    total_safety_json_files = 0
    for safety_subdir in os.listdir(safety_dir):
        safety_subdir_path = os.path.join(safety_dir, safety_subdir)
        if os.path.isdir(safety_subdir_path):
            num_json_files = count_json_files_in_directory(safety_subdir_path)
            total_safety_json_files += num_json_files
            with open(output_path / f"json_file_counts_{safety_subdir}.txt", "w") as f:
                f.write(f"{num_json_files}" + "\endinput")

            logger.info(f"Number of json files in {safety_subdir}: {num_json_files}")
    with open(output_path / "json_file_counts_safety.txt", "w") as f:
        f.write(f"{total_safety_json_files}" + "\endinput")
    return total_safety_json_files


def main():
    chembench_repo = obtain_chembench_repo()
    root_dir = os.path.join(chembench_repo, "data")
    process_directory(root_dir)

    num_dai = len(
        glob(
            os.path.join(chembench_repo, "data", "safety", "pubchem_data", "DAI*.json")
        )
    )
    with open(output_path / "json_file_counts_dai.txt", "w") as f:
        f.write(f"{num_dai}" + "\endinput")

    logger.info(f"Number of json files in DAI: {num_dai}")

    num_hstatements = len(
        glob(
            os.path.join(
                chembench_repo, "data", "safety", "pubchem_data", "h_statements*.json"
            )
        )
    )
    with open(output_path / "json_file_counts_h_statements.txt", "w") as f:
        f.write(f"{num_hstatements}" + "\endinput")

    logger.info(f"Number of json files in h_statements: {num_hstatements}")

    num_pictograms = len(
        glob(
            os.path.join(
                chembench_repo, "data", "safety", "pubchem_data", "pictograms*.json"
            )
        )
    )

    with open(output_path / "json_file_counts_pictograms.txt", "w") as f:
        f.write(f"{num_pictograms}" + "\endinput")

    logger.info(f"Number of json files in pictograms: {num_pictograms}")

    num_preference = len(
        glob(
            os.path.join(
                chembench_repo, "data", "preference", "*.json"
            )
        )
    )

    with open(output_path / "json_file_counts_preference.txt", "w") as f:
        f.write(f"{num_preference}" + "\endinput")

    logger.info(f"Number of json files about chemical preference: {num_preference}")

if __name__ == "__main__":
    main()
