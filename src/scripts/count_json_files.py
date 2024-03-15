import os
from glob import glob
from utils import obtain_chembench_repo
from paths import output as output_path


def count_json_files_in_directory(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return len(json_files)


def process_directory(directory):
    total_json_files = 0
    for sub_dir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, sub_dir)
        if os.path.isdir(sub_dir_path) and sub_dir != 'smarts':
            if sub_dir == 'safety':
                total_json_files += process_safety_directory(sub_dir_path)
            elif sub_dir == 'icho':
                num_json_files = count_json_files_in_directory(sub_dir_path)
                total_json_files += num_json_files
                with open(output_path / "json_file_counts.txt", 'a') as f:
                    f.write(f'{sub_dir} has {num_json_files} .json files\n')
            else:
                num_json_files = count_json_files_in_directory(sub_dir_path)
                total_json_files += num_json_files
                with open(output_path / "json_file_counts.txt", 'a') as f:
                    f.write(f'{sub_dir} has {num_json_files} .json files\n')

    with open(output_path / "json_file_counts.txt", 'a') as f:
        f.write(f'Total number of JSON files in {directory}: {total_json_files}\n')


def process_safety_directory(safety_dir):
    total_safety_json_files = 0
    for safety_subdir in os.listdir(safety_dir):
        safety_subdir_path = os.path.join(safety_dir, safety_subdir)
        if os.path.isdir(safety_subdir_path):
            num_json_files = count_json_files_in_directory(safety_subdir_path)
            total_safety_json_files += num_json_files
            with open(output_path / "json_file_counts.txt", 'a') as f:
                f.write(f'{safety_subdir} of safety has {num_json_files} .json files\n')

    return total_safety_json_files


def main():
    chembench_repo = obtain_chembench_repo()
    root_dir = os.path.join(chembench_repo, "data")
    process_directory(root_dir)


if __name__ == '__main__':
    main()