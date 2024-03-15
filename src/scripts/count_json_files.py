import os
import glob

def count_json_files_in_directory(directory):
    json_files = glob.glob(os.path.join(directory, '*.json'))
    return len(json_files)

def process_directory(directory):
    total_json_files = 0
    for sub_dir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, sub_dir)
        if os.path.isdir(sub_dir_path) and sub_dir != 'smarts':
            if sub_dir == 'safety':
                total_json_files += process_safety_directory(sub_dir_path)
            else:
                num_json_files = count_json_files_in_directory(sub_dir_path)
                total_json_files += num_json_files
                print(f'{sub_dir} has {num_json_files} .json files')
    
    print(f'Total number of JSON files in {directory}: {total_json_files}')

def process_safety_directory(safety_dir):
    total_safety_json_files = 0
    for safety_subdir in os.listdir(safety_dir):
        safety_subdir_path = os.path.join(safety_dir, safety_subdir)
        if os.path.isdir(safety_subdir_path):
            num_json_files = count_json_files_in_directory(safety_subdir_path)
            total_safety_json_files += num_json_files
            print(f'{safety_subdir} of safety has {num_json_files} .json files')
    
    return total_safety_json_files

def main():
    root_dir = '/home/ta45woj/chem-bench/data'
    process_directory(root_dir)

if __name__ == '__main__':
    main()