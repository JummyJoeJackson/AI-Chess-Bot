import os

def split_jsonl_folder(src_folder, out_folder, out_mb=25):
    os.makedirs(out_folder, exist_ok=True)
    src_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.jsonl')])

    for i, src_file in enumerate(src_files):
        print(f'Processing file {i+1}/{len(src_files)}: {src_file}')
        src_path = os.path.join(src_folder, src_file)
        part_count = 1
        current_size = 0
        current_lines = []

        with open(src_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_size = len(line.encode('utf-8'))
                if current_size + line_size > out_mb * 1024 * 1024 and current_lines:
                    out_path = os.path.join(out_folder, f"{os.path.splitext(src_file)[0]}_part_{part_count}.jsonl")
                    with open(out_path, 'w', encoding='utf-8') as out_file:
                        out_file.writelines(current_lines)
                    part_count += 1
                    current_lines = []
                    current_size = 0

                current_lines.append(line)
                current_size += line_size

            if current_lines:
                out_path = os.path.join(out_folder, f"{os.path.splitext(src_file)[0]}_part_{part_count}.jsonl")
                with open(out_path, 'w', encoding='utf-8') as out_file:
                    out_file.writelines(current_lines)

    print("All files processed.")

# Used for data cleaning from original 70gb LiChess .jsonl file
split_jsonl_folder(r"~\AI-Chess-Bot\Training\evals", r"~\AI-Chess-Bot")
