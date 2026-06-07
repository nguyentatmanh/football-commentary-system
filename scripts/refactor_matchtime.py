import os
import shutil
import re

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matchtime_dir = os.path.join(base_dir, "MatchTime")
    target_base = os.path.join(base_dir, "src", "football_ai", "commentary")

    if not os.path.exists(matchtime_dir):
        print(f"Error: MatchTime directory not found at {matchtime_dir}")
        print("Please make sure you cloned the repository to the root directory as MatchTime.")
        return

    print("=== Restructuring MatchTime into football_ai ===")
    
    # 1. Define mappings
    folders_to_copy = {
        "alignment": os.path.join(target_base, "alignment"),
        "evaluation": os.path.join(target_base, "evaluation"),
        "features": os.path.join(target_base, "features"),
        "models": os.path.join(target_base, "models"),
    }

    files_to_copy = {
        "matchvoice_dataset.py": os.path.join(target_base, "matchvoice_dataset.py"),
        "train.py": os.path.join(target_base, "train.py"),
        "inference.py": os.path.join(target_base, "inference.py"),
        "inference_single_video_CLIP.py": os.path.join(target_base, "inference_single_video_CLIP.py"),
        "soccer_words_llama3.pkl": os.path.join(target_base, "soccer_words_llama3.pkl"),
    }

    # Create directories if they do not exist
    os.makedirs(target_base, exist_ok=True)
    
    # Copy folders
    for src_folder, dst_folder in folders_to_copy.items():
        src_path = os.path.join(matchtime_dir, src_folder)
        if os.path.exists(src_path):
            if os.path.exists(dst_folder):
                print(f"Removing existing destination folder: {dst_folder}")
                shutil.rmtree(dst_folder)
            shutil.copytree(src_path, dst_folder)
            print(f"Copied folder {src_folder} -> {dst_folder}")
            # Ensure __init__.py exists in target subfolders
            init_file = os.path.join(dst_folder, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w", encoding="utf-8") as f:
                    f.write("#\n")
        else:
            print(f"Warning: Source folder {src_path} does not exist.")

    # Copy files
    for src_file, dst_path in files_to_copy.items():
        src_path = os.path.join(matchtime_dir, src_file)
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"Copied file {src_file} -> {dst_path}")
        else:
            print(f"Warning: Source file {src_path} does not exist.")

    # 2. Fix imports in copied Python files
    print("\n=== Fixing imports in copied Python files ===")

    # Import replacement rules
    replacements = [
        # In files under target_base/models/
        (re.compile(r"from models\.Qformer import"), "from football_ai.commentary.models.Qformer import"),
        # In top-level commentary files
        (re.compile(r"from matchvoice_dataset import"), "from football_ai.commentary.matchvoice_dataset import"),
        (re.compile(r"from models\.matchvoice_model import"), "from football_ai.commentary.models.matchvoice_model import"),
        # In alignment files
        (re.compile(r"from matchtime_model import"), "from football_ai.commentary.alignment.matchtime_model import"),
    ]

    def process_file(file_path):
        if not file_path.endswith(".py"):
            return
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        new_content = content
        for pattern, replacement in replacements:
            new_content = pattern.sub(replacement, new_content)

        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated imports in {os.path.relpath(file_path, base_dir)}")

    # Walk through the restructured directory and process all python files
    for root, _, files in os.walk(target_base):
        for file in files:
            process_file(os.path.join(root, file))

    print("\n=== Refactoring completed successfully! ===")
    print("Now you can delete or rename the original 'MatchTime' folder if desired.")

if __name__ == "__main__":
    main()
