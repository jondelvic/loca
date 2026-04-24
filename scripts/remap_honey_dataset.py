import os
from pathlib import Path

splits = ['train', 'valid', 'test']
base_folder = Path("../Dataset/Panpakornk-HoneybeeHiveClassification")

# 0 (uncap) -> 4, 1 (cap) -> 5, 2 (other) -> 6
mapping = {'0': '4', '1': '5', '2': '6'}

def remap_labels():
    for split in splits:
        label_path = base_folder / split / 'labels'
        
        if not label_path.exists(): # skip if folder not found
            continue

        files = list(label_path.glob("*.txt"))
        print(f"Processing {len(files)} files in {split}...")

        for lbl in files:
            new_lines = []
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    old_id = parts[0]
                    if old_id in mapping:
                        parts[0] = mapping[old_id]
                        new_lines.append(" ".join(parts) + "\n")
                    else:
                        new_lines.append(line)

            with open(lbl, 'w') as f:
                f.writelines(new_lines)

if __name__ == "__main__":
    remap_labels()