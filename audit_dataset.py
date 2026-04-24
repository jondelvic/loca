import os
import yaml
from pathlib import Path

def audit():
    with open('loca.yaml', 'r') as f: 
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    valid_ids = set(config['names'].keys())
    
    print(f"valid class ids: {valid_ids}\n")

    # collect all image directories based off yaml
    search_dirs = []
    for split in ['train', 'val', 'test']:
        paths = config.get(split, [])
        if isinstance(paths, str): paths = [paths]
        search_dirs.extend(paths)

    stats = {i: 0 for i in valid_ids}
    errors = []
    total_images = 0

    for rel_dir in search_dirs:
        img_dir = base_path / rel_dir
        lbl_dir = Path(str(img_dir).replace('images', 'labels'))
        
        if not img_dir.exists():
            continue # skip missing dir

        print(f"ccurrently scanning: {rel_dir}...")
        
        images = list(img_dir.glob('*.[jJ][pP][gG]')) + list(img_dir.glob('*.[pP][nG][gG]'))
        total_images += len(images)

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            if "CombCount" not in str(img_path) and not lbl_path.exists():
                errors.append(f"MISSING LABEL: {img_path.name}")
                continue
            
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        errors.append(f"EMPTY LABEL: {lbl_path.name}")
                    
                    for line_no, line in enumerate(lines):
                        parts = line.strip().split()
                        if not parts: continue
                        
                        # check valid class ids
                        cls_id = int(parts[0])
                        if cls_id not in valid_ids:
                            errors.append(f"INVALID ID [{cls_id}] in {lbl_path.name} at line {line_no}")
                        else:
                            stats[cls_id] += 1
                        
                        # check if normalized
                        coords = [float(x) for x in parts[1:]]
                        if any(c > 1.0 or c < 0.0 for c in coords):
                            errors.append(f"NOT NORMALIZED in {lbl_path.name}: {coords}")

    print("DATASET AUDIT REPORT")
    print(f"Total Images Scanned: {total_images}")
    print("\nClass Distribution (Instances):")
    for id, name in config['names'].items():
        print(f"  {id} ({name}): {stats[id]}")
    
    if errors:
        print(f"\nFOUND {len(errors)} ERRORS:")
        for e in errors[:10]: # show 10 errors only for now
            print(f"  - {e}")
        if len(errors) > 10: print(f"  ... and {len(errors)-10} more.")
    else:
        print("\nno errors.")

if __name__ == "__main__":
    audit()