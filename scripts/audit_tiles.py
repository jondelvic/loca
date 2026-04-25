import os
from pathlib import Path

def audit_tiled_dataset(root_dir):
    root = Path(root_dir)
    print(f"ANALYZING: {root.absolute()}")
    print("-" * 50)

    stats = {}
    mismatches = []
    bad_labels = []
    khokthong_samples = {}

    for path in root.rglob("*"):
        if path.is_dir(): continue
        
        parts = path.parts
        if "images" not in parts and "labels" not in parts: continue
        
        split = "unknown"
        for s in ['train', 'valid', 'test']:
            if s in parts: split = s
            
        source = parts[parts.index('tiled-dataset') + 1]
        key = (source, split)
        
        if key not in stats:
            stats[key] = {'images': 0, 'labels': 0}

        if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            stats[key]['images'] += 1
            if "HoneybeeHive" in source:
                original_name = path.name.split('_y')[0]
                khokthong_samples[original_name] = khokthong_samples.get(original_name, 0) + 1
        
        elif path.suffix == '.txt':
            stats[key]['labels'] += 1
            with open(path, 'r') as f:
                for line in f:
                    vals = line.split()
                    if not vals: continue
                    cls_id = int(vals[0])
                    coords = list(map(float, vals[1:]))
                    
                    if not (0 <= cls_id <= 6):
                        bad_labels.append(f"Invalid ID {cls_id} in {path.name}")
                    
                    for c in coords:
                        if not (0.0 <= c <= 1.0):
                            bad_labels.append(f"OOB Coord {c} in {path.name}")

    print(f"{'Source':<30} | {'Split':<10} | {'Images':<8} | {'Labels':<8} | {'Status'}")
    print("-" * 80)
    
    for (source, split), counts in sorted(stats.items()):
        img, lbl = counts['images'], counts['labels']
        status = "MATCH" if img == lbl else "MISMATCH"
        if img != lbl: mismatches.append(f"{source}/{split}")
        print(f"{source[:30]:<30} | {split:<10} | {img:<8} | {lbl:<8} | {status}")

    print("-" * 80)
    
    # Check Khokthong Recovery
    avg_tiles = sum(khokthong_samples.values()) / len(khokthong_samples) if khokthong_samples else 0
    print(f"Khokthong Recovery: Average {avg_tiles:.2f} tiles per 960x960 image.")
    if avg_tiles > 1.1:
        print("SUCCESS: Multi-tiling confirmed for small frames.")
    else:
        print("WARNING: Khokthong images still appearing as single tiles.")

    if bad_labels:
        print(f"\nSEMANTIC ERRORS: Found {len(bad_labels)} label issues (Check coordinate clamping).")
    else:
        print("\nSEMANTIC PASS: All Class IDs (0-6) and coordinates are valid.")

if __name__ == "__main__":
    audit_tiled_dataset('../tiled-dataset')