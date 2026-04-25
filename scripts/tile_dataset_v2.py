import os
import cv2
import shutil
from glob import glob
from tqdm import tqdm

# ==========================================
# CONFIGURATION 
# ==========================================
RAW_SOURCE_BASE = "../dataset"             
DEST_BASE = "../tiled-dataset-v2"          

HB_SOURCE = os.path.join(RAW_SOURCE_BASE, "Panpakornk-HoneybeeHiveClassification")
HB_DEST = os.path.join(DEST_BASE, "Panpakornk-HoneybeeHiveClassification")

# Tiling Parameters 
TILE_SIZE = 640
OVERLAP = 0.2  
STRIDE = int(TILE_SIZE * (1 - OVERLAP))

# Class Mapping: 0->4 (Uncapped), 1->5 (Capped), 2->6 (Other)
CLASS_MAP = {0: 4, 1: 5, 2: 6}

audit_stats = {
    'original_images': 0,
    'tiled_images_generated': 0,
    'empty_tiles_discarded': 0,
    'original_labels': {4: 0, 5: 0, 6: 0},
    'tiled_labels': {4: 0, 5: 0, 6: 0}
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def yolo_to_abs(yolo_box, img_w, img_h):
    x_c, y_c, w, h = yolo_box
    abs_w, abs_h = w * img_w, h * img_h
    x_min = (x_c * img_w) - (abs_w / 2)
    y_min = (y_c * img_h) - (abs_h / 2)
    x_max = x_min + abs_w
    y_max = y_min + abs_h
    return [x_min, y_min, x_max, y_max]

def abs_to_yolo(abs_box, tile_w, tile_h):
    x_min, y_min, x_max, y_max = abs_box
    abs_w = x_max - x_min
    abs_h = y_max - y_min
    x_c = x_min + (abs_w / 2)
    y_c = y_min + (abs_h / 2)
    return [x_c / tile_w, y_c / tile_h, abs_w / tile_w, abs_h / tile_h]

def process_tile(image, bboxes, x_offset, y_offset, tile_name, dest_img_dir, dest_lbl_dir):
    tile = image[y_offset : y_offset + TILE_SIZE, x_offset : x_offset + TILE_SIZE]
    
    h, w = tile.shape[:2]
    if h < TILE_SIZE or w < TILE_SIZE:
        pad_h = TILE_SIZE - h
        pad_w = TILE_SIZE - w
        tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    new_bboxes = []
    
    for bbox in bboxes:
        cls_id, x_min, y_min, x_max, y_max = bbox
        
        ix_min = max(x_min, x_offset)
        iy_min = max(y_min, y_offset)
        ix_max = min(x_max, x_offset + TILE_SIZE)
        iy_max = min(y_max, y_offset + TILE_SIZE)
        
        if ix_min < ix_max and iy_min < iy_max:
            shifted_box = [ix_min - x_offset, iy_min - y_offset, ix_max - x_offset, iy_max - y_offset]
            
            orig_area = (x_max - x_min) * (y_max - y_min)
            new_area = (ix_max - ix_min) * (iy_max - iy_min)
            
            if new_area / orig_area > 0.15: 
                yolo_fmt = abs_to_yolo(shifted_box, TILE_SIZE, TILE_SIZE)
                new_bboxes.append([cls_id] + yolo_fmt)
                # Safely increment count
                if cls_id in audit_stats['tiled_labels']:
                    audit_stats['tiled_labels'][cls_id] += 1
                else:
                    audit_stats['tiled_labels'][cls_id] = 1

    if len(new_bboxes) > 0:
        cv2.imwrite(os.path.join(dest_img_dir, f"{tile_name}.jpg"), tile)
        with open(os.path.join(dest_lbl_dir, f"{tile_name}.txt"), "w") as f:
            for b in new_bboxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        audit_stats['tiled_images_generated'] += 1
    else:
        audit_stats['empty_tiles_discarded'] += 1

# ==========================================
# MAIN EXECUTION
# ==========================================
def tile_honeybee_dataset():
    print("--- Starting 'True Tiling' for HoneybeeHiveClassification ---")
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        img_src_dir = os.path.join(HB_SOURCE, split, 'images')
        lbl_src_dir = os.path.join(HB_SOURCE, split, 'labels')
        
        if not os.path.exists(img_src_dir):
            continue
            
        img_dest_dir = os.path.join(HB_DEST, split, 'images')
        lbl_dest_dir = os.path.join(HB_DEST, split, 'labels')
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)
        
        images = glob(os.path.join(img_src_dir, "*.jpg"))
        
        for img_path in tqdm(images, desc=f"Tiling {split} split"):
            audit_stats['original_images'] += 1
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            lbl_path = os.path.join(lbl_src_dir, basename + ".txt")
            bboxes = []
            
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        
                        # Fix: Handle both Standard BBoxes and Polygon Masks!
                        if len(parts) >= 5:
                            orig_cls = int(parts[0])
                            new_cls = CLASS_MAP.get(orig_cls, orig_cls) 
                            
                            if new_cls in audit_stats['original_labels']:
                                audit_stats['original_labels'][new_cls] += 1
                            else:
                                audit_stats['original_labels'][new_cls] = 1
                            
                            if len(parts) == 5:
                                # It's already a standard YOLO BBox
                                yolo_box = [float(p) for p in parts[1:]]
                            else:
                                # It's a Polygon Mask! Convert to BBox min/max
                                x_coords = [float(p) for p in parts[1::2]]
                                y_coords = [float(p) for p in parts[2::2]]
                                
                                x_c = (min(x_coords) + max(x_coords)) / 2.0
                                y_c = (min(y_coords) + max(y_coords)) / 2.0
                                w = max(x_coords) - min(x_coords)
                                h = max(y_coords) - min(y_coords)
                                yolo_box = [x_c, y_c, w, h]
                                
                            abs_box = yolo_to_abs(yolo_box, img_w, img_h)
                            bboxes.append([new_cls] + abs_box)
            
            for y in range(0, img_h, STRIDE):
                for x in range(0, img_w, STRIDE):
                    tile_name = f"{basename}_tile_x{x}_y{y}"
                    process_tile(img, bboxes, x, y, tile_name, img_dest_dir, lbl_dest_dir)

def print_audit_report():
    print("\n==========================================")
    print("        DATASET V2 AUDIT REPORT           ")
    print("==========================================")
    print(f"Original Honeybee Images Processed : {audit_stats['original_images']}")
    print(f"Tiled Images Generated             : {audit_stats['tiled_images_generated']}")
    print(f"Empty Tiles Discarded (Background) : {audit_stats['empty_tiles_discarded']}")
    print("\n--- Label Retention Audit ---")
    
    classes = ['4 (Uncapped Honey)', '5 (Capped Honey)  ', '6 (Other)         ']
    keys = [4, 5, 6]
    
    for cls_name, key in zip(classes, keys):
        orig = audit_stats['original_labels'].get(key, 0)
        tiled = audit_stats['tiled_labels'].get(key, 0)
        multiplier = (tiled / orig * 100) if orig > 0 else 0
        print(f"Class {cls_name}: Original={orig:<6} | Tiled Output={tiled:<6} | Retention: {multiplier:.1f}%")
        
    print("\nNote: Retention > 100% is mathematically expected because of the 20% window overlap;")
    print("cells residing on the border of a stride will naturally appear in two adjacent tiles.")
    print("==========================================\n")

if __name__ == "__main__":
    os.makedirs(DEST_BASE, exist_ok=True)
    tile_honeybee_dataset()
    print_audit_report()