import cv2
import os
import glob
import numpy as np
from PIL import Image

def get_tile_origins(img_dim, tile_size, stride):
    origins = list(range(0, img_dim - tile_size + 1, stride))
    if not origins or origins[-1] + tile_size < img_dim:
        origins.append(img_dim - tile_size)
    return origins

def mirror_tiled_dataset(input_root, output_root, tile_size=640, overlap=0.25):
    stride = int(tile_size * (1 - overlap)) 
    
    for root, dirs, files in os.walk(input_root):
        if os.path.basename(root) == 'images':
            rel_path = os.path.relpath(root, input_root)
            out_img_dir = os.path.join(output_root, rel_path)
            out_lbl_dir = out_img_dir.replace('images', 'labels')
            lbl_dir = root.replace('images', 'labels')
            
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)
            
            img_paths = []
            for ext in ['*.jpg', '*.JPG', '*.png', '*.jpeg']:
                img_paths.extend(glob.glob(os.path.join(root, ext)))
            
            print(f"\nMirroring: {rel_path} ({len(img_paths)} images)")

            for img_path in img_paths:
                name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(lbl_dir, f"{name}.txt")
                
                if not os.path.exists(label_path):
                    continue
                
                try:
                    with Image.open(img_path) as temp_img:
                        w_img, h_img = temp_img.size
                except Exception as e:
                    print(f"  ❌ Error reading {img_path}: {e}")
                    continue
                
                x_origins = get_tile_origins(w_img, tile_size, stride)
                y_origins = get_tile_origins(h_img, tile_size, stride)

                tiles_to_process = []
                for y in y_origins:
                    for x in x_origins:
                        tile_name = f"{name}_y{y}_x{x}"
                        if not os.path.exists(os.path.join(out_img_dir, f"{tile_name}.jpg")):
                            tiles_to_process.append((x, y, tile_name))

                if not tiles_to_process:
                    continue 

                print(f"Processing: {name} ({len(tiles_to_process)} new tiles)")
                img = cv2.imread(img_path)
                if img is None:
                    continue

                with open(label_path, 'r') as f:
                    raw_labels = [line.split() for line in f.readlines()]

                for x, y, tile_name in tiles_to_process:
                    new_labels = []
                    for lbl in raw_labels:
                        if len(lbl) < 5: 
                            continue
                        
                        # map only the first 5 values to handle label noise
                        cls_id, x_c, y_c, w_b, h_b = map(float, lbl[:5])
                        
                        # calculate global pixel coordinates
                        abs_xc, abs_yc = x_c * w_img, y_c * h_img
                        abs_w, abs_h = w_b * w_img, h_b * h_img
                        
                        # ff object center is in this tile
                        if x <= abs_xc < x + tile_size and y <= abs_yc < y + tile_size:
                            # calculate relative center in tile-space
                            rel_xc = (abs_xc - x) / tile_size
                            rel_yc = (abs_yc - y) / tile_size
                            
                            # prevent box-spill for valid YOLO format
                            x_min = max(0, rel_xc - (abs_w / (2 * tile_size)))
                            y_min = max(0, rel_yc - (abs_h / (2 * tile_size)))
                            x_max = min(1.0, rel_xc + (abs_w / (2 * tile_size)))
                            y_max = min(1.0, rel_yc + (abs_h / (2 * tile_size)))
                            
                            final_w = x_max - x_min
                            final_h = y_max - y_min
                            final_xc = x_min + (final_w / 2)
                            final_yc = y_min + (final_h / 2)

                            new_labels.append(f"{int(cls_id)} {final_xc:.6f} {final_yc:.6f} {final_w:.6f} {final_h:.6f}")

                    if new_labels:
                        tile = img[y:y+tile_size, x:x+tile_size]
                        cv2.imwrite(os.path.join(out_img_dir, f"{tile_name}.jpg"), tile)
                        with open(os.path.join(out_lbl_dir, f"{tile_name}.txt"), 'w') as f:
                            f.write("\n".join(new_labels))

if __name__ == "__main__":
    mirror_tiled_dataset(input_root='../dataset', output_root='../tiled-dataset')
    print("\nTililing")