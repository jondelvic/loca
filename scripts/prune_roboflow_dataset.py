# This was the code use to create a small manual intervention tool in identifying the non-augmented dataset from Khokthong et al.
# Will not work in this directory.

import os
import cv2
import shutil
from collections import defaultdict

ROOT_DIR = ".." 
OUT_DIR = "../../Khokthong_Manual_Final"

def manual_picker():
    img_dir = os.path.join(ROOT_DIR, 'train', 'images')
    lbl_dir = os.path.join(ROOT_DIR, 'train', 'labels')
    
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    clusters = defaultdict(list)
    for f in files:
        clusters[f.split('.rf.')[0]].append(f)

    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels'), exist_ok=True)

    print(f"Total clusters to review: {len(clusters)}")
    print("Instructions: Press '1', '2', or '3' to pick the original. Press 'q' to quit.")

    for base_name, group in clusters.items():
        display_img = []
        for f in group:
            img = cv2.imread(os.path.join(img_dir, f))
            img = cv2.resize(img, (400, 400)) 
            cv2.putText(img, f"Press {group.index(f)+1}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            display_img.append(img)
        
        combined = cv2.hconcat(display_img)
        cv2.imshow(f"Pick the Original: {base_name}", combined)
        
        key = cv2.waitKey(0)
        selected_idx = -1
        
        if key == ord('1'): selected_idx = 0
        elif key == ord('2'): selected_idx = 1
        elif key == ord('3'): selected_idx = 2
        elif key == ord('q'): break

        if selected_idx != -1:
            selected_name = group[selected_idx]
            # Copy Image
            shutil.copy(os.path.join(img_dir, selected_name), os.path.join(OUT_DIR, 'images', selected_name))
            # Copy Label
            lbl = os.path.splitext(selected_name)[0] + ".txt"
            if os.path.exists(os.path.join(lbl_dir, lbl)):
                shutil.copy(os.path.join(lbl_dir, lbl), os.path.join(OUT_DIR, 'labels', lbl))
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_picker()