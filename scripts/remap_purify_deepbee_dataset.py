# Python code used to exclude honey and class remapping.
# Will not work in this project repo. Only here as reference.

import os
import pandas as pd

LABELS_TRAIN_PATH = "../labels_train.csv"
LABELS_TEST_PATH = "../labels_test.csv"

CLASS_MAPPING = {
    'eggs': 0,
    'larves': 1,
    'capped': 2,
    'pollen': 3,
    'nectar': 4,       # nectar -> uncapped_honey
    'dontcare': 6      # dontcare -> other
}

EXCLUDED_CLASSES = ['honey'] #ignore!!!
# ============================================

def process_deepbee_data(df, label_col):
    df_filtered = df[~df[label_col].isin(EXCLUDED_CLASSES)].copy()
    
    df_filtered['class_id'] = df_filtered[label_col].map(CLASS_MAPPING)
    
    return df_filtered

def parse_labels_train(csv_path):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path, header=None,
                     names=['id', 'x', 'y', 'label', 'filename'])

    df_clean = process_deepbee_data(df, 'label')

    print(f"\n{'='*50}")
    print("PURIFIED TRAINING SET Summary")
    print(f"{'='*50}")
    print(f"Original annotations: {len(df):,}")
    print(f"Annotations after deleting 'honey': {len(df_clean):,}")
    print(f"Unique images remaining: {df_clean['filename'].nunique()}")
    
    print("\nNew Class Distribution (Mapped to YAML IDs):")
    print("-" * 30)
    counts = df_clean['label'].value_counts()
    for label, count in counts.items():
        target_id = CLASS_MAPPING.get(label)
        print(f"  ID {target_id} ({label:12s}): {count:7,}")

    return df_clean

def parse_labels_test(csv_path):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df_clean = process_deepbee_data(df, 'class name')

    print(f"\n{'='*50}")
    print("PURIFIED TEST SET Summary")
    print(f"{'='*50}")
    print(f"Annotations after deleting 'honey': {len(df_clean):,}")

    return df_clean

if __name__ == "__main__":
    train_df = parse_labels_train(LABELS_TRAIN_PATH)
    test_df = parse_labels_test(LABELS_TEST_PATH)