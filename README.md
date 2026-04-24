# LOCA: Localized Offline Comb Analyzer
**Kaggle Dataset**: [Beehive Cell Classification Dataset](https://www.kaggle.com/datasets/jondelvic/beehive-cell-classification-dataset)
<br>
**MEGA**: [LOCA Dataset](https://mega.nz/folder/NG9njCRC#nRA12_W_CsbHPYU7EzNzrg)

## Cloud Training
**Kaggle Notebook**: [LOCA Training Notebook](https://www.kaggle.com/code/jondelvic/loca-sample-training)
<br>
**Google Colab**: TODO

## Local Training
1. Clone the repo
2. Install `pip install ultralytics`
3. Data Setup: Download the dataset from the Kaggle link above (then place them in the /Dataset directory)
4. Run training
```
from ultralytics import YOLO
model = YOLO('yolo11n.pt)
model.train(data='loca.yaml', epochs=100, imgsz=960, batch 8)
```

## Class Mapping
| ID | Class Name | Source / Mapping |
| :--- | :--- | :--- |
| 0 | `egg` | DeepBee `eggs` |
| 1 | `larva` | DeepBee `larves` |
| 2 | `capped_brood` | DeepBee `capped` |
| 3 | `pollen` | DeepBee `pollen` |
| 4 | `uncapped_honey` | DeepBee `nectar` + Khokthong `uncap` |
| 5 | `capped_honey` | Khokthong `cap` |
| 6 | `other` | DeepBee `dontcare` + Khokthong `other` |

## Data Sources
https://github.com/avsthiago/deepbee-source <br>
https://github.com/Panpakornk/honeybeehiveclassification <br>
https://github.com/jakebruce/CombCount