# LOCA: Localized Offline Comb Analyzer
**Kaggle Dataset**: 
- [RAW](https://www.kaggle.com/datasets/jondelvic/beehive-cell-classification-dataset)
- [TILED](https://www.kaggle.com/datasets/jondelvic/beehive-cell-classification-dataset-tiled)

**MEGA**: [RAW LOCA Dataset](https://mega.nz/folder/NG9njCRC#nRA12_W_CsbHPYU7EzNzrg)

## Cloud Training
**Kaggle Notebook**: 
- [LOCA Training Phase 1](https://www.kaggle.com/code/jondelvic/loca-initial-training-phase-1)
- [LOCA Training Phase 2](https://www.kaggle.com/code/jondelvic/loca-training-phase-2-initial-optimizations)
- [LOCA Training Phase 3](https://www.kaggle.com/code/jondelvic/loca-training-phase-3-tiled-dataset)
- [LOCA Training Phase 4](https://www.kaggle.com/code/jondelvic/loca-training-phase-4-yolov11s)
- [LOCA Training Phase 5](https://www.kaggle.com/code/jondelvic/loca-training-phase-5-updated-dataset-yolo26) (PENDING)

**Google Colab**: TODO

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