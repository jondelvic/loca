# LOCA: Localized Offline Comb Analyzer

## Structure
```text
/LOCA
├── Dataset/
│   ├── DeepBee/        # 1,187 purified images
│   ├── Panpakornk/     # 269 unique base frames
│   └── CombCount/      # 30 generalization images
├── loca.yaml           # YOLO configuration
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
https://github.com/avsthiago/deepbee-source
https://github.com/Panpakornk/honeybeehiveclassification
https://github.com/jakebruce/CombCount