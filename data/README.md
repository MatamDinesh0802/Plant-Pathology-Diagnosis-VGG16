# Data

## Plant Pathology 2020 — FGVC7

The model is trained on the [**Plant Pathology 2020**](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)
Kaggle competition dataset — high-resolution photographs of apple leaves
labelled with one of four foliar disease classes.

| | |
|---|---|
| Samples | 1,821 train images + 1,821 test images |
| Classes (4) | `healthy`, `multiple_diseases`, `rust`, `scab` |
| Image format | JPG |
| Total size | ~1.5 GB |

### Download

The easiest path is via Kaggle:

```bash
# Requires a Kaggle API key at ~/.kaggle/kaggle.json
pip install kaggle

# Accept the competition rules once at:
# https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules
kaggle competitions download -c plant-pathology-2020-fgvc7 -p data/raw/
unzip data/raw/plant-pathology-2020-fgvc7.zip -d data/raw/
```

After unzipping you should have:

```
data/raw/
├── images/                # all JPGs
├── train.csv              # image_id + 4 one-hot labels
├── test.csv               # image_id only
└── sample_submission.csv
```

### Citation

> Thapa, R., Zhang, K., Snavely, N., Belongie, S., & Khan, A. (2020).
> The Plant Pathology Challenge 2020 data set to classify foliar disease of apples.
> *Applications in Plant Sciences*, 8(9), e11390.
