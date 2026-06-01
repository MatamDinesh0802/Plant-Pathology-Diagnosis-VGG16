# Plant Pathology Diagnosis — VGG16 on Apple Leaves

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-VGG16-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![ONNX](https://img.shields.io/badge/Inference-ONNX-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Detect **4 apple leaf disease classes** — `healthy`, `multiple_diseases`, `rust`, `scab` —
from photographs using a fine-tuned **VGG16** trained on the
[**Plant Pathology 2020** (FGVC7)](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7) dataset.

Training happens in a [**Colab notebook**](notebooks/01_train_vgg16.ipynb)
(free GPU), and the trained model is exported to **ONNX** so the Streamlit
demo runs without TensorFlow installed locally.

Refresh of a college project (May 2023) — original notebook preserved at
[`reports/original_notebook.ipynb`](reports/original_notebook.ipynb).

> The original college project trained InceptionV3 on rice diseases despite the
> repo name. This refresh aligns the implementation with the name: **VGG16** as the
> backbone, **apple Plant Pathology 2020** as the dataset.

---

## 🖥️ Demo

The Streamlit app supports any uploaded JPG/PNG apple leaf photo.

- **🌿 Diagnose** — upload an image, get the predicted disease + 4-way probability bars.
- **🔍 Image analysis** — color histogram, green-dominance stat, mean RGB. Works even without a trained model.
- **📊 Model performance** — accuracy, weighted F1, per-class precision/recall, confusion matrix, training curves.
- **📖 About** — methodology, limitations, citations.

The UI ships with a custom leaf-green theme + injected CSS — no default Streamlit chrome.

---

## 🚀 Run locally

```bash
git clone https://github.com/MatamDinesh0802/Plant-Pathology-Diagnosis-VGG16.git
cd Plant-Pathology-Diagnosis-VGG16

bash setup.sh                          # creates project-local .venv, installs deps
source .venv/bin/activate

streamlit run app/streamlit_app.py     # launch the demo
```

Sign in to the Streamlit demo with **Username: `Admin`** · **Password: `Admin@123`** (static demo credentials).

If `models/vgg16_model.onnx` is present the Predict tab works end-to-end. If not,
the **Image analysis** tab still works (color histogram + stats from any uploaded image).

### To train the model

Open [`notebooks/01_train_vgg16.ipynb`](notebooks/01_train_vgg16.ipynb) in **Google Colab** (GPU runtime):

1. Upload your `kaggle.json` API token.
2. Accept the [competition rules](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules) on Kaggle first.
3. Run all cells — downloads dataset, trains VGG16 (head-only → fine-tune block5), evaluates, exports.
4. Download `plant_pathology_artifacts.zip` and drop the files into `models/` and `reports/`.

---

## 🧪 Methodology

| | |
|---|---|
| **Data** | Plant Pathology 2020 — 1,821 apple leaf images, 4 classes |
| **Input** | 224 × 224 × 3 RGB |
| **Backbone** | VGG16 (ImageNet weights) |
| **Head** | GAP → Dropout(0.3) → Dense(128) ReLU → Dropout(0.3) → Dense(4) softmax |
| **Augmentation** | Random horizontal flip, brightness, contrast |
| **Loss** | sparse categorical cross-entropy |
| **Optimizer** | Adam (lr 1e-4 head-only, then 1e-5 fine-tune block5) |
| **Stopping** | EarlyStopping(patience=6, restore_best=True) on val accuracy |
| **Split** | stratified 70/15/15 train/val/test, seed 42 |
| **Serving** | ONNX Runtime (no TF dep at inference) |

---

## 🗂️ Project structure

```
Plant-Pathology-Diagnosis-VGG16/
├── app/                            # Streamlit demo
│   ├── streamlit_app.py            # main UI
│   ├── .streamlit/config.toml      # leaf-green theme
│   └── assets/style.css            # injected CSS
├── notebooks/
│   └── 01_train_vgg16.ipynb        # Colab training notebook
├── src/plant_pathology/            # modular Python package
│   ├── config.py                   # paths, labels, image params
│   ├── data.py                     # train.csv → DataFrame
│   ├── features.py                 # VGG16 preprocessing (BGR + mean subtract)
│   ├── model.py                    # Keras VGG16 (used by notebook)
│   ├── predict.py                  # ONNX-runtime inference
│   └── evaluate.py                 # prints metrics.json
├── data/
│   ├── raw/                        # Plant Pathology 2020 (gitignored, ~1.5 GB)
│   └── README.md                   # download instructions
├── reports/
│   ├── figures/                    # confusion matrix, training curves, samples
│   ├── metrics.json                # canonical results from the notebook
│   ├── original_notebook.ipynb     # college-project archive (InceptionV3 on rice)
│   ├── original_app.py.archive     # original Flask serving script
│   └── original_report.pdf
├── tests/                          # pytest smoke tests
├── models/                         # trained ONNX
├── requirements.txt
├── setup.sh                        # one-shot venv + install
└── LICENSE
```

---

## ⚠️ Limitations

- Apple-leaf-specific. Predictions on other species or non-leaf images are unreliable.
- Image quality matters — lighting, focus, framing, and reasonable size all affect accuracy.
- The dataset is from a single growing region (Washington state) — generalization unverified.
- This is a research/portfolio demonstration, not a production agronomy tool.

---

## 📚 Citation

```bibtex
@article{thapa2020plant,
  title={The Plant Pathology Challenge 2020 data set to classify foliar disease of apples},
  author={Thapa, Ranjita and Zhang, Kai and Snavely, Noah and Belongie, Serge and Khan, Awais},
  journal={Applications in Plant Sciences},
  volume={8},
  number={9},
  pages={e11390},
  year={2020},
  publisher={Wiley Online Library}
}
```

---

## 👤 Author

**Matam Dinesh Chandra** — [GitHub](https://github.com/MatamDinesh0802) · [Email](mailto:matamdinesh0802@gmail.com)
