# 👄 Lip Reading Assistant — Visual Speech Recognition

> Predicting spoken words from lip movement alone — no audio input — using a CNN trained on the GRID corpus.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square)

**Final Year Project · Pune Institute of Computer Technology · Aug 2023 – Apr 2024**

---

## 🎯 Results

| Metric | Value |
|---|---|
| Word-level classification accuracy | **~80%** |
| Dataset | GRID Corpus — 33,000 videos, 34 speakers |
| Training convergence speedup | **~35%** via hyperparameter tuning |
| Audio dependency | **None** — visual input only |

---

## ⚙️ How it works

```
Video input
    │
    ▼
Lip region extraction (OpenCV)
    │
    ▼
Frame sampling @ 25 fps
    │
    ▼
Pixel normalisation + tensor conversion
    │
    ▼
Noise filtering
    │
    ▼
CNN inference → predicted word
```

---

## 📁 Project structure

```
lip-reading-assistant/
│
├── preprocessing/
│   ├── lip_extractor.py      # Lip region detection and cropping
│   ├── frame_sampler.py      # Uniform frame sampling at 25 fps
│   └── tensor_pipeline.py    # Normalisation + tensor conversion
│
├── model/
│   ├── cnn_architecture.py   # CNN model definition
│   ├── train.py              # Training with LR scheduling
│   └── evaluate.py           # Accuracy + confusion matrix
│
├── notebooks/
│   └── experiment_log.ipynb  # Hyperparameter tuning experiments
│
└── README.md
```

---

## ▶️ How to run

```bash
git clone https://github.com/AmeyRathi12/lip-reading-assistant
cd lip-reading-assistant
pip install -r requirements.txt
python preprocessing/tensor_pipeline.py
python model/train.py
python model/evaluate.py
```

Dataset: Download [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) and place under `data/grid_corpus/`.

---

## 💡 Key design decisions

- **Modular pipeline:** Each preprocessing stage is independent — swap dataset without rewriting inference logic
- **LR scheduling:** Reduced convergence time by ~35%
- **Zero audio:** Purely visual — applicable to hearing-impaired assistive technology use cases
