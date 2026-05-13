# Classifying Animal Swarms from an Aerial Viewpoint

---

## Overview

**QuAM** (Query Answering Machine) is a 6-class image classifier for aerial and side-view herd imagery. Conservationists capture thousands of frames per day from drones and static cameras — but existing ML models are built for individual animals, not herds. Dense aggregations with occlusion, overlap, and cluttered backgrounds break standard classifiers. QuAM addresses this gap directly.

We ran **eleven experiments** across three iterations, climbing the ladder of feature complexity from color histograms to ResNet50 transfer features. The final model, ResNet50 + SVM (RBF kernel), reaches **93.30% test accuracy** with a tight 6.5-point train→test gap.

<p align="center">
  <img src="" alt="animalHerd.png" width="700"/>
</p>

---

## The Six Classes

| Class | Images (post-cleaning) |
|-------|----------------------|
| Buffalo | 655 |
| Elephant | 957 |
| Hoofed Grazers | 1,465 |
| Musk Ox | 971 |
| Wildebeest | 1,193 |
| Zebra | 629 |
| **Total** | **5,870** |

Class imbalance ratio: **2.3×** (Hoofed Grazers vs Zebra). Handled via `class_weight='balanced'`.

---

## Data Pipeline

```
Scrape          →   Manual Review     →   Standardise       →   Wrangle
DuckDuckGo          Reject individuals,   Crop watermark bar,   Drop blurry (Laplacian < 10),
+ Selenium          cartoons, watermarks  centre-crop,          dHash duplicate check,
12,408 URLs         → 5,926 URLs          resize to 640×640     validate channels
                                          RGB JPEG              → 5,870 images
```

- `scrape_6_groups.py` — Selenium scraper across 6 classes × 7 queries
- `image_cleaner_v2.py` — Blur detection, perceptual hashing, channel validation
- Only 62 images dropped for blurriness (1%); 0 near-duplicates survived hashing

---

## Three Iterations

### Iteration 1 — Color Histograms + KNN (Baseline)
- **Features:** 32-bin RGB histograms → 96-d L2-normalized vector
- **Model:** KNN (best k=1, 5-fold CV)
- **Result:** 45.6% test accuracy — severe train→test gap (54 pts)
- **Finding:** Color alone loses spatial structure; Zebra collapses to grey average in histogram space

### Iteration 2 — Handcrafted Features

| Experiment | Features | Classifier | CV | Test |
|-----------|----------|------------|-----|------|
| 2.1 | HOG (8,100-d) | SVM (RBF) | 47.27% | 44.72% |
| 2.2 | HOG + HSV + color moments (1,821-d) | SVM (RBF) | 55.05% | 55.25% |
| 2.3 | Combined (1,821-d) | MLP (512→256, BN+Dropout) | 57.44% | 56.30% |

- **Finding:** Same features, different classifiers → only +1 pt. The bottleneck is the features, not the model.

### Iteration 3 — ResNet50 Transfer Features

| # | Approach | Features | Classifier | CV | Test |
|---|----------|----------|------------|-----|------|
| 3.1 | CNN from scratch — vanilla | Raw 128×128 | 3 conv → Dense | — | 62.69% |
| 3.2 | CNN from scratch — improved | Raw + augmentation | BN + Dropout + AdamW | 34.34% | 48.01% |
| 3.3 | ResNet50 + KNN | 2,048-d | KNN (k=7) | 88.54% | 87.85% |
| 3.4 | ResNet50 + Random Forest | 2,048-d → PCA 95% | RF (200 trees) | 88.14% | 88.59% |
| **3.5** | **ResNet50 + SVM (RBF)** | **2,048-d** | **SVM (C=10, γ=scale)** | **93.04%** | **93.30%** |

---

## Final Model

**ResNet50 (frozen, ImageNet weights) → GAP → 2,048-d → StandardScaler → SVM (RBF, C=10)**

- **Test accuracy:** 93.30%
- **CV:** 93.04% ± 0.57
- **Train→test gap:** 6.5 pts

### Per-class F1 (test set)

| Class | F1 |
|-------|----|
| Musk Ox | 0.972 |
| Hoofed Grazers | 0.949 |
| Zebra | 0.949 |
| Elephant | 0.945 |
| Wildebeest | 0.923 |
| Buffalo | 0.838 |

Buffalo is the hardest class in every iteration — most errors are Buffalo→Wildebeest, an irreducible perceptual overlap from above.

---

## Central Finding

> **Features matter ~40× more than classifiers.**  
> Changing features (color hist → ResNet): **+42.2 pts**.  
> Changing classifiers (same features, SVM → MLP): **+1.0 pt**.
---
## Tech Stack

`Python` · `TensorFlow/Keras` · `ResNet50` · `scikit-learn` · `OpenCV` · `Selenium` · `pandas` · `matplotlib` · `imagehash`

---

## Authors

**Iroda Ibrohimova** · **Madina Mirzatayeva** · **Amen Ghabara**  
15-288 Machine Learning in a Nutshell · Carnegie Mellon University Qatar · Spring 2026

[iibrohim@andrew.cmu.edu](mailto:iibrohim@andrew.cmu.edu) · [LinkedIn](https://www.linkedin.com/in/iroda-ibrohimova-73098924a)
