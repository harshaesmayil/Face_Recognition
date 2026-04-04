# Face Recognition: PCA vs. Feature Selection
**A comparative study of Dimensionality Reduction techniques on the ORL Face Dataset.**

## 📌 Project Overview
This project implements a complete face recognition pipeline using the **ORL Database of Faces**. The core objective was to evaluate and compare three different approaches to handling high-dimensional image data before classification with a **k-Nearest Neighbor (k-NN)** algorithm.

### Key Objectives:
* Implement **Feature Extraction** using Principal Component Analysis (PCA).
* Implement **Filter-based Feature Selection** using ANOVA (SelectKBest).
* Implement **Wrapper-based Feature Selection** using Forward Selection and Backward Elimination.
* Analyze the trade-off between computational cost and recognition accuracy.

---

## 📊 Methodologies

### 1. Feature Extraction (PCA)
* **Technique:** Principal Component Analysis.
* **Process:** Flattened 92 × 112 images (10,304 features) were projected into a 100-dimensional subspace.
* **Output:** "Eigenfaces" representing the directions of maximum variance.

### 2. Filter Method (SelectKBest)
* **Technique:** ANOVA F-test (SelectKBest).
* **Process:** Statistically evaluated each pixel's importance and selected the top 100 most informative pixels.

### 3. Wrapper Methods (SFS)
* **Technique:** Sequential Feature Selector (Forward/Backward).
* **Process:** Used a k-NN classifier to iteratively add or remove features to find the most optimal subset of 50 features from the PCA components.

---

## 🚀 Results & Performance
The experiment yielded the following results (using $k=3$ for k-NN):

| Method | Type | Feature Reduction | Accuracy |
| :--- | :--- | :--- | :--- |
| **PCA + kNN** | Feature Extraction | 10304 → 100 | **94.17%** |
| **SelectKBest + kNN** | Filter Method | 10304 → 100 | 58.33% |
| **Forward Wrapper** | Wrapper Method | 100 → 50 | 60.83% |
| **Backward Wrapper** | Wrapper Method | 100 → 50 | 55.00% |

### **Key Takeaway:**
**PCA (Feature Extraction)** significantly outperformed selection methods. This is because facial recognition relies on the **spatial relationship** between pixels (global structure). While Filter and Wrapper methods pick individual "best" pixels, PCA creates new features (Eigenfaces) that represent the face as a whole.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Environment:** Google Colab / Jupyter Notebook
* **Libraries:** * `Scikit-Learn` (PCA, kNN, Feature Selection)
    * `OpenCV` (Image processing)
    * `NumPy` & `Pandas` (Data handling)
    * `Matplotlib` (Visualization)

---

## 📂 Dataset Structure
The project uses the **ORL Dataset**, organized as follows:
```text
/ORL/
  /s1/ (10 images)
  /s2/ (10 images)
  ...
  /s40/ (10 images)
