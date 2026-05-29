<div align="center">

# Automated Glaucoma Detection from Fundus Images using Attention-Guided Deep Learning
### B.Tech Final Year Thesis Project (BTP)

**Group 129**  
Prabhmeet Singh (2022UCM2305) • Aryan Jain (2022UCM2330) • Radhacharan (2022UCM2365)  

**Supervised by:** Dr. Surendra Nagar

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Python 3](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

</div>

---

## 📑 Abstract

Glaucoma is an optic neuropathy and a leading cause of irreversible blindness worldwide. Early detection is clinically vital to halt its progression. This final-year thesis proposes a highly robust, real-time machine learning pipeline to automate glaucoma screening from retinal fundus images. By combining a computationally efficient **EfficientNet-B0** backbone with a **Convolutional Block Attention Module (CBAM)**, our model dynamically focuses on diagnostically critical regions such as the optic disc and cup. Optimized with a custom Focal Loss, Label Smoothing, and Mixed Precision training, the pipeline aims to maximize clinical sensitivity while mitigating class imbalance inherent in medical datasets.

---

## 🎯 Problem Statement and Motivation

The diagnosis of glaucoma heavily relies on the manual examination of the optic nerve head via fundus imaging or Optical Coherence Tomography (OCT). This process is time-consuming, requires high clinical expertise, and is prone to inter-observer variability. 
Our project addresses the need for a **rapid, reliable, and automated screening tool** that can be deployed in resource-constrained clinical settings to flag early signs of glaucoma, prioritizing high sensitivity to minimize false-negative misdiagnoses.

---

## 📊 Dataset Description

The models are trained and evaluated on a custom-curated fundus dataset (**Glumoma2.0**). The dataset is strictly partitioned into non-overlapping training, validation, and testing sets to ensure unbiased evaluation.

| Split | Number of Images | Description |
| :--- | :--- | :--- |
| **Training Set** | 8,865 | Used for initial feature learning and model fine-tuning. |
| **Validation Set** | 985 | Used for hyperparameter tuning and threshold selection targeting 90%+ sensitivity. |
| **Test Set** | 2,463 | Hold-out set for final clinical metric evaluation. |
| **Total** | **12,313** | |

*Note: The dataset involves severe visual feature complexities and class imbalances, which are handled programmatically in the training pipeline.*

---

## 🧠 Proposed Methodology & Architecture

The thesis introduces a novel hybrid architecture tailored specifically for medical image analysis:

### 1. Backbone: EfficientNet-B0
Pre-trained on ImageNet, EfficientNet-B0 was selected for its optimal balance of high feature extraction capability and low computational overhead (essential for real-time inference).

### 2. Attention Mechanism: CBAM
A **Convolutional Block Attention Module (CBAM)** is integrated immediately following the backbone. The CBAM operates in two sequential sub-modules:
* **Channel Attention:** Learns "what" features are meaningful across the 1280 feature maps.
* **Spatial Attention:** Learns "where" the meaningful features are located structurally, effectively learning to isolate the optic disc and retinal nerve fiber layer defects without manual annotations.

### 3. Custom Classification Head
The extracted and attention-refined features are passed through a custom classification head designed to prevent overfitting while capturing high-level representations:
* Global Average Pooling (GAP) for spatial dimensionality reduction.
* Deep dense blocks `(512 -> 256 -> 128)` utilizing `Swish` activation and Batch Normalization.
* **Residual Connections** are injected directly into the dense blocks to preserve gradient flow during deep fine-tuning.
* High Dropout rates (up to 40%) for aggressive regularization.

---

## ⚙️ Optimization & Training Strategy

To ensure clinical viability and model convergence, the training pipeline incorporates several advanced deep learning techniques:

* **Two-Phase Fine-Tuning Strategy:**
  * *Phase 1 (Domain Adaptation):* Backbone frozen; Head trained at learning rate $3 \times 10^{-4}$ for 12 epochs to establish basic decision boundaries.
  * *Phase 2 (End-to-End Fine-Tuning):* Entire network unfrozen; trained at a highly reduced learning rate $1 \times 10^{-5}$ for 25 epochs using AdamW optimizer with a weight decay of $1 \times 10^{-5}$.
* **Modified Focal Loss & Label Smoothing:** We implemented a custom Focal Loss function ($\alpha = 0.65, \gamma = 2.0$) natively integrated with Label Smoothing ($0.05$). This forces the network to focus on hard, borderline cases while preventing overconfidence in predictions.
* **Real-time Data Augmentation:** The training pipeline applies stochastic transformations (Rotations, Horizontal Flips, Brightness, and Contrast shifts) using optimized `tf.data` pipelines.
* **Mixed Precision (FP16):** Tensor cores (T4 GPUs) are fully utilized via `mixed_float16` policies, halving memory consumption and accelerating the training loop.

---

## 📈 Evaluation & Clinical Metrics

Unlike standard classification tasks, medical screening demands a rigorous evaluation protocol centered around patient safety.

1. **Threshold Optimization:** The evaluation script (`glaucoma_evaluation_[GRP_129]_.ipynb`) dynamically recalibrates the sigmoid decision threshold. Instead of a default $0.5$, the threshold is calculated on the validation set to strictly guarantee a **Sensitivity $\ge$ 90%**.
2. **Metrics Tracked:**
   * **Sensitivity (Recall):** Primary metric. Represents the True Positive Rate (ensuring diseased patients aren't missed).
   * **Specificity:** True Negative Rate (minimizing false alarms).
   * **ROC-AUC:** Measures the overall discriminative capacity of the model across all thresholds.
   * **Precision-Recall AUC (PR-AUC):** Provides a robust measure of performance under class imbalance.

---

## 📂 Repository Structure

```text
├── glaucoma_pipeline_[GRP_129]_.ipynb      # Core training script (Data ingestion, Architecture, 2-phase training)
├── glaucoma_evaluation_[GRP_129]_.ipynb    # Clinical evaluation (Threshold tuning, Test set metrics, ROC/PR curves)
├── README.md                               # Project documentation
└── report/                                 # [Future] Final Thesis Report & Presentation slides
```

---

## 🚀 Execution Guide

### Prerequisites
* Python 3.8+
* TensorFlow 2.10+
* Scikit-Learn, OpenCV, Pandas, Matplotlib, Seaborn

### 1. Data Preparation
Ensure the `Glumoma2.0` dataset is mounted or available locally. Update the `CONFIG` paths inside the notebooks:
```python
"train_dir": "/path/to/dataset/train",
"val_dir": "/path/to/dataset/val",
"test_dir": "/path/to/dataset/test",
"save_dir": "/path/to/models/"
```

### 2. Training the Pipeline
Open `glaucoma_pipeline_[GRP_129]_.ipynb` and execute all cells. 
* The script automatically copies data to local NVMe drives (`/content/local_data`) for faster I/O if running on Colab.
* Model checkpoints will be generated in `save_dir`.

### 3. Clinical Evaluation
Open `glaucoma_evaluation_[GRP_129]_.ipynb`. Ensure it points to the generated `.keras` checkpoint from Phase 2. Run the notebook to compute optimal thresholds and generate confusion matrices and ROC curves.

---

## 🔮 Future Scope & Limitations

* **Explainable AI (XAI):** Implementing Grad-CAM or Integrated Gradients to visually highlight the regions (e.g., cupping) that trigger positive classifications, aiding clinical trust.
* **Multi-Modal Integration:** Fusing fundus images with OCT scans or patient metadata (IOP, Age, Family History) to create a more comprehensive diagnostic tool.
* **Edge Deployment:** Quantizing the EfficientNet-B0 model to INT8 formats for deployment on edge devices like mobile phones or embedded clinic cameras (e.g., via TensorFlow Lite).

---

<div align="center">
<i>This work was developed as a Final Year B.Tech Thesis Project.</i>
</div>
