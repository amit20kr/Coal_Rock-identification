# Physics-Informed PCViT for Coal and Gangue Classification

> An end-to-end AI pipeline utilizing Near-Infrared (NIR) Spectroscopy and a custom Pyramid Convolutional Vision Transformer (PCViT) to automate lithological sorting in subterranean mining environments.

---

## Executive Summary

In modern mechanized mining operations, differentiating high-value coal from coal-measure rock (gangue) is a critical bottleneck. Traditional optical sensors fail in visually homogeneous, high-dust environments. This project replaces superficial visual sorting with **Chemical Perception**, utilizing NIR spectroscopy to probe molecular vibrational characteristics (such as C-H bonds in coal and Al-OH lattices in clay).

By engineering a robust chemometric data ingestion pipeline and designing a hybrid deep learning architecture, this system successfully classifies 24 distinct lithological classes with **>95% accuracy**, strictly protecting the fatal boundary between fuel and waste.

---

## System Architecture (Explicit Breakdown)

The pipeline is designed to eliminate the "Black Box" nature of standard deep learning by forcing the network to correlate its predictions with established geological physics.

1. **Forensic Data Ingestion:** A schema-agnostic "Surgical Loader" automatically handles messy CSV metadata, string-to-float corruptions, and inconsistent sensor orientations (transposition anomalies) to guarantee a strict 1500-band 1D output.
2. **Chemometric Math Engine:** Applies Standard Normal Variate (SNV) to mathematically strip away physical light scattering caused by dust and distance. Savitzky-Golay (SG) 1st and 2nd derivatives are subsequently applied to convert absolute intensity into shape and curvature, isolating sharp mineral absorption edges.
3. **Hybrid PCViT Model:**
* **The "Eye" (1D-CNN Stem):** Aggressively compresses the 1500-length sequence to 375 tokens via strided convolutions, hunting for localized features like the 2200nm clay trap.
* **The "Brain" (Transformer Body):** Utilizes Multi-Head Self-Attention to mathematically correlate distant spectral features (e.g., linking a 1400nm moisture feature to a 2200nm clay feature) to synthesize a global mineralogical context.



---

## Repository Pipeline

The project is structured sequentially across four core computational notebooks to ensure explicit traceability from raw data to live deployment.

| Phase | Notebook File | Primary Objective | Explicit Engineering Logic Executed |
| --- | --- | --- | --- |
| **01** | `eda.ipynb` | **Physics Verification** | Applies Min-Max normalization to isolate absorption shapes. Validates the fundamental separation criteria between flat, featureless coal signatures and deep absorption canyons found in complex silicate rocks. |
| **02** | `CoalRockPreprocess.ipynb` | **Data Engineering** | Executes the automated cleaning pipeline. Handles class imbalances, applies SNV and SG derivatives, translates 1D signals into 3-channel spatial tensors, and permanently synchronizes the `label_encoder.pkl`. |
| **03** | `FinalTrainingCR.ipynb` | **Model Optimization** | Defines the PyTorch PCViT architecture. Implements a two-stage training dynamic utilizing Focal Loss and aggressive dynamic augmentation (Gaussian noise, band-masking) to simulate chaotic mine environments. |
| **04** | `Inference_Dashboard.ipynb` | **Live Production Testing** | Simulates a live conveyor sensor stream. Autonomously routes raw signals through the math engine and outputs a hierarchical Softmax Top-K probability matrix alongside a mapped XRF geochemistry "Digital Passport". |

---

## Model Performance Metrics

To align with asymmetrical industrial economics (where classifying hard rock as coal causes catastrophic damage to downstream crushers), the model was strictly evaluated on Recall.

* **Global Accuracy:** > 95.0%
* **Fuel-Waste Boundary Violation Rate:** 0.00%
* **Regularization Impact:** The "Ankle Weights" training strategy (noise injection + 10% dropout) successfully prevented model fragility, ensuring validation accuracy remained tightly coupled with training performance without succumbing to data leakage or memorization.

---

## Quick Start / Reproduction Guide

**1. Clone the repository and install dependencies:**

```bash
git clone https://github.com/amit20kr/coal_rock-identification.git
cd coal_rock-identification
pip install torch numpy pandas matplotlib scipy pillow

```

**2. Execute the Pipeline:**

* Run `eda.ipynb` to visualize the raw spectral physics.
* Run `CoalRockPreprocess.ipynb` to build the `X_train.npy`, `y_train.npy`, and `label_encoder.pkl` artifacts.
* Run `FinalTrainingCR.ipynb` to train the PCViT architecture and save the weights to `pcvit_expert.pth`.
* Run the `Inference` cells to boot the live monitoring dashboard and test the model against simulated real-time data.

---

## Tech Stack

* **Deep Learning Framework:** PyTorch, TorchVision
* **Math & Data Engineering:** NumPy, Pandas, SciPy (Signal Processing)
* **Visualization & UI:** Matplotlib (GridSpec, Patches)
* **Environment:** Jupyter Notebook / Google Colab

---

**Amit Kumar**
*B.Tech in Mining Engineering | Indian Institute of Technology (BHU) Varanasi*
*Specializing in AI/Machine Learning, Deep Learning, and Architectural System Design 
