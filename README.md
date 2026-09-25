# Venturing into the (W)OODs: An Exploration Beyond IID Scenarios

This repository contains the comprehensive empirical evaluation suite and codebase for the **ATML Programming Assignment 1**. The project investigates model robustness when the Independent and Identically Distributed (IID) assumptions are violated. It bridges out-of-distribution performance to underlying representations, architectures, and training objectives through four targeted tasks.

## Project Structure

The repository is divided into four main task directories, alongside shared utilities and the final report. Each task directory contains its own localized `README.md` detailing its experimental framework and code structure.

*   **`task1/` — Inductive Biases and Feature Representations:**
    Isolates and evaluates the architectural priors (shape, texture, color, and spatial translation) of ResNet-50, ViT-B/16, and CLIP using the STL-10 dataset and AdaIN cue-conflict images.

*   **`task2/` — Unsupervised Domain Adaptation (UDA):**
    Adapts models from multiple source domains (Photo, Art, Cartoon) to a completely unlabeled target domain (Sketch) using the PACS dataset. Evaluates marginal (DAN, DANN) and class-conditional (CDAN) alignment strategies.

*   **`task3/` — Domain Generalization (DG):**
    Evaluates out-of-distribution generalization to the unseen Sketch domain without target access during training. Compares Empirical Risk Minimization (ERM) against explicit source alignment (DAN-DG) and Sharpness-Aware Minimization (SAM).

*   **`task4/` — Open-Set Recognition (OSR):**
    Investigates how well models reject unknown inputs (CIFAR-100) while recognizing known classes (CIFAR-10). Compares post-hoc scoring (MSP, MLS, Energy, Mahalanobis) against proactive methods (GCSC, PROSER, RPL).

*   **`shared/` — Shared Utilities:**
    Contains common data loading protocols, dataset splitting scripts (`make_splits.py`), and standard utilities reused across multiple tasks (e.g., PACS data loaders).

*   **`report/` — Final Report:**
    Contains the `report.tex` LaTeX source code, figures, and the compiled `report.pdf` which synthesizes the findings across all four tasks.

## Setup & Requirements

1. **Environment:** A Python virtual environment is recommended (e.g., `.venv/`).
2. **Dependencies:** Install the required packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Cache:** Datasets (like CIFAR-10/100, STL-10, PACS) will generally be downloaded and cached automatically to `data_cache/` or `shared/data/` by the respective scripts.

For detailed execution instructions, please refer to the `README.md` inside each respective `taskX/` directory.
