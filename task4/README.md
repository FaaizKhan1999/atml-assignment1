# Task 4: Open-Set Recognition (OSR)

This repository contains the implementation for Task 4, which explores Open-Set Recognition (OSR) by evaluating how well different model training strategies and post-hoc scoring functions distinguish known classes (CIFAR-10) from unknown, unseen classes (filtered CIFAR-100). The objective is to analyze the trade-offs between closed-set accuracy (CSA) and the ability to reject near and far semantic unknowns.

## Experimental Framework

Evaluations are conducted on a modified ResNet-18 backbone adapted for $32 \times 32$ spatial dimensions. We divide unknown samples into near-semantic and far-semantic pools to isolate the impact of conceptual proximity on rejection performance. We evaluate rejection efficacy using AUROC and FPR@95TPR across the following methodologies:

*   **Vanilla Baseline & Post-hoc Scoring:** Evaluates standard post-hoc scores—Maximum Softmax Probability (MSP), Maximum Logit Score (MLS), Energy, and Mahalanobis distance—on a network trained via standard cross-entropy.
*   **GCSC (Good Closed-Set Classifier):** Investigates whether enforcing robust in-distribution generalization (via `RandAugment`) inherently improves unknown rejection boundaries.
*   **PROSER (Placeholder Learning):** A proactive architecture that appends dummy classifiers and utilizes manifold mixup between known classes to synthesize proxy data placeholders, directly optimizing the network to reject out-of-distribution inputs.
*   **RPL (Reciprocal Point Learning):** Shifts the classification paradigm by pushing known-class features away from their corresponding, learned reciprocal anchors, applying open-space regularization to bound the known latent space.

## Directory Structure

*   `cache/`: Stores intermediate `.pt` tensors (logits and features) extracted by `extract_outputs.py` for rapid CPU evaluation.
*   `configs/`: Contains YAML configuration files dictating hyperparameter setups for each strategy (`vanilla.yaml`, `gcsc.yaml`, `proser.yaml`, `rpl.yaml`).
*   `data/`
    *   `cifar10.py` / `cifar100_unknowns.py`: PyTorch datasets and loaders for known and unknown datasets.
    *   `make_splits.py`: Generates a reproducible stratified validation split for threshold calibration.
*   `evaluation/`
    *   `failure_analysis.py`: Extracts and analyzes specific semantic failures (e.g., falsely accepted near unknowns).
    *   `metrics.py`: Computes core OSR metrics (AUROC, FPR@95TPR) and validation thresholds.
*   `methods/`
    *   `vanilla.py`, `gcsc.py`, `proser.py`, `rpl.py`: Implementation of the respective training loops and custom loss functions.
    *   `manifold_mixup.py`: Beta-sampled hidden-state mixup logic used by PROSER.
*   `models/`
    *   `resnet_cifar.py`: The ResNet-18 architecture modified for small-resolution CIFAR images.
*   `results/`: Contains output tables, JSON metrics, and matplotlib figures (e.g., score distributions).
*   `scores/`
    *   `msp.py`, `mls.py`, `energy.py`, `mahalanobis.py`: Implementations of the respective post-hoc novelty scoring algorithms.
*   `train.py`: Central training dispatcher for all models.
*   `extract_outputs.py`: GPU-accelerated script that executes inference on trained checkpoints and caches raw logits/features.
*   `evaluate_osr.py`: Main CPU evaluation script that ingests cached features to generate tabular results and ROC plots.
