# Task 4: Open-Set Recognition (OSR)

This repository contains the implementation for Task 4, which explores Open-Set Recognition (OSR) by evaluating how well different model training strategies and post-hoc scoring functions distinguish known classes (CIFAR-10) from unknown, unseen classes (filtered CIFAR-100)[cite: 1]. The objective is to analyze the trade-offs between closed-set accuracy (CSA) and the ability to reject near and far semantic unknowns[cite: 1].

## Implemented Methods & Reference Literature

The codebase implements a baseline classifier, strong augmentation techniques, and representation-altering methodologies to evaluate open-set rejection limits. These implementations are directly based on the following literature:

*   **Vanilla Baseline & Post-hoc Scoring:** Evaluates Maximum Softmax Probability (MSP), Maximum Logit Score (MLS), Energy, and Mahalanobis distance on a standard ResNet-18[cite: 1].
    *   *Hendrycks and Gimpel (2017)*: Motivates the baseline maximum-softmax scoring (MSP)[cite: 1].
    *   *Liu et al. (2020)*: Provides the mathematical foundation for the Energy-based score using all classifier logits[cite: 1].
*   **GCSC (Good Closed-Set Classifier):** Uses `RandAugment` to test if improving known-class generalization inherently improves unknown rejection[cite: 1].
    *   *Vaze et al. (2022)*: Explores the relationship between closed-set classifier quality and Maximum Logit Score (MLS)[cite: 1].
*   **PROSER (Placeholder Learning):** Appends dummy classifiers and utilizes manifold mixup between known classes to synthesize proxy data placeholders[cite: 1]. 
    *   *Zhou et al. (2021)*: Provides the architecture and loss functions for learning classifier and data placeholders for OSR[cite: 1].
*   **RPL (Reciprocal Point Learning) [Optional Extension]:** Learns what each class is *not* by pushing features away from learned reciprocal points and applying open-space regularization[cite: 1].
    *   *Chen et al. (2020)*: Introduces reciprocal points to bound the known feature space[cite: 1].

## Repository Structure

The codebase isolates dataset construction, model training, score extraction, and final evaluation so that all metrics are computed on identical, reproducible splits[cite: 1, 3].

```text
task4/
├── cache/                  # Stores cached .pt files (logits and features) from extract_outputs.py
├── configs/                # YAML configuration files (base.yaml, vanilla.yaml, gcsc.yaml, proser.yaml, rpl.yaml)
├── data/
│   ├── cifar10.py          # CIFAR-10 training, validation, and test data loaders
│   ├── cifar100_unknowns.py # CIFAR-100 near and far unknown data loaders
│   ├── cifar10_split_seed6304.json # Cached index map enforcing the exact 90/10 stratified split
│   └── make_splits.py      # Generates the stratified validation split using seed 6304
├── evaluation/
│   ├── failure_analysis.py # Isolates specific semantic failures (incorrectly accepted unknowns)
│   ├── metrics.py          # Computes AUROC, FPR@95TPR, and 95th percentile validation thresholds
│   └── thresholds.py       # (Deprecated/Merged into metrics.py during refactoring)
├── methods/
│   ├── gcsc.py             # GCSC training loop with RandAugment
│   ├── manifold_mixup.py   # Beta-sampled hidden-state mixup for PROSER data placeholders
│   ├── proser.py           # PROSER training loop with CP and DP loss functions
│   ├── rpl.py              # Reciprocal point margin optimization
│   └── vanilla.py          # Standard cross-entropy training loop
├── models/
│   └── resnet_cifar.py     # Modified ResNet-18 (3x3 conv1, no initial maxpool) for 32x32 images
├── results/                # Outputs for tables, JSON metrics, and matplotlib figures
├── scores/
│   ├── energy.py           # Negative log-sum-exp scoring
│   ├── mahalanobis.py      # Feature-space distance scoring using class means and diagonal covariance
│   ├── mls.py              # Maximum Logit Score
│   └── msp.py              # Maximum Softmax Probability
├── evaluate_osr.py         # Main CPU evaluation script generating tabular results and ROC plots
├── extract_outputs.py      # GPU script to execute inference and save unaugmented logits/features to cache/
├── README.md               # Task documentation
└── train.py                # Central CLI dispatcher for all model training
