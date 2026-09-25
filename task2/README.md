# Task 2: Unsupervised Domain Adaptation (UDA)

This repository contains the implementation for Task 2, which explores Unsupervised Domain Adaptation (UDA). The goal is to evaluate whether reducing the statistical discrepancy between labeled source domains and an unlabeled target domain effectively preserves class-discriminative information, or if it inadvertently induces negative transfer.

## Experimental Framework

Evaluations are conducted using the PACS dataset. The models are trained on three labeled source domains (Photo, Art Painting, Cartoon) and adapted to one unlabeled target domain (Sketch). We employ a pre-trained ResNet-18 backbone and evaluate four adaptation paradigms:

*   **Source-Only (ERM):** Establishes the baseline unadapted transferability of the network, trained exclusively on source data using standard cross-entropy.
*   **DAN (Maximum Mean Discrepancy):** Enforces marginal alignment by applying an MMD penalty (using a multi-bandwidth RBF kernel) to the feature space prior to the classifier head.
*   **DANN (Domain-Adversarial Neural Network):** Enforces marginal alignment via a binary domain discriminator optimized with a Gradient Reversal Layer (GRL). We also include a controlled study varying the maximum GRL strength.
*   **CDAN (Conditional Adversarial Domain Adaptation):** Executes class-conditional alignment by conditioning the domain discriminator on the classifier's probability predictions.

To independently quantify the domain-specific information retained, we freeze the adapted backbones and compute a **Domain Separability Score** by training a logistic regression classifier to distinguish source from target features.

## Directory Structure

*   `configs/`
    *   `base.yaml`: Base configuration settings for the experiments.
*   `evaluation/`
    *   `domain_separability.py`: Computes the domain separability score using held-out feature representations.
    *   `metrics.py`: General evaluation and logging utilities.
*   `methods/`
    *   `mmd.py`: Implementation of the Maximum Mean Discrepancy (MMD) loss.
    *   `source_only.py`: Implementation of the baseline ERM training logic.
*   `models/`
    *   `backbone.py`: Contains the `DomainAdaptationResNet` architecture (ResNet-18).
    *   `classifier_head.py`: Defines the classification layer.
    *   `domain_discriminator.py`: Defines the adversarial discriminator network.
    *   `grl.py`: Implements the Gradient Reversal Layer.
*   `results/`
    *   Contains plotting scripts (`plotting.py`), JSON history files, CSV metrics, and generated figures detailing the training curves and the controlled GRL study.
*   `train.py`: Central training script orchestrating the dataset loading, model initialization, and execution of the selected adaptation method (Source-only, DAN, DANN, CDAN).
*   `utils.py`: Helper functions for seed setting and general utility.
*   `evaluate_final.py`: Runs the final evaluation suite on the trained checkpoints to generate aggregate tables and per-class performance metrics.
