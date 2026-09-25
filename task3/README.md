# Task 3: Domain Generalization (DG)

This repository contains the implementation for Task 3, which explores Domain Generalization (DG). Unlike unsupervised domain adaptation, domain generalization strictly prohibits any access to target-domain data during training and validation. The objective is to determine whether empirical risk minimization across diverse source domains is sufficient, or if explicit representation invariance or local parameter stability is required for true generalization.

## Experimental Framework

Evaluations are conducted on the PACS dataset. The model is trained on labeled source domains (Photo, Art Painting, Cartoon) and must generalize to a strictly unseen target domain (Sketch). We employ a pre-trained ResNet-18 backbone and contrast three distinct generalization strategies:

*   **ERM (Empirical Risk Minimization):** The Source-Only baseline inherited from Task 2, which optimizes the standard cross-entropy risk over combined source domains without any explicit alignment.
*   **DAN-DG (Domain Adversarial Network for DG):** Adapts explicit marginal feature alignment (via an MMD penalty) to enforce representation invariance strictly across the observed source domains, minimizing source-domain discrepancy.
*   **SAM (Sharpness-Aware Minimization):** Encourages local parameter stability by optimizing for uniformly flat loss landscapes without explicitly penalizing domain discrepancies. A controlled study evaluates varying neighborhood perturbation radii ($\rho \in \{0.01, 0.05, 0.1\}$).

We also implement diagnostic tools to quantify the residual domain-specific information (Source-Domain Separability Score) and standard local parameter flatness (Local Sharpness Proxy).

## Directory Structure

*   `checkpoints/`: Stores the best `.pt` model weights for each trained strategy.
*   `configs/`: Contains YAML configuration files dictating hyperparameter setups for SAM (`sam_01.yaml`, `sam_05.yaml`, etc.) and DAN-DG.
*   `evaluation/`
    *   `domain_metrics.py`: Standard logging and metric aggregation functions.
    *   `sharpness.py`: Computes the standardized local sharpness proxy via a fixed gradient-ascent perturbation.
    *   `source_domain_separability.py`: Computes the separability score among the observed source domains.
*   `methods/`
    *   `dan_dg.py`: Training logic and MMD objective modified for multi-source alignment.
    *   `sam.py`: The Sharpness-Aware Minimization optimizer implementation.
*   `results/`: Contains output JSON diagnostics, CSV tables, and the `plotting.py` script for visualizing training curves and the impact of the SAM perturbation radius.
*   `selection/`
    *   `source_validation.py`: Handles model checkpoint selection based strictly on source-validation performance to respect the DG protocol.
*   `train.py`: Central training dispatcher for executing SAM and DAN-DG strategies.
*   `evaluate_sketch.py`: The final evaluation script that tests the selected models against the strictly unseen Sketch target domain.
*   `run_diagnostics.py`: Orchestrates the computation of the source-domain separability and local sharpness metrics on trained checkpoints.
