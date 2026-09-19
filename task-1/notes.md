# Representation & Inductive Bias Analysis (Task 1 Notes)

## 1. Reliance on Shape, Texture, and Color
The models exhibit minimal reliance on color; grayscale conversion and hue rotation cause negligible accuracy drops across all backbones (ranging from a 1.6% drop for ResNet on grayscale to a 4.6% drop for ViT on hue rotation). 

In contrast, cue-conflict evaluations reveal a strong structural preference for global shape over local texture, though the degree varies by architecture. ResNet-50 demonstrates the weakest shape bias at 90.3%. This architectural vulnerability to texture is visible in the disagreement examples, where ResNet predicts "ship" and "horse" based on texture cues, while ViT and CLIP correctly identify the underlying "bird" and "dog" shapes. CLIP (95.2%) and ViT (94.1%) possess the strongest shape biases. 

**The Role of Coverage:** Cue coverage ranges from 75.8% (ResNet) to 82.7% (ViT), meaning the models fail to recognize *either* shape or texture in roughly 20% of the candidate images. The visual failure cases (e.g., predicting "monkey" for a deer/truck conflict or "cat" for an airplane/car conflict) prove that AdaIN style transfer often destroys semantic meaning entirely. However, because coverage remains sufficiently high (~80%), the shape bias percentages are calculated from a statistically robust pool of valid decisions, validating the conclusion that these models are fundamentally shape-biased.

## 2. Spatial Structure: Locality and Global Organization
**Translation:** All models demonstrate exceptional translation invariance. At maximum displacement ($\delta=32$), accuracy drops by less than 2% across the board. ViT-B/16 is the most position-independent, maintaining a 99.0% prediction consistency even at $\delta=32$. This confirms that neither convolutions nor patch-based attention mechanisms are rigidly tied to absolute pixel coordinates.

**Patch Shuffling:** Destroying global spatial organization exposes stark differences in locality. ResNet and ViT suffer moderate accuracy drops (-9.4% and -7.8%, respectively) when images are shuffled into a 4x4 grid. This indicates they are capable of behaving like "bag-of-features" classifiers, aggregating local evidence (e.g., a wheel, a wing) to make correct predictions even when the global structure is scrambled. Conversely, CLIP suffers a severe 19.8% accuracy drop. CLIP relies heavily on holistic, global organization to extract semantic meaning and struggles to classify isolated local patches.

## 3. Feature Representations vs. Prediction Stability
There is a profound mismatch between prediction stability and feature stability for the ImageNet-trained models. While ResNet and ViT maintain highly stable predictions under cue-conflict (90.3% and 94.1% shape bias, respectively), their underlying feature representations shift drastically, dropping to cosine stabilities of 0.47. This mismatch suggests that the models extract entirely different features when style is altered, yet the linear classifier head still maps these shifted embeddings to the correct shape class. 

CLIP exhibits a strong agreement between features and predictions. Its representations are remarkably stable under cue-conflict (cosine stability of 0.79), which aligns perfectly with its 95.2% shape bias. 

**Zero-Shot vs. Linear Probing:** CLIP's Zero-Shot behavior almost perfectly mirrors its trained linear head. Both exhibit ~94-95% shape bias, and both suffer massive ~18-19% accuracy drops under patch shuffling. This confirms that the model's inductive biases are deeply embedded in the frozen pre-trained representations, not learned by the downstream classifier head.

## 4. Architecture vs. Pretraining
The results isolate which biases stem from architecture and which stem from the pretraining objective:

*   **Architecture (CNN vs. Transformer):** ViT's near-perfect translation consistency (99.0%) compared to ResNet (97.7%) suggests that global self-attention mechanisms are inherently less position-sensitive than the geometric receptive fields of convolutions. Additionally, ResNet's higher susceptibility to texture (choosing texture 33 times vs. ViT's 22) reflects the localized nature of convolutional feature extraction.
*   **Pretraining Data & Supervision (ImageNet vs. Contrastive Language-Image):** The starkest differences exist between ViT and CLIP, despite both utilizing a Vision Transformer backbone. ViT (supervised on ImageNet) easily categorizes shuffled patches (-7.8% drop), showing that object classification encourages models to lock onto local discriminative features. CLIP (trained on 400M image-text pairs) requires global structural coherence to align with language semantics, making it highly vulnerable to patch shuffling (-19.8% drop). CLIP's pretraining also forces it to ignore stylistic variations, resulting in much higher feature stability under cue-conflict (0.79) compared to ImageNet-trained ViT (0.47).
