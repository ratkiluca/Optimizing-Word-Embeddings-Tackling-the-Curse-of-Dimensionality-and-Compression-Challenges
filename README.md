# Optimizing Word Embeddings: Tackling the Curse of Dimensionality and Compression Challenges

**Master's Thesis (2025)**

## 📌 Abstract
Knowledge distillation is a standard technique for compressing large pre-trained language models like BERT. However, conventional methods typically rely on pointwise loss functions (e.g., Mean Squared Error) that encourage the student model to mimic the teacher's output distribution exactly—including its high-frequency spectral noise and anisotropic geometry. In this thesis, we propose a novel geometric distillation objective, **QR-EOS** (QR-Decomposition based Eigenspace Overlap Score), which explicitly optimizes for subspace alignment rather than vector-level fidelity.

## 🚀 Key Results (Int8 Quantization)
The table below demonstrates the "Structural Collapse" of the MSE model versus the robustness of the Hybrid model when quantized to 8-bit integers.

| Model | Effective Rank | Retrieval (P@1) | Clustering (Sil.) |
|-------|----------------|-----------------|-------------------|
| **Teacher (BERT)** | 246 (Stable)   | 0.72            | 0.040             |
| MSE Student        | 89 (Collapse)  | 0.65            | 0.028             |
| **Hybrid (QR-EOS)**| **140 (Robust)**| **0.72** | **0.052**|
