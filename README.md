# Optimizing Word Embeddings: Tackling the Curse of Dimensionality and Compression Challenges

**Master's Thesis (2025)** *Official Code Implementation*

## 📌 Abstract
Knowledge distillation is a standard technique for compressing large pre-trained language models like BERT. However, conventional methods typically rely on pointwise loss functions (e.g., Mean Squared Error) that encourage the student model to mimic the teacher's output distribution exactly—including its high-frequency spectral noise and anisotropic geometry. In this thesis, we propose a novel geometric distillation objective, \textbf{QR-EOS} (QR-Decomposition based Eigenspace Overlap Score), which explicitly optimizes for subspace alignment rather than vector-level fidelity.

We compare standard distillation (MSE, Cosine) against our proposed spectral regularization across multiple downstream tasks. While standard methods achieve high performance on fine-grained regression (STS-B), we reveal that they suffer from a ``structural collapse'' under stress: when quantized to Int8, the MSE student loses 68\% of its effective rank and its embedding space degenerates into a narrow cone. In contrast, our Hybrid (QR+Cosine) student acts as a robust geometric compressor. It achieves a 62\% improvement in cluster separability (Silhouette Score) and retains 100\% of its retrieval precision after quantization, while offering a $2.0\times$ inference speedup and $59\%$ size reduction. Our findings suggest that geometric constraints are essential for creating compact, deployment-ready embeddings that are structurally robust to low-precision environments.

## 🚀 Key Results
| Model | Int8 Rank | Retrieval (P@1) | Clustering (Sil.) |
|-------|-----------|-----------------|-------------------|
| MSE   | 89 (Collapse)| 0.65            | 0.028             |
| **Hybrid (QR-EOS)** | **140 (Robust)** | **0.72** | **0.052** |
