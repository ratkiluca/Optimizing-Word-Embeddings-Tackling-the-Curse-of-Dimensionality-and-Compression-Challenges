
import torch
import torch.nn.functional as F


def cosine_distill_loss(student_emb, teacher_emb):
    cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=1)
    return 1.0 - cos_sim.mean()


def mse_distill_loss(student_emb, teacher_emb):
    return F.mse_loss(student_emb, teacher_emb)


def qr_decomposition(x):
    if x.dim() < 2:
        x = x.unsqueeze(0)
    try:
        Q, R = torch.linalg.qr(x)
    except Exception:
        Q, R = torch.qr(x)
    return Q


def qr_eos_loss(student_emb, teacher_emb, k=16):
    # Ensure we don't exceed batch size
    current_k = min(k, student_emb.shape[0], student_emb.shape[1])

    Q_s = qr_decomposition(student_emb)
    Q_t = qr_decomposition(teacher_emb)

    # Truncate to k
    Q_s_k = Q_s[:, :current_k]
    Q_t_k = Q_t[:, :current_k]

    # Compute EOS
    interaction = torch.matmul(Q_s_k.t(), Q_t_k)
    eos_score = (torch.norm(interaction, p='fro') ** 2) / current_k

    return 1.0 - eos_score
