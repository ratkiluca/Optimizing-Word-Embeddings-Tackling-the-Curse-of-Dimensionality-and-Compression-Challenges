import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from datasets import load_dataset


def eval_stsb(student, tokenizer, device, max_eval=None):
    stsb = load_dataset("glue", "stsb", split="validation")
    if max_eval:
        stsb = stsb.select(range(max_eval))

    student.backbone.eval()
    preds_s, targets = [], []

    with torch.no_grad():
        for i in range(len(stsb)):
            pair = stsb[i]

            # Sentence 1
            i1 = tokenizer(
                pair["sentence1"], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            emb1 = student(i1["input_ids"], i1["attention_mask"])

            # Sentence 2
            i2 = tokenizer(
                pair["sentence2"], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            emb2 = student(i2["input_ids"], i2["attention_mask"])

            cos = F.cosine_similarity(emb1, emb2)
            preds_s.append(cos.item())
            targets.append(pair["label"])

    student.backbone.train()
    pearson, _ = pearsonr(preds_s, targets)
    return {"student_pearson": pearson}
