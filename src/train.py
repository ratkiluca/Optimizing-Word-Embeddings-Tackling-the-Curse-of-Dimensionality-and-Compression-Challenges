
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Relative imports handling for running as script vs module
try:
    from src.models import EncoderWrapper, reinit_last_layers
    from src.losses import cosine_distill_loss, mse_distill_loss, qr_eos_loss
    from src.data import prepare_snli_sentences
    from src.eval import eval_stsb
    from src.config import *
except ImportError:
    # Fallback if running from root
    from models import EncoderWrapper, reinit_last_layers
    from losses import cosine_distill_loss, mse_distill_loss, qr_eos_loss
    from data import prepare_snli_sentences
    from eval import eval_stsb
    from config import *


def train_student(loss_mode="hybrid",
                  output_dir="./saved_models/student",
                  reinit_layers=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    print("Loading Teacher...")
    teacher = EncoderWrapper(TEACHER_MODEL, device)
    for p in teacher.backbone.parameters(): p.requires_grad = False
    teacher.backbone.eval()

    print("Loading Student...")
    student = EncoderWrapper(STUDENT_MODEL, device)
    if reinit_layers > 0:
        reinit_last_layers(student, num_layers=reinit_layers)
    student.backbone.train()

    input_ids, attention_mask = prepare_snli_sentences(tokenizer, limit=TRAIN_LIMIT, max_len=MAX_SEQ_LEN)
    dataset_size = input_ids.shape[0]

    optimizer = torch.optim.AdamW(student.backbone.parameters(), lr=LEARNING_RATE)

    best_pearson = -1.0

    for epoch in range(EPOCHS):
        perm = np.random.permutation(dataset_size)
        pbar = tqdm(range(0, dataset_size, BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for start in pbar:
            idx = perm[start:start+BATCH_SIZE]
            if len(idx) < 2: continue

            b_input = input_ids[idx].to(device)
            b_attn = attention_mask[idx].to(device)

            with torch.no_grad():
                T_emb = teacher(b_input, b_attn)

            S_emb = student(b_input, b_attn)

            l_cos = cosine_distill_loss(S_emb, T_emb)
            l_mse = mse_distill_loss(S_emb, T_emb)
            l_qr = qr_eos_loss(S_emb, T_emb, k=QR_K)

            if loss_mode == "cosine": loss = l_cos
            elif loss_mode == "mse": loss = l_mse
            elif loss_mode == "qr": loss = l_qr
            elif loss_mode == "hybrid": loss = l_cos + l_qr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"L_Cos": f"{l_cos.item():.3f}", "L_QR": f"{l_qr.item():.3f}"})

        stats = eval_stsb(teacher, student, tokenizer, device, max_eval=EVAL_LIMIT)
        print(f"--> Epoch {epoch+1} STS-B Pearson: {stats['student_pearson']:.4f}")

        if stats['student_pearson'] > best_pearson:
            best_pearson = stats['student_pearson']
            student.backbone.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved new best model to {output_dir}")

    return student
