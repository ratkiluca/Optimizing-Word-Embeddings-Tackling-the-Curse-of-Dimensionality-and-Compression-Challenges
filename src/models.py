
import torch
import torch.nn as nn
from transformers import AutoModel


class EncoderWrapper(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.backbone = AutoModel.from_pretrained(model_name).to(device)

    def forward(self, input_ids, attention_mask):
        # Standard BERT forward pass
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)

        # Mean Pooling (As defined in Section 3 of Thesis)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask


def reinit_last_layers(model, num_layers=3):
    print(f"--> Re-initializing the last {num_layers} layers of the student..")
    encoder_layers = model.backbone.encoder.layer
    layers_to_reset = encoder_layers[-num_layers:]
    for layer in layers_to_reset:
        layer.apply(model.backbone._init_weights)
