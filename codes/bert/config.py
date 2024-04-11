import torch

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def bert_config():
    return {
        "device": device,
        "n_segments": 3,
        "max_len": 64,
        "embed_size": 512,
        "n_layers": 4,
        "attn_heads": 8,
        "dropout": 0.1
    }


DEVICE = torch.device(device)
