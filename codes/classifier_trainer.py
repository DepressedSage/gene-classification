import os
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from optimizer import ScheduledOptim
from bert_model import BERT, BERTLM
from config import DEVICE
from config import bert_config


class ClassifierDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        config = bert_config()
        model = BERT(
            vocab_size=config.vocab_size,
            n_segments=config.n_segments,
            max_len=config.max_len,
            embed_size=config.embed_size,
            n_layers=config.n_layers,
            attn_heads=config.attn_heads,
            dropout=config.dropout,
            )
        model.to(DEVICE)
        checkpoint_path = '../checkpoints/epoch_10'
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            output_embeddings = model(self.data[idx])

        last_layer_embeddings = output_embeddings[-1]
        return last_layer_embeddings, self.labels[idx]


class ClassifierTrainer:
    def __init__(
            self,
            model,
            train_dataloader,
            val_dataloader=None,
            test_dataloader=None,
            lr=1e-4,
            weight_decay=0.01,
            patience=5,
            log_freq=10,
            device='cpu',
            save_dir='../classifier',
            ):

        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.patience = patience

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
                self.optimizer, self.model.embed_size, n_warmup_steps=10000
                )

        self.criterion = nn.CrossEntropyLoss()
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epochs, dry_run=False):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.val_dataloader is not None:
            for epoch in range(1, epochs + 1):
                self.iteration(epoch, self.train_dataloader, mode="train")
                val_loss = self.iteration(epoch, self.val_dataloader, mode="validate")
                if val_loss < self.best_val_loss:
                    print(f'Validation loss decreased from {self.best_val_loss} to {val_loss}.')
                    self.best_val_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'Early stopping counter: {self.counter} out of {self.patience}.')
                    if self.counter > self.patience:
                        print(f'Early stopping after {epoch} epochs.')
                        break
                if not dry_run and epoch % 5 == 0:
                    self.save_checkpoint(f"epochs/epoch_{epoch}")
        else:
            for epoch in range(1, epochs + 1):
                self.iteration(epoch, self.train_dataloader, mode="train")
                if not dry_run and epoch % 5 == 0:
                    self.save_checkpoint(f"epochs/epoch_{epoch}")

        def test(self, epoch):
            self.iteration(epoch, self.test_dataloader, mode="test")

        def iteration(self, epoch, data_loader, mode="train"):

            avg_loss = 0.01

            data_iter = tqdm.tqdm(
                    enumerate(data_loader),
                    desc=f"Epoch {epoch}, {mode}",
                    total=len(data_loader),
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    )

            for i, data in data_iter:

                data = {key: value.to(self.device) for key, value in data.items()}

                inputs, labels = data

                classification_output = self.model(inputs)

                avg_loss += self.criterion(classification_output, labels).item()

                if mode == "train":
                    self.optim_schedule.zero_grad()
                    avg_loss.backward()
                    self.optim_schedule.step()

                post_fix = {
                    "epoch": epoch,
                    "mode": mode,
                    "avg_loss": avg_loss / (i + 1),
                    }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

            if mode == "validate":
                return avg_loss / len(data_loader)

    def save_checkpoint(self, path):
        checkpoint_path = os.path.join(self.save_dir, path)
        checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}.")
