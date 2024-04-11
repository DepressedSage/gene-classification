import time
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

from config import DEVICE
from classifier_trainer import ClassifierTrainer, ClassifierDataset

if __name__ == '__main__':
    MAX_LEN = 64
    tokenizer = BertTokenizer.from_pretrained('../bert-it-1/bert-it-vocab.txt',
                                              local_files_only=True)

    train_df = pd.read_excel('../data/Train.xlsx')
    sequences = [
            ' '.join([pep for pep in sequence]) for sequence in train_df['String']
            ]
    sequence_labels = train_df['Property']

    train_data = ClassifierDataset(data=sequences, labels=sequence_labels)

    train_loader = DataLoader(
            train_data, batch_size=10, shuffle=True, pin_memory=True
            )

    val_df = pd.read_excel('../data/Validation.xlsx')
    sequences = [
            ' '.join([pep for pep in sequence]) for sequence in val_df['String']
            ]
    sequence_labels = val_df['Property']

    val_dataset = ClassifierDataset(data=sequences, labels=sequence_labels)

    val_loader = DataLoader(
            val_dataset, batch_size=10, shuffle=True, pin_memory=True
            )
