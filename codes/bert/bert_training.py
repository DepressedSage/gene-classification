import time
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

from config import DEVICE
from trainer import BERTTrainer
from bert_model import BERTModel, BERTLM
from bert_dataset import BERTDataset


if __name__ == '__main__':
    MAX_LEN = 64
    tokenizer = BertTokenizer.from_pretrained('../../bert-it-1/bert-it-vocab.txt',
                                              local_files_only=True)
    train = True

    bert_model = BERTModel(
      vocab_size=len(tokenizer.vocab),
      n_segments=3,
      max_len=MAX_LEN,
      embed_size=512,
      n_layers=4,
      attn_heads=8,
      dropout=0.1
    )

    bert_lm = BERTLM(bert_model, len(tokenizer.vocab))

    epochs = 100
    torch.cuda.empty_cache()

    if not train:
        test_df = pd.read_excel('../../data/Test.xlsx')
        sequences = [' '.join([pep for pep in sequence]) for sequence in test_df['String']]
        sequence_labels = test_df['Property']

        test_data = BERTDataset(
           sequences, sequence_labels, train=False, seq_len=MAX_LEN, tokenizer=tokenizer)

        test_loader = DataLoader(
           test_data, batch_size=10, shuffle=True, pin_memory=True)

        bert_trainer = BERTTrainer(bert_lm,
                                   patience=10,
                                   device=DEVICE,
                                   test_dataloader=test_loader,
                                   load_pretrained=True,
                                   load_checkpoint="bert_lm_final")

        bert_trainer.test()

    if train:
        train_df = pd.read_excel('../../data/Train.xlsx')
        sequences = [' '.join([pep for pep in sequence]) for sequence in train_df['String']]
        sequence_labels = train_df['Property']

        train_data = BERTDataset(
           sequences, sequence_labels, seq_len=MAX_LEN, tokenizer=tokenizer)

        train_loader = DataLoader(
           train_data, batch_size=10, shuffle=True, pin_memory=True)

        val_df = pd.read_excel('../../data/Validation.xlsx')
        sequences = [' '.join([pep for pep in sequence]) for sequence in train_df['String']]
        sequence_labels = train_df['Property']

        val_data = BERTDataset(
           sequences, sequence_labels, seq_len=MAX_LEN, tokenizer=tokenizer)

        val_loader = DataLoader(
           val_data, batch_size=10, shuffle=True, pin_memory=True)

        print(DEVICE)
        bert_trainer = BERTTrainer(bert_lm,
                                   patience=10,
                                   train_dataloader=train_loader,
                                   val_dataloader=val_loader,
                                   device=DEVICE)
        bert_trainer.train(epochs)
        bert_trainer.save_checkpoint(f'bert_lm_{time.time()}')
