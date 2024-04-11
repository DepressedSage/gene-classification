import os
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
import torch
from transformers import BertTokenizer
from bert_model import BERTModel, BERTLM
from bert_dataset import BERTDataset
from torch.utils.data import DataLoader

# 1. Load the trained BERTLM model
checkpoint_path = '../checkpoints/bert_lm_final.pth'
checkpoint = torch.load(checkpoint_path)
MAX_LEN = 30
tokenizer = BertTokenizer.from_pretrained('../../bert-it-1/bert-it-vocab.txt',
                                          local_files_only=True)

train_df = pd.read_excel('../../data/Train.xlsx')
val_df = pd.read_excel('../../data/Validation.xlsx')

sequences = []
sequence_labels = []

sequences.append(
        [' '.join([pep for pep in sequence]) for sequence in train_df['String']])
sequence_labels.append([label for label in train_df['Property']])

sequences.append([' '.join([pep for pep in sequence]) for sequence in val_df['String']])
sequence_labels.append([label for label in val_df['Property']])

sequences = [item for sublist in sequences for item in sublist]
sequence_labels = [item for sublist in sequence_labels for item in sublist]

val_data = BERTDataset(sequences,
                       sequence_labels,
                       seq_len=MAX_LEN,
                       tokenizer=tokenizer,
                       train=False
                       )

val_loader = DataLoader(
   val_data, batch_size=1, shuffle=True, pin_memory=True)

bert_model = BERTModel(
  vocab_size=len(tokenizer.vocab),
  n_segments=3,
  max_len=MAX_LEN,
  embed_size=512,
  n_layers=4,
  attn_heads=8,
  dropout=0.1,
  cpu=True
)

bertlm = BERTLM(bert_model, len(tokenizer.vocab))
bertlm.load_state_dict(checkpoint['model_state_dict'])
bertlm.eval()

# 2. Extract the BERT model
bert_model = bertlm.bert

# 3. Generate embeddings
output_folder = "embeddings"
os.makedirs(output_folder, exist_ok=True)

batch_size = 10
batch_count = 0
embedding_list = []
labels_list = []

for i, data in enumerate(tqdm(val_loader, desc="Generating BERT embeddings")):
    embedding = bert_model(data['bert_input'], data['segment_label'])
    flattened_embedding = torch.flatten(embedding, start_dim=1).detach().numpy()  # Flatten along the feature dimension

    embedding_list.append(flattened_embedding)
    labels_list.append(data['sequence_label'].numpy())

    if len(embedding_list) == batch_size:
        df = pd.DataFrame({
            'embeddings': embedding_list,
            'labels': labels_list
        })

        df.to_hdf(f'{output_folder}/embeddings_{batch_count}.h5', key='embeddings', mode='w')
        embedding_list.clear()
        labels_list.clear()
        batch_count += 1

# Save the remaining embeddings if the last batch is not a full batch
if len(embedding_list) > 0:
    df = pd.DataFrame({
        'embeddings': embedding_list,
        'labels': labels_list
    })

    df.to_hdf(f'{output_folder}/embeddings_{batch_count}.h5', key='embeddings', mode='w')

# Assuming you have your input data in sample_seq and sample_seg
