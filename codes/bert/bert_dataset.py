import itertools
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd


class BERTDataset(Dataset):
    def __init__(self,
                 sequences,
                 sequence_labels,
                 tokenizer,
                 seq_len=64,
                 train=False):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sequence_lines = len(sequences)
        self.lines = sequences
        self.sequence_labels = sequence_labels
        self.train = train

    def __len__(self):
        return self.sequence_lines

    def __getitem__(self, item):

        # Step 1: get sentence, either negative or positive (saved as is_next_label)
        t1 = self.get_corpus_line(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        # t2_random, t2_label = self.random_word(t2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        # t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        # t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_label = ([1 for _ in range(len(t1))])[:self.seq_len]
        bert_input = (t1)[:self.seq_len]
        bert_label = (t1_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        # sequence_label

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "sequence_label": [self.sequence_labels[item]]}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if not self.train:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

            elif self.train and prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item]


if __name__ == "__main__":
    train_df = pd.read_excel('../data/Train.xlsx')
    sequences = train_df['String']
    sequence_labels = train_df['Property']

    MAX_LEN = 64
    tokenizer = BertTokenizer.from_pretrained('../bert-it-1/bert-it-vocab.txt',
                                              local_files_only=True)
    train_data = BERTDataset(sequences, sequence_labels, seq_len=MAX_LEN, tokenizer=tokenizer)
    train_loader = DataLoader(
       train_data, batch_size=32, shuffle=True, pin_memory=True)
    sample_data = next(iter(train_loader))
    print(train_data[random.randrange(len(train_data))])
