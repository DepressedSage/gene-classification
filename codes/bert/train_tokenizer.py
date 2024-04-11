import os
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
import time
import tqdm
import pandas as pd
MAX_LEN = 64

# WordPiece tokenizer


# save data as txt file
def save_data(sequences):
    os.makedirs('../data', exist_ok=True)
    text_data = []
    file_count = 0

    for sample in tqdm.tqdm([x[0] for x in sequences]):
        text_data.append(sample)

        # once we hit the 10K mark, save to file
        if len(text_data) == 10000:
            with open(f'../data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1

    paths = [str(x) for x in Path('../data').glob('**/*.txt')]
    return paths


# training own tokenizer
def train_tokenizer(paths):
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train(
        files=paths,
        vocab_size=30_000,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )

    os.makedirs('../bert-it-1', exist_ok=True)
    # tokenizer_name = f'bert-it{time.time()}'
    tokenizer_name = f'bert-it'
    tokenizer.save_model('../bert-it-1', tokenizer_name)
    return f'../bert-it-1/{tokenizer_name}-vocab.txt'


if __name__ == '__main__':
    # loading all data into memory using argument parser
    train_df = pd.read_excel('../data/Train.xlsx')
    sequences = train_df['String']
    sequence_labels = train_df['Property']

    paths = save_data(sequences)
    tokenizer_path = train_tokenizer(paths)
    print(f'Tokenizer saved at {tokenizer_path}')
