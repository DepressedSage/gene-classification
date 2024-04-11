import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ProteinEmbedder:
    def __init__(self):
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>']
        self.char_to_index = None
        self.index_to_char = None

    def fit(self, sequences):
        # Define the set of unique characters (amino acids) and special tokens
        unique_characters = sorted(set(''.join(sequences)) | set(self.special_tokens))

        # Create a mapping from characters to indices
        self.char_to_index = {char: i for i, char in enumerate(unique_characters)}

        # Create a reverse mapping from indices to characters
        self.index_to_char = {i: char for char, i in self.char_to_index.items()}

    def embed_sequences(self, sequences):
        # Convert each sequence to its index representation
        indexed_sequences = [[self.char_to_index[char] for char in seq] for seq in sequences]

        # Convert indexed sequences to PyTorch tensors
        tensor_sequences = [torch.tensor(seq) for seq in indexed_sequences]

        # Pad sequences to a fixed length
        padded_sequences = pad_sequence(tensor_sequences, batch_first=True, padding_value=self.char_to_index['<PAD>'])

        return padded_sequences

    def indices_to_tokens(self, indices):
        return [self.index_to_char[index] for index in indices]


class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.embedder = ProteinEmbedder()
        self.embedder.fit(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        embedded_sequence = self.embedder.embed_sequences([sequence])[0]
        return embedded_sequence, label[idx]
