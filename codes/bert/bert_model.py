import math
import torch
import torch.nn as nn

from config import DEVICE as device


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):
            # for each dimension of the each position
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self,
                 vocab_size,
                 n_segments,
                 max_len,
                 embed_size,
                 dry_run=False,
                 dropout=0.1):
        """
        :param vocab_size: total vocab size        self.embed_size = embed_size  # Store embed_size as an attribute
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, max_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = nn.Embedding(n_segments, embed_size, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_size, padding_idx=0)

        # self.position = PositionalEmbedding(d_model=embed_size, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)], device=torch.device('cuda'))  # Place on CUDA device
        # self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(self.pos_inp) + self.segment(segment_label)
        return self.dropout(x)


class BERTModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_segments,
                 max_len,
                 embed_size,
                 n_layers,
                 attn_heads,
                 dropout=0.1):
        super().__init__()
        self.embed_size = embed_size  # Store embed_size as an attribute
        self.embedding = BERTEmbedding(vocab_size=vocab_size, n_segments=n_segments,
                                       embed_size=embed_size, max_len=max_len,
                                       dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=attn_heads,
                                                        dim_feedforward=embed_size*4, dropout=dropout)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

    def forward(self, sequence, segment_label):
        x = self.embedding(sequence, segment_label)
        x = self.encoder_block(x)
        return x


class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class LabelClassification(torch.nn.Module):
    """
    Next Sentence Prediction Model
    2-class classification model
    """

    def __init__(self, hidden):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        self.hidden_size = hidden
        self.linear = torch.nn.Linear(hidden, 1)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # Assuming x is of size [batch_size, sequence_length, hidden_size]

        # Average pooling along the sequence dimension to obtain a single representation for the entire sequence
        sequence_representation = torch.mean(x, dim=1)  # [batch_size, hidden_size]

        # Pass the sequence representation through the linear layer
        logits = self.linear(sequence_representation)  # [batch_size, 1]

        # Apply sigmoid activation function to obtain a single probability for the entire sequence
        probabilities = torch.sigmoid(logits)  # [batch_size, 1]

        return probabilities

    def f(self, x):
        # Assuming x is of size [batch_size, sequence_length, hidden_size]
        batch_size, sequence_length, hidden_size = x.size()

        # Reshape the input tensor to [batch_size * sequence_length, hidden_size]
        x = x.view(-1, self.hidden_size)

        # Pass the reshaped tensor through the linear layer
        logits = self.linear(x)

        # Reshape the logits back to [batch_size, sequence_length, 2]
        logits = logits.view(batch_size, sequence_length, -1)

        # Apply softmax along the last dimension
        probabilities = self.softmax(logits)

        return probabilities


class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERTModel, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.embed_size = bert.embed_size
        self.classifier = LabelClassification(self.bert.embed_size)
        self.mask_lm = MaskedLanguageModel(self.bert.embed_size, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.classifier(x), self.mask_lm(x)


if __name__ == '__main__':
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3
    MAX_LEN = 64
    EMBED_DIM = 512
    N_LAYERS = 12
    ATTN_HEADS = 8
    DROPOUT = 0.1

    sample_seq = torch.randint(high=VOCAB_SIZE, size=[MAX_LEN,])
    sample_seg = torch.randint(high=N_SEGMENTS, size=[MAX_LEN,])

    embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    embedding_tensor = embedding(sample_seq, sample_seg)
    print(embedding_tensor.size())

    bert = BERTModel(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
    out = bert(sample_seq, sample_seg)
    bertlm = BERTLM(bert, VOCAB_SIZE)
    out_classifer, out_mask_lm = bertlm(sample_seq, sample_seg)
    print(out_classifer.size(), out_mask_lm.size())
    print(out_classifer, out_mask_lm)
