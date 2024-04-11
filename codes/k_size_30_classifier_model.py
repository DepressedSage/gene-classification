import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMLayer, self).__init__()
        self.bilstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=1,
                bidirectional=True, batch_first=True
                )

    def forward(self, x):
        print(x.shape)
        output, _ = self.bilstm(x)  # Apply BiLSTM
        print(output.shape)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attention_weights = F.softmax(self.dense(x), dim=1)  # Compute attention weights
        weighted_output = torch.sum(x * attention_weights, dim=1)  # Apply attention weights
        return weighted_output


class CNNLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(CNNLayer, self).__init__()
        self.conv1d = nn.Conv1d(input_size, output_size, kernel_size)

    def forward(self, x):
        x = self.conv1d(x)
        return x


class AttentionModel(nn.Module):
    def __init__(self, input_size, max_time_steps):
        super(AttentionModel, self).__init__()
        self.cnn = CNNLayer(input_size, 30, 64)
        self.bilstm = BiLSTMLayer(30, 64)
        total_params = sum(p.numel() for p in self.bilstm.parameters())
        print(f"Total number of parameters in BiLSTM: {total_params}")
        self.attention = AttentionLayer(128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):

        print(x.shape)
        x = self.cnn(x)
        print(x.shape)
        x = F.dropout(x, p=0.3, training=self.training)
        print(x.shape)
        return x
        x = self.bilstm(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc4(x)
        return F.sigmoid(x)


if __name__ == "__main__":

    bilstm = BiLSTMLayer(input_size=64, hidden_size=64)
    bilstm.to(torch.device("cuda"))
    # Create an instance of the AttentionModel
    max_time_steps = 30
    input_size = 512  # Assuming input size matches the hidden size of BERT model
    model = AttentionModel(input_size=30, max_time_steps=512)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    batch_size = 16
    sequence_length = 15360

    input_tensor = torch.randn(batch_size, sequence_length).to(device)
    reshape_tensor = input_tensor.view(16, 30, 512)
    output = model(reshape_tensor)

    input_tensor = torch.randn(16, 30, 512).to(device)
    output = bilstm(input_tensor)
    print(output.shape)
    # print(output)
