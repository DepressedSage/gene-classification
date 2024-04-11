import torch
import torch.nn as nn


class CNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(CNNLayer, self).__init__()
        self.layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size, stride, padding),
                nn.ReLU(),
                )

    def forward(self, x):
        output = self.layer(x)
        print(output.shape)
        return output


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMLayer, self).__init__()
        self.bilstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=2,
                bidirectional=True, batch_first=True
                )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        output, _ = self.bilstm(x)  # Apply BiLSTM
        return output


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_size)
        attention_weights = torch.softmax(self.dense(x), dim=1)  # Compute attention weights
        weighted_output = torch.sum(x * attention_weights, dim=1)  # Apply attention weights
        return weighted_output


class ClassificationLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassificationLayer, self).__init__()
        self.dense = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.dense(x)


class ClassificationModule(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationModule, self).__init__()
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(64, num_classes),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.classifier(x)


class Classifier(nn.Module):
    def __init__(self, input_size, sequence_length, num_classes):
        super(Classifier, self).__init__()
        self.cnn = CNNLayer(
                in_channels=input_size, out_channels=64,
                kernel_size=3, stride=1, padding=1
                )
        self.bilstm = BiLSTMLayer(input_size=64, hidden_size=64)
        self.attention = AttentionLayer(hidden_size=128)
        self.classifier = ClassificationModule(
                input_size=128, num_classes=num_classes
                )

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        print(x.shape)
        x = self.cnn(x)  # Apply CNN
        x = self.bilstm(x)  # Apply BiLSTM
        x = self.attention(x)  # Apply Attention
        x = self.classifier(x)  # Apply Classification
        return x


if __name__ == '__main__':
    batch_size = 1
    sequence_length = 15360
    hidden_size = 1

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

# Create a random input tensor
    input_tensor = torch.randn(batch_size, sequence_length).to(device)
    reshape_tensor = input_tensor.view(1, 30, 512)

# Create an instance of the AttentionLayer
    model = Classifier(input_size=512, sequence_length=30, num_classes=2)
    model.to(device)


# Calculate the output of the attention model
    output = model(reshape_tensor)

# Output shape: (batch_size, hidden_size)
    print("Output shape:", output.shape)
    print("Output tensor:")
    print(output)
