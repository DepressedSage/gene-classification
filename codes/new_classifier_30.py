import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Set seed for reproducibility
torch.manual_seed(153)

TIME_STEPS_LIST = [30]
INPUT_SIZE = 512

max_time_steps = max(TIME_STEPS_LIST)
max_columns = max_time_steps * INPUT_SIZE

folder_path = './bert/embeddings/'
df_list = []
for filename in os.listdir(folder_path):
    if filename.endswith('.h5'):
        file_path = os.path.join(folder_path, filename)
        tmp_df = pd.read_hdf(file_path)
        df_list.append(tmp_df)

print("Concatenating DataFrames...")
df = pd.concat(df_list, axis=0)

print("Data shape:", df.shape)
print("Data columns:", df.columns)
print("Converting embeddings to list...")
n = 20
features = [torch.tensor(em).view(30, 512) for em in list(df.embeddings)[:n]]
# features = [torch.tensor(em) for em in list(df.embeddings)]
labels = [torch.tensor(la) for la in list(df.labels)[:n]]

# Testing if embeddings are converted to tensors
print("Features:", features[0].shape)


class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Define the model architecture
class AttentionModel(nn.Module):
    def __init__(self, max_time_steps, input_size):
        super(AttentionModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=1)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=64*2, hidden_size=64, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(in_features=64*2, out_features=max_time_steps)
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(64*max_time_steps, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_size, max_time_steps)
        x = self.conv1d(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, max_time_steps, input_size)
        print(x.shape)
        x, _ = self.lstm(x)
        print(x.shape)

        x = self.attention(x)
        print(x.shape)

        attention_weights = self.softmax(x)

        print(attention_weights.shape)


        #x = x * attention_weights.unsqueeze(2)
        x = x * attention_weights

        # x = torch.bmm(attention_weights.unsqueeze(1), x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)


# Convert features and labels to PyTorch tensors
features_dataset = SimpleDataset(features, labels)

kf = KFold(n_splits=2, shuffle=True)

num_epochs = 100

batch_size = 2
# Initialize lists to store evaluation metrics across folds
accuracy_scores = []
mcc_scores = []
sensitivity_scores = []
specificity_scores = []
precision_scores = []
fscore_scores = []
roc_auc_scores = []

for k, (train_index, test_index) in enumerate(kf.split(features_dataset)):
    train_dataset = [features_dataset[i] for i in train_index]
    test_dataset = [features_dataset[i] for i in test_index]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = AttentionModel(max_time_steps, INPUT_SIZE)
    # Reset model parameters for each fold
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Training fold", k + 1)
    print("Training the model...")
    for epoch in range(num_epochs):
        print("Epoch", epoch + 1)
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluation on test set
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            predictions = torch.argmax(outputs, dim=1).numpy()
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(all_labels, all_predictions)
    fscore = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)

    accuracy_scores.append(accuracy)
    mcc_scores.append(mcc)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    precision_scores.append(precision)
    fscore_scores.append(fscore)
    roc_auc_scores.append(roc_auc)

# Calculate average evaluation metrics across folds
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
avg_mcc = sum(mcc_scores) / len(mcc_scores)
avg_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
avg_specificity = sum(specificity_scores) / len(specificity_scores)
avg_precision = sum(precision_scores) / len(precision_scores)
avg_fscore = sum(fscore_scores) / len(fscore_scores)
avg_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)

metrics_dict = {
        "Accuracies": accuracy_scores,
        "MCCs": mcc_scores,
        "Sensitivities": sensitivity_scores,
        "Specificities": specificity_scores,
        "Precisions": precision_scores,
        "F-Scores": fscore_scores,
        "ROC AUCs": roc_auc_scores
    }

metrics_excel_filename = "metrics.xlsx"

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_excel(metrics_excel_filename, index=False)
print("Metrics saved to", metrics_excel_filename)

print("Average Accuracy:", avg_accuracy)
print("Average MCC:", avg_mcc)
print("Average Sensitivity:", avg_sensitivity)
print("Average Specificity:", avg_specificity)
print("Average Precision:", avg_precision)
print("Average F1-Score:", avg_fscore)
print("Average ROC AUC Score:", avg_roc_auc)

avg_metrics_dict = {
    "Average Accuracy": [avg_accuracy],
    "Average MCC": [avg_mcc],
    "Average Sensitivity": [avg_sensitivity],
    "Average Specificity": [avg_specificity],
    "Average Precision": [avg_precision],
    "Average F1-Score": [avg_fscore],
    "Average ROC AUC Score": [avg_roc_auc]
}

# Convert the dictionary to a pandas DataFrame
avg_metrics_df = pd.DataFrame(avg_metrics_dict)

# Define the filename for saving metrics
metrics_excel_filename = "avg_metrics.xlsx"

# Save metrics to an Excel file
avg_metrics_df.to_excel(metrics_excel_filename, index=False)

print("Average metrics saved to", metrics_excel_filename)

# Save the model (you can save the best model based on a specific metric if needed)
torch.save(model.state_dict(), './attention_model.pth')
print("Model saved as 'attention_model.pth'")
