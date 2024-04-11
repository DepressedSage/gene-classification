import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from oh_encoding_dataset import ProteinDataset
from classification_models import Classifier

class Trainer:
    def __init__(self, model, criterion, optimizer, device=torch.device('cpu')):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_model = None
        self.best_val_loss = float('inf')

    def train(self, dataset, num_epochs, batch_size, num_folds=10, patience=5):
        kf = KFold(n_splits=num_folds, shuffle=True)
        for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}/{num_folds}")

            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

            # Initialize model weights for each fold
            self.model.apply(self.initialize_weights)

            # Early stopping variables
            patience_count = 0
            best_val_loss = float('inf')

            for epoch in range(num_epochs):
                # Train one epoch
                train_loss = self.train_epoch(train_loader)

                # Validate
                val_loss = self.validate(val_loader)

                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Check for early stopping
                if val_loss < best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = self.model.state_dict()
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print("Early stopping triggered.")
                        break

            print()

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                total_loss += loss.item()
        return total_loss / len(data_loader)


# Example usage:
# Define your model, criterion, optimizer, and dataset
# Then create an instance of Trainer and call the train method
if __name__ == "__main__":
    train_df = pd.read_excel('../data/Train.xlsx')
    val_df = pd.read_excel('../data/Validation.xlsx')

    train_dataset = ProteinDataset(train_df['String'], train_df['Property'])
    val_dataset = ProteinDataset(val_df['String'], val_df['Property'])

    model = Classifier()

