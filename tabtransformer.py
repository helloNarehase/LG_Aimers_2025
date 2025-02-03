import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, ff_dim=128, num_layers=1, dropout_rate=0.2):
        super(TabTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, ff_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(ff_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.fc(x)
        return self.sigmoid(x)

class Live_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# Training and Evaluation
def train_model(X_train, X_val, y_train, y_val, epochs=40, batch_size=512, learning_rate=1e-6):
    train_dataset = Live_Dataset(X_train, y_train)
    val_dataset = Live_Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = TabTransformer(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
    
        # Evaluation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            roc_auc = roc_auc_score(all_targets, all_preds)
            pr_auc = average_precision_score(all_targets, all_preds)

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}')
    
    pred_proba = np.array(all_preds)
    y_val_pred = (pred_proba >= 0.3).astype(int)
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))
    print("AUC PR:", average_precision_score(y_val, pred_proba))
    print("ROC AUC:", roc_auc_score(y_val, pred_proba))

    return model