import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report

# =======================
# 재현성을 위한 시드 고정
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# FTTransformer를 위한 Transformer Block 정의
# ---------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (batch_size, num_tokens, embed_dim)
        # MultiheadAttention는 (seq_len, batch_size, embed_dim)를 기대하므로 전치
        x_transposed = x.transpose(0, 1)  # (num_tokens, batch_size, embed_dim)
        attn_out, _ = self.attn(x_transposed, x_transposed, x_transposed)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, num_tokens, embed_dim)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x

# ---------------------------------------------------
# FTTransformer 모델 정의
# ---------------------------------------------------
class FTTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=8, num_layers=4, dropout=0.3, mlp_hidden_dim=256):
        """
        Args:
            input_dim: 입력 피처 수 (토큰 수)
            embed_dim: 각 피처 토큰의 임베딩 차원
            num_heads: 멀티헤드 어텐션의 헤드 수
            num_layers: Transformer Block의 개수
            dropout: 드롭아웃 확률
            mlp_hidden_dim: 최종 MLP 헤드의 은닉 차원
        """
        super(FTTransformer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 각 피처(스칼라)를 개별적으로 임베딩하기 위한 파라미터.
        # 각 피처 i에 대해: token_i = x_i * weight_i + bias_i
        self.feature_weight = nn.Parameter(torch.empty(input_dim, embed_dim))
        self.feature_bias = nn.Parameter(torch.empty(input_dim, embed_dim))
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.zeros_(self.feature_bias)
        
        # 여러 Transformer Block 구성
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 최종 분류를 위한 MLP 헤드
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        # 각 피처를 (batch_size, input_dim, 1)로 확장
        x = x.unsqueeze(-1)
        # 각 피처별 선형 변환 적용 (브로드캐스팅 활용)
        x = x * self.feature_weight + self.feature_bias  # (batch_size, input_dim, embed_dim)
        
        # Transformer Block들을 통과
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        
        # 토큰들에 대해 평균(pooling) → (batch_size, embed_dim)
        x = x.mean(dim=1)
        out = self.mlp(x)  # (batch_size, 1)
        return out
    
    def predict_proba(self, X, batch_size=256):
        """
        입력 X에 대해 배치 단위로 확률 예측을 수행합니다.
        """
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_logits = []
        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(next(self.parameters()).device)
                logits = self.forward(batch_X)
                all_logits.append(logits.cpu())
        all_logits = torch.cat(all_logits, dim=0).numpy()
        probas = 1 / (1 + np.exp(-all_logits))
        return np.concatenate([1 - probas, probas], axis=1)

# ---------------------------------------------------
# Focal Loss 정의 (로지트 기반)
# ---------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------------------------------------------------
# Dataset 클래스 정의
# ---------------------------------------------------
class LiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# ---------------------------------------------------
# Training 및 Evaluation 함수 정의
# ---------------------------------------------------
def train_model(X_train, X_val, y_train, y_val, epochs=100, batch_size=128, learning_rate=1e-4, patience=20):
    train_dataset = LiveDataset(X_train, y_train)
    val_dataset = LiveDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    model = FTTransformer(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = FocalLoss(alpha=0.25, gamma=1.5)
    
    best_auc = 0
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy().flatten())
        val_loss /= len(val_loader)
        roc_auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}')
        scheduler.step(roc_auc)
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            torch.save(model.state_dict(), "best_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        
    pred_proba = np.array(all_preds)
    y_val_pred = (pred_proba >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(all_targets, y_val_pred))
    print(classification_report(all_targets, y_val_pred))
    print("AUC PR:", average_precision_score(all_targets, pred_proba))
    print("ROC AUC:", roc_auc_score(all_targets, pred_proba))

    return model