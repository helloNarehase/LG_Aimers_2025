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
# 재현성 위한 시드 고정
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 완전한 재현성을 위해 cuDNN 설정 (다만 속도는 떨어질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=8, ff_dim=256, num_layers=4, dropout_rate=0.3):
        """
        Args:
            input_dim: 입력 피처 수
            embed_dim: 각 피처 토큰의 임베딩 차원
            num_heads: 멀티헤드 어텐션의 헤드 수
            ff_dim: Transformer의 Feed Forward Network 차원
            num_layers: Transformer 인코더 레이어 수
            dropout_rate: 드롭아웃 확률
        """
        super(TabTransformer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # 각 피처(스칼라)를 embed_dim으로 임베딩 (모든 피처가 동일한 임베딩 레이어 사용)
        self.feature_embedding = nn.Linear(1, embed_dim)
        
        # 분류를 위한 [CLS] 토큰 (학습 가능한 파라미터)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # [CLS] 토큰과 각 피처 토큰에 대해 위치 정보를 더해주기 위한 positional embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, input_dim + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 여러 Transformer Encoder Layer를 구성
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim,
                                       nhead=num_heads,
                                       dim_feedforward=ff_dim,
                                       dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 최종 분류를 위한 FC 레이어
        self.fc = nn.Linear(embed_dim, 1)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.feature_embedding.weight)
        nn.init.zeros_(self.feature_embedding.bias)
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.size(0)
        # 각 피처를 (batch, input_dim, 1) 형태로 변환
        x = x.unsqueeze(-1)
        # 각 피처에 대해 임베딩: (batch, input_dim, embed_dim)
        x = self.feature_embedding(x)
        # [CLS] 토큰을 생성하여 피처 임베딩 앞에 붙임: (batch, 1 + input_dim, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 위치 정보를 더함
        x = x + self.positional_embedding
        x = self.dropout(x)
        
        # Transformer는 (seq_len, batch, embed_dim) 형태를 기대하므로 전치
        x = x.transpose(0, 1)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        # [CLS] 토큰의 출력만 취함 (분류용)
        cls_rep = x[:, 0, :]  # (batch, embed_dim)
        output = self.fc(cls_rep)  # (batch, 1)
        return output
    
    def predict_proba(self, X, batch_size=256):
        """
        입력 X에 대해 배치 단위로 확률 예측을 수행합니다.
        """
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        # TensorDataset과 DataLoader를 사용해 배치 단위로 데이터를 처리
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_logits = []
        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(next(self.parameters()).device)
                logits = self.forward(batch_X)  # (batch, 1)
                all_logits.append(logits.cpu())
        
        # 모든 배치의 결과를 이어 붙임
        all_logits = torch.cat(all_logits, dim=0).numpy()
        # Sigmoid를 통해 확률로 변환
        probas = 1 / (1 + np.exp(-all_logits))
        # [1 - probas, probas] 형태로 반환
        return np.concatenate([1 - probas, probas], axis=1)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # logits 사용
        pt = torch.exp(-BCE_loss)  # 확률 변환
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class LiveDataset(Dataset):
    def __init__(self, X, y):
        # CPU 상에 데이터를 보관 (GPU로의 이동은 학습 loop에서 진행)
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# Training and Evaluation
def train_model(X_train, X_val, y_train, y_val, epochs=100, batch_size=128, learning_rate=1e-4, patience=20):
    train_dataset = LiveDataset(X_train, y_train)
    val_dataset = LiveDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    model = TabTransformer(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = FocalLoss(alpha=0.25, gamma=1.5)
    
    best_auc = 0
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # 배치 단위에서 GPU로 이동
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