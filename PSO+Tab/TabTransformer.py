import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import gc



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1, head_dim = 96):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        
        self.wv = nn.Linear(embed_size, self.head_dim * self.heads, bias=False)
        self.wk = nn.Linear(embed_size, self.head_dim * self.heads, bias=False)
        self.wq = nn.Linear(embed_size, self.head_dim * self.heads, bias=False)
        self.wo = nn.Linear(self.head_dim * self.heads, embed_size, bias=False)
        
    def forward(self, x):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.dropout(self.wq(x)), self.dropout(self.wk(x)), self.dropout(self.wv(x))

        xq = xq.view(bsz, seqlen, self.heads, self.head_dim)
        keys = xk.view(bsz, seqlen, self.heads, self.head_dim)
        values = xv.view(bsz, seqlen, self.heads, self.head_dim)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, embed_size, dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, dim)
        self.fc2 = nn.Linear(dim, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TabTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, head_dim=92, expansion=2, dropout=0.1):
        super().__init__()
        self.attention = Attention(embed_size, heads, dropout, head_dim)
        self.attention_norm = RMSNorm(embed_size)
        self.ffn = FeedForward(embed_size, expansion, dropout)
        self.ffn_norm = RMSNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out = self.attention(self.attention_norm(x))
        x = x + self.dropout(attn_out)
        
        ff_out = self.ffn(self.ffn_norm(x))
        out = x + self.dropout1(ff_out)
        return out

class FTTransformer(nn.Module):
    def __init__(self, num_classes, num_categorical, num_numerical, embedding_dim=16, layers=6, heads=8, expansion_factor=4, dropout=0.1):  
        super().__init__()

        self.embedding_dim = embedding_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        self.categorical_embeddings = nn.ModuleList([nn.Embedding(num_categories, embedding_dim) for num_categories in num_categorical])
        for embedding in self.categorical_embeddings:
            nn.init.xavier_uniform_(embedding.weight)

        self.numerical_embedding = nn.Sequential(
            nn.LayerNorm(num_numerical),
            nn.Linear(num_numerical, num_numerical, bias=False),
        )
        
        total_tokens = len(num_categorical)
        self.transformer_blocks = nn.ModuleList(
            [TabTransformerBlock(embedding_dim, heads, expansion=expansion_factor, dropout=dropout) for _ in range(layers)]
        )

        self.norm = nn.LayerNorm(total_tokens+num_numerical)
        self.mlp_head = nn.Sequential(
            nn.Linear(total_tokens+num_numerical, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, categorical_data, numerical_data):
        cat_embeds = torch.stack([embedding(categorical_data[:, i]) for i, embedding in enumerate(self.categorical_embeddings)], dim=1)
        num_embeds = self.numerical_embedding(numerical_data)
        
        # x = torch.cat([self.cls_token.expand(cat_embeds.shape[0], -1, -1), cat_embeds], dim=1)
        x = cat_embeds
        
        for block in self.transformer_blocks:
            x = block(x)

        # x = torch.cat([x[:, 0], num_embeds], 1)
        x = torch.cat([x.mean(-1), num_embeds], 1)
        x = self.norm(x)  # CLS 토큰
        return self.mlp_head(x)

class EStop:
    
    def __init__(self, down=True, n = 5):
        print('\033[96m' + 'Start Early Stopping!' + '\033[0m')

        self.down = down
        self.ag = float("inf") if down else float("-inf")
        self.stack = 0
        self.n = n

    def step(self, value):
        
        if (self.down and self.ag > value) or ((not self.down) and self.ag < value):
            self.stack = 0
            self.ag = value
        else:
            self.stack += 1
            print('\033[31m' + f'Counting! +{self.stack}' + '\033[0m')
        
        if self.stack > self.n:
            print('\033[31m' + 'Run Early Stopping!' + '\033[0m')
        return self.stack > self.n


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train_model(model:FTTransformer, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device="cuda"):
    reduce_lr = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)
    es = EStop(down= False, n= 12)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        all_labels, all_preds = [], []

        for i, (categorical, numberical, y_batch) in enumerate(train_loader):
            
            categorical, numberical, y_batch = \
            \
                categorical.to(device, non_blocking=True),\
                numberical.to(device, non_blocking=True),\
                y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()

            outputs = model(categorical, numberical)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())

            del categorical, numberical, y_batch, outputs, loss
            # if i % 100 == 0:
            #     clear_memory()

        scheduler.step()


        # Validation phase
        model.eval()
        val_loss = 0.0
        val_labels, val_preds = [], []

        with torch.no_grad():
            for categorical, numberical, y_batch in val_loader:
                categorical, numberical, y_batch = \
                \
                    categorical.to(device, non_blocking=True),\
                    numberical.to(device, non_blocking=True),\
                    y_batch.to(device, non_blocking=True)
                outputs = model(categorical, numberical)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                val_labels.extend(y_batch.cpu().numpy())
                val_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())

        # Compute AUCPR for validation
        aucpr_val = average_precision_score(val_labels, val_preds)

        # Compute PR AUC for validation
        precision_val, recall_val, _ = precision_recall_curve(val_labels, val_preds)
        pr_auc_val = auc(recall_val, precision_val)

        # Compute ROC AUC for validation
        roc_auc_val = roc_auc_score(val_labels, val_preds)
        
            # return model
        
        # 🔥 Memory cleanup after validation
        # clear_memory()
        # if epoch % 2 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f},", 
            f"Validation Loss: {val_loss / len(val_loader):.4f}, AUCPR: {aucpr_val:.4f},",
            f"PR AUC: {pr_auc_val:.4f}, ROC-AUC: {roc_auc_val:.4f}")

        # Reduce LR if validation loss does not improve
        reduce_lr.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        if es.step(aucpr_val):
            return 
