import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        assert embed_size % heads == 0, "Embedding size must be divisible by number of heads"
        self.head_dim = embed_size // heads
        self.dropout = nn.Dropout(dropout)
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        query = query.reshape(N, query_len, self.heads, self.head_dim).transpose(1, 2)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).transpose(1, 2).contiguous().view(N, query_len, self.heads * self.head_dim)
        
        out = self.fc_out(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion=2, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size * expansion)
        self.fc2 = nn.Linear(embed_size * expansion, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TabTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, expansion=2, dropout=0.1):
        super(TabTransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask = None):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        x = self.dropout(x)
        ff_out = self.ff(x)
        out = self.norm2(ff_out + x)
        out = self.dropout(out)
        return out

class FTTransformer(nn.Module):
    def __init__(self, num_classes, num_categorical, num_numerical, embedding_dim=16, embed_size=384, layers=8, heads=8, dropout=0.1):  # depth를 layers로 변경
        super(FTTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_categorical = num_categorical
        self.num_numerical = num_numerical

        self.categorical_embeddings = nn.ModuleList([nn.Embedding(num_categories, embedding_dim) for num_categories in num_categorical])
        
        self.numerical_layer = nn.Linear(num_numerical, embedding_dim)

        self.concat_layer = nn.Linear(len(num_categorical) * embedding_dim + embedding_dim, embed_size)

        self.transformer_blocks = nn.ModuleList([TabTransformerBlock(embed_size, heads, dropout=dropout) for _ in range(layers)]) # TabTransformerBlock을 여러 개 쌓기
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, 2, dropout)
        self.dropout = nn.Dropout(dropout)

        self.dense_relu = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(128, num_classes)


    def forward(self, categorical_data, numerical_data):
        # Categorical embeddings
        categorical_embeddings = [embedding(categorical_data[:, i]) for i, embedding in enumerate(self.categorical_embeddings)]
        categorical_embeddings = torch.cat(categorical_embeddings, dim=-1)

        # Numerical layer
        numerical_output = self.numerical_layer(numerical_data)

        # Combine categorical and numerical data
        x = torch.cat([categorical_embeddings, numerical_output], dim=-1)
        x = self.concat_layer(x) # Apply the concatenation layer

        # Reshape for the Multi-Head Attention Layer
        x = x.unsqueeze(1)  # Add a sequence dimension (batch_size, 1, 319)

        # Transformer block (simplified - only one block as per the diagram)
        for block in self.transformer_blocks:
            x = block(x, x, x)

        x = x.squeeze(1)

        x = self.dense_relu(x)
        x = self.output_layer(x)
        return x


    

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import gc

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train_model(model:FTTransformer, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device="cuda"):
    reduce_lr = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

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

        # 🔥 Memory cleanup after validation
        clear_memory()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f},", 
              f"Validation Loss: {val_loss / len(val_loader):.4f}, AUCPR: {aucpr_val:.4f},",
              f"PR AUC: {pr_auc_val:.4f}, ROC-AUC: {roc_auc_val:.4f}")

        # Reduce LR if validation loss does not improve
        reduce_lr.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")