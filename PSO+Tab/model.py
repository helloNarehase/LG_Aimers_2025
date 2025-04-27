import torch 
from torch import nn
import torch.nn.functional as F

import math

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class FFN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FFN, self).__init__()
        
        # W_K, W_V는 각각 입력에 대한 변환을 위한 파라미터
        self.W_K = nn.Linear(input_dim, hidden_dim, bias=False)  # K에 해당하는 가중치
        self.W_V = nn.Linear(input_dim, hidden_dim, bias=False)  # V에 해당하는 가중치
        self.W_O = nn.Linear(hidden_dim, input_dim, bias=False)  # V에 해당하는 가중치

    def forward(self, x):
        return self.W_O(F.silu(self.W_V(x)) * self.W_K(x))


class Attention(nn.Module):
    def __init__(self, n_heads, n_kv_heads, dim):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads


        self.wq = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
    
class LinearTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_prob = 0.5):
        super().__init__()
        self.lattn = Attention(num_heads, 4, embed_dim)
        self.drop1 = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(embed_dim)  # Fully Connected 레이어에 대한 Batch Normalization


        self.ffn = FFN(embed_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout_prob)
        self.ln2 = nn.LayerNorm(embed_dim)  # Fully Connected 레이어에 대한 Batch Normalization
    
    def forward(self, x):
        u = self.drop1(self.ln1(self.lattn(x))) + x
        return self.drop2(self.ln2((self.ffn(u) + u)))

class Enc(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim=64, n_layers = 1, dropout_prob=0.2, ):
        super(Enc, self).__init__()
        
        # Conv1D 레이어
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(32)  # Conv1D 출력에 대한 Batch Normalization
        # self.cv = nn.Linear(32, embed_dim)

        # Dropout 추가
        self.dropout = nn.Dropout(dropout_prob)

        
        # MLP 레이어
        self.ffn = nn.ModuleList([LinearTransformerBlock(embed_dim, num_heads, hidden_dim, dropout_prob) for i in range(n_layers)])
        self.head = nn.Linear(embed_dim * input_dim, 1)  # num_classes=1로 설정 (이진 분류)

    def forward(self, x):
        # Conv1D 레이어
        x = x.unsqueeze(1)  # Conv1D는 (batch_size, channels, length) 형식을 요구하므로 차원 추가
        x = self.conv1d(x)
        # x = self.bn1(x)  # Batch Normalization
        x = F.relu(x)
        
        # Dropout 적용
        x = self.dropout(x)
        x = x.transpose(-1, -2)

        for l in self.ffn:
            x = l(x)
            x = self.dropout(x)
        x = x.flatten(1)

        return self.head(x), x
    
    def set_last_layer_weights(self, new_weights):
        """사용자가 마지막 레이어의 가중치를 임의로 설정하는 메서드"""
        with torch.no_grad():  # 가중치 수정시 그래디언트 계산을 하지 않도록 설정
            self.head.weight.data = new_weights

    def set_last_layer_weights_and_bias(self, new_bias):
        """사용자가 마지막 레이어의 bias를 임의로 설정하는 메서드"""
        with torch.no_grad():
            self.head.bias.data = new_bias
        