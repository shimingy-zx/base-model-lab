# 模型架构实现 (TinyTransformer)

在 `train.py` 中，`TinyTransformer` 类继承自 PyTorch 的 `nn.Module`，是核心模型的全部实现（仅 26 行代码）。

---

## 💻 完整代码清单

```python
# train.py L19-L44
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=32, width=32, heads=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(block_size, width)
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width)
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        batch, length = tokens.shape
        if length > self.block_size:
            raise ValueError('Input exceeds block_size')
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        normalized = self.norm1(hidden)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        hidden = hidden + attended
        hidden = hidden + self.feed_forward(self.norm2(hidden))
        return self.output(self.final_norm(hidden))
```

---

## 🧮 26,727 参数量精确数学拆解

很多初学者好奇：2.7 万参数究竟分布在哪些层？让我们逐层精确计算（设 $V=199, W=32, T=32, H=2$）：

| 层级名称 | 对应结构与公式 | 权重参数 (Weights) | 偏置参数 (Biases) | 该层总参数量 |
| :--- | :--- | :--- | :--- | :--- |
| `token_embedding` | $V \times W$ | $199 \times 32 = 6,368$ | 0 | **6,368** |
| `position_embedding` | $T \times W$ | $32 \times 32 = 1,024$ | 0 | **1,024** |
| `norm1` (LayerNorm) | 缩放 $\gamma$ + 偏移 $\beta$ | $32$ | $32$ | **64** |
| `attention` (in_proj) | $3 \times W \times W$ ($Q, K, V$) | $3 \times 32 \times 32 = 3,072$ | $3 \times 32 = 96$ | **3,168** |
| `attention` (out_proj) | $W \times W$ | $32 \times 32 = 1,024$ | $32$ | **1,056** |
| `norm2` (LayerNorm) | 缩放 $\gamma$ + 偏移 $\beta$ | $32$ | $32$ | **64** |
| `feed_forward.0` | $W \to 4W$ | $32 \times 128 = 4,096$ | $128$ | **4,224** |
| `feed_forward.2` | $4W \to W$ | $128 \times 32 = 4,096$ | $32$ | **4,128** |
| `final_norm` (LayerNorm) | 缩放 $\gamma$ + 偏移 $\beta$ | $32$ | $32$ | **64** |
| `output` (Linear) | $W \to V$ | $32 \times 199 = 6,368$ | $199$ | **6,567** |
| **总计 (Total)** | | | | **26,727** |

每个参数都是单精度浮点数（`float32`，占 4 字节），模型权重总文件大小仅约 **104 KB**！

---

## 📐 前向传播张量形状变换流

跟踪张量在每一行代码中的形状（Shape）变化，是理解深度学习网络的最快途径：

```text
tokens: (B, L)
   │
   ├── token_embedding(tokens)         ──> (B, L, 32)
   └── position_embedding(positions)   ──> (L, 32) (广播相加)
   │
hidden: (B, L, 32)
   │
   ├── norm1(hidden)                   ──> (B, L, 32)
   ├── attention(..., attn_mask=mask)  ──> (B, L, 32)
   └── hidden = hidden + attended      ──> (B, L, 32) (残差连接 1)
   │
   ├── norm2(hidden)                   ──> (B, L, 32)
   ├── feed_forward(...)               ──> (B, L, 32)
   └── hidden = hidden + ffn_out       ──> (B, L, 32) (残差连接 2)
   │
   ├── final_norm(hidden)              ──> (B, L, 32)
   └── output(...)                     ──> (B, L, 199)
   │
Logits: (B, L, 199) (最终未归一化输出)
```
