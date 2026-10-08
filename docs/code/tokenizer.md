# 字符级分词器 (CharTokenizer)

分词（Tokenization）是自然语言处理的第一道大门，它负责将人类可读的连续文本转换为计算机可以处理的整数数组。

---

## 💻 源码实现剖析

在 `train.py` 中，分词器实现仅有短短 12 行代码：

```python
# train.py L7-L17
class CharTokenizer:
    def __init__(self, text):
        # 1. 提取语料中出现过的所有唯一字符，并按字典序排序
        self.chars = sorted(set(text))
        # 2. 构建字符到整数索引的字典 (String to Index)
        self.stoi = {char: i for i, char in enumerate(self.chars)}

    def encode(self, text):
        # 将文本字符串映射为整数 ID 列表
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        # 将整数 ID 列表反解为原始字符串
        return ''.join(self.chars[i] for i in ids)
```

### 双向映射机制
- **`self.chars` (List[str])**：扮演 `itos`（Index to String）的角色。例如 `self.chars[10]` 返回对应字符。
- **`self.stoi` (Dict[str, int])**：反向查找字典。例如 `self.stoi['雨']` 返回整数 `128`。
- **确定性与无损还原**：通过单元测试 `test_round_trip_chinese_text` 验证：
  $$\text{decode}(\text{encode}(\text{text})) \equiv \text{text}$$

---

## ⚖️ 字符级 vs 现代工业级分词器对比

为什么工业级大模型（如 GPT-4, LLaMA, DeepSeek）不使用纯字符级分词？

| 维度 | 字符级分词 (CharTokenizer) | 子词分词器 (BPE / Byte-Pair Encoding) |
| :--- | :--- | :--- |
| **词表大小 (Vocab Size)** | 极小（几百到几千） | 较大（32,000 ~ 150,000+） |
| **分词粒度** | 每个汉字/标点为一个 Token | 常用词、词根、高频组合为一个 Token（如“人工智能”可能是 1~2 个 Token） |
| **序列长度** | 相同文本对应的 Token 数量长（单字为单位） | Token 序列大幅缩短，提升上下文窗口容量 |
| **未登录词 (OOV)** | 遇到未见字符直接 KeyError | 基于字节级 (Byte-level)，几乎永不 OOV，任何 utf-8 均可编码 |
| **实现与训练复杂度** | 极简，几十行 Python 原生集合运算 | 需要在大规模语料上运行 BPE 合并算法统计频次 |

### 教学实验的权衡选择
在只有约 400 字符的教学场景下：
1. 如果使用几万词表的工业分词器，词嵌入矩阵（$V \times W$）将极其庞大且极度稀疏，绝大多数权重无法得到更新；
2. 字符级分词使词表恰好收敛在 199 个字符，模型参数仅 2.7 万，能够在普通笔记本 CPU 上以极低显存/内存瞬间完成全量训练。
