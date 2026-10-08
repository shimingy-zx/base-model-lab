# 预训练：Next-Token Prediction

大语言模型是如何从“一堆完全混乱的随机数”成长为能够连贯组织语言的实体的？

答案就在于 **自监督学习（Self-Supervised Learning）** 中的核心任务：**Next-Token Prediction（下一个词预测）**。

---

## 🎯 训练任务构造：错位滑动窗口

无需人工对语料打标签，文本本身就是最好的老师。

给定一段长度为 $N$ 的语料，模型按固定窗口长度（`block_size`，如 32）切出输入样本 $x$，而目标 $y$ 恰好是 $x$ 整体向右偏移 1 个字符的切片：

```text
原始连续文本:  [雨, 停, 了, ，, 院, 子, 里, 的, 石, 板, ...]
输入序列 (x):  [雨, 停, 了, ，, 院, 子, 里, 的, 石]
目标序列 (y):  [停, 了, ，, 院, 子, 里, 的, 石, 板]
```

在 `train.py` 中的具体实现代码如下：

```python
# train.py L54-L58
def batch():
    # 随机采样 batch_size 个起始位置
    starts = torch.randint(data.numel() - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y
```

对每一个时间步 $t$：
- 看到 `x[0]`（雨），模型尝试预测 `y[0]`（停）；
- 看到 `x[0:2]`（雨停），模型尝试预测 `y[1]`（了）；
- 看到 `x[0:3]`（雨停了），模型尝试预测 `y[2]`（，）；
- ……以此类推。

借助 Transformer 的并行计算与因果掩码，这 $T$ 个预测过程在**单次前向传播矩阵运算中同时完成**！

---

## 📉 损失函数：交叉熵损失 (Cross Entropy)

模型输出层计算出每个位置对于词表中所有字符的未归一化打分（Logits）。

交叉熵损失度量的是：**模型给真实出现的下一个字符所分配的概率有多高**。

$$\mathcal{L} = - \frac{1}{B \times T} \sum_{i=1}^{B \times T} \log P(y_i \mid x_{\le i})$$

在 PyTorch 中一行搞定：

```python
# train.py L73
loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
```

### 理论初始损失
在初始阶段，模型参数完全随机，对词表（大小为 $V = 199$）中的所有字符一视同仁，预测概率约为 $1/V$。
因此理论初始损失为：

$$\mathcal{L}_{\text{init}} \approx -\ln\left(\frac{1}{199}\right) = \ln(199) \approx 5.293$$

这与实验实测的 `initial_train_loss = 5.451` 高度吻合！这也验证了模型起初完全处于均匀瞎猜状态。

---

## ⚡ 优化器与训练技巧

### 1. AdamW 优化器
相比于传统 SGD，AdamW 结合了动量（Momentum）和自适应学习率（RMSProp），并正确实施了权重衰减（Weight Decay）。
在小模型训练中，学习率通常设置为较高值（如 `0.003`），以便在数百步内迅速收敛。

### 2. 梯度裁剪 (Gradient Clipping)
```python
# train.py L76
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
自回归模型的注意力机制在训练早期容易出现较大的梯度跳变。将梯度范数截断到 1.0，能够有效防止“梯度爆炸”导致参数被冲乱。
