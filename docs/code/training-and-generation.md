# 训练循环与温度采样 (generate)

在构建完模型网络后，系统需要两套核心机制：
1. **训练循环 (`train_steps`)**：通过梯度更新降低预测误差；
2. **文本生成 (`generate`)**：使用自回归循环与多项分布采样逐字续写。

---

## 🔁 训练函数 `train_steps()` 详解

```python
# train.py L46-L79
def train_steps(model, data, steps=250, block_size=32, batch_size=16, learning_rate=0.003):
    if data.numel() <= block_size + 1:
        raise ValueError('Corpus must be longer than block_size + 1')
    if block_size != model.block_size:
        raise ValueError('block_size must match model')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def batch():
        starts = torch.randint(data.numel() - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in starts])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
        return x, y

    # 固定一个评估批次，用以量化对比训练前后的绝对损失变化
    x_eval, y_eval = batch()
    def eval_loss():
        model.eval()
        with torch.no_grad():
            logits = model(x_eval)
            result = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y_eval.reshape(-1)).item()
        return result

    initial = eval_loss()
    model.train()
    for _ in range(steps):
        x, y = batch()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        optimizer.zero_grad(set_to_none=True) # 显存更友好的清空梯度
        loss.backward()                       # 反向传播计算梯度
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度防暴走裁剪
        optimizer.step()                      # 更新网络权重
        
    return [initial, eval_loss()]
```

### 关键工程亮点
- **`set_to_none=True`**：在 PyTorch 中，将梯度置为 `None` 比写 `0.0` 更加轻量，减少无用的内存写操作。
- **固定评估批次 `x_eval, y_eval`**：在相同的测试基准上客观比对初态损失与终态损失。

---

## ✍️ 自回归生成循环 `generate()` 详解

自回归模型在生成时，是一步只产生一个 Token 的：

```python
# train.py L81-L91
def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. 窗口截断：只取最后 block_size 个 token 输入模型
            context = tokens[:, -model.block_size:]
            # 2. 前向传播：只取最后一个时间步的 logits
            logits = model(context)[:, -1] / temperature
            # 3. 概率转换与多项式随机采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # 4. 拼接回序列末尾，进入下一步循环
            tokens = torch.cat((tokens, next_token), dim=1)
            
    return tokenizer.decode(tokens[0].tolist())
```

### 🌡️ 温度系数 (Temperature) 的数学原理

在上述代码中，`logits / temperature` 扮演了关键角色：

$$P(x_i) = \frac{\exp(\frac{z_i}{T})}{\sum_j \exp(\frac{z_j}{T})}$$

- **当 $T \to 0$（低温采样）**：
  - 差距被无限放大，概率最大的候选词概率接近 100%（退化为贪心搜索 Greedy Search），输出非常机械、确定。
- **当 $T = 1.0$（基准分布）**：
  - 保持模型原生预测的概率分布。
- **当 $T > 1.0$（高温采样）**：
  - 所有字符的概率被“抹平”，平滑度增高，模型更有可能采到罕见字，多样性增加，但过高容易产生语法混乱。

### 🌊 流式生成 (Streaming) 的实现

在 `chat.py` 中，生成逻辑被改写为了生成器（Generator）：
```python
# chat.py L10-L19
def stream_generate(model, tokenizer, prompt, max_new_tokens=60, temperature=0.8):
    ...
    for _ in range(max_new_tokens):
        ...
        yield tokenizer.chars[next_token.item()]
```
外部循环通过 `yield` 逐字接收字符，结合终端 `sys.stdout.flush()` 和微小的休眠时间，完美还原类似 ChatGPT 逐字输出的打字机交互体验。
