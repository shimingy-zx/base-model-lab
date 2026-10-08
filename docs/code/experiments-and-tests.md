# 实验工具与因果掩码单测

一个优秀的深度学习项目必须具备严谨的工程闭环与自动化质量验证。在 Base Model Lab 中，`experiment.py`、`sample.py` 与 `test_train.py` 构成了完整的工程护城河。

---

## 🔬 自动化实验驱动器 (`experiment.py`)

`experiment.py` 负责把训练过程封装为可参数化、可重现的标准实验流程：

1. **固定随机种子**：通过 `torch.manual_seed(9)` 保证在相同机器环境下实验结果高度可复现；
2. **权重持久化**：不仅保存模型的 `state_dict`，还将 `chars` 字符表、`block_size`、`width`、`heads` 元数据一起打包打包存入 `result/model.pt`；
3. **指标审计输出**：自动生成 `result/result.json`，记录初始损失、终态损失、步数、参数量与样例。

---

## 🧪 单元测试套件 (`test_train.py`)

深度学习代码常常因为“张量维度能跑通但数学逻辑错误（静默失败）”而难以排查。`test_train.py` 设计了 4 个极具针对性的单测用例：

### 1. 分词器互逆测试 (`test_round_trip_chinese_text`)
```python
def test_round_trip_chinese_text(self):
    text = '春天来了，花儿开了。\n'
    tok = CharTokenizer(text)
    self.assertEqual(tok.decode(tok.encode(text)), text)
    self.assertEqual(len(tok.encode(text)), len(text))
```
确保中文编码与解码不出现字符丢失或长度偏移。

---

### 2. 因果自注意力防偷看测试 (关键数学性质)
这是大模型预训练中最核心的断言之一：**输入序列后续字符发生改变时，模型对前序字符的预测输出绝对不能发生任何变化**。

```python
# test_train.py L20-L30
def test_future_characters_do_not_change_previous_predictions(self):
    torch.manual_seed(7)
    model = TinyTransformer(vocab_size=16, block_size=8, width=16, heads=2)
    model.eval()
    
    # 构造两个序列，前 3 个 token 完全一致，但第 4、5 个 token 截然不同
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 8, 9]])
    
    with torch.no_grad():
        logits_a = model(a)
        logits_b = model(b)
        
    self.assertEqual(tuple(logits_a.shape), (1, 5, 16))
    # 严格断言：对前 3 个位置计算得到的 logits 必须严格相等（误差 < 1e-6）
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], rtol=0, atol=1e-6)
```

::: tip 为什么这个测试至关重要？
如果因果掩码 `torch.triu(..., diagonal=1)` 写错或者未生效，模型就会发生“未来信息泄露”（Data Leakage）。模型在前向传播中能够提前“偷看”答案，导致训练损失虽然急速下降，但在自回归逐步生成时彻底崩溃。这个单测彻底杜绝了这种隐蔽 bug。
:::

---

### 3. 损失单调下降与生成测试 (`test_training_reduces_loss_and_generates_new_tokens`)
确保整个优化器、梯度反向传播链路能够正常工作，经过 30 步微小训练后，`eval_loss` 必须小于初始损失，且采样生成具有指定前缀。

---

### 4. CLI 端到端全流程测试 (`test_command_saves_checkpoint_and_sample`)
在临时独立文件夹中拉起子进程执行 `experiment.py` 与 `sample.py`，断言退出码为 0，且生成的权重和报告完整可用。
