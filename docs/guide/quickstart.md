# 快速上手复现

只需几行简单的命令，你就可以在本地快速配置环境、运行单元测试并完成一次完整的基座模型预训练实验。

## 📋 环境准备

- **操作系统**: macOS / Linux / Windows (WSL 推荐)
- **Python 版本**: 推荐 Python 3.10 或 3.11
- **算力要求**: 普通 CPU 即可，内存消耗极低（< 200MB）

---

## 🛠️ 第一步：创建虚拟环境并安装依赖

在项目根目录下打开终端，执行以下命令：

```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活虚拟环境 (macOS/Linux)
source .venv/bin/activate
# Windows PowerShell 请使用: .venv\Scripts\Activate.ps1

# 3. 安装纯 CPU 版本的 PyTorch 与 NumPy (极快，无需下载庞大 CUDA 包)
pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1'
pip install 'numpy==2.2.6'
```

---

## 🧪 第二步：运行单元测试

在动手训练之前，强烈推荐先运行单元测试，确保分词器、注意力掩码与梯度下降逻辑全部正常：

```bash
python -m unittest -v test_train.py
```

预期输出示例：
```text
test_command_saves_checkpoint_and_sample (test_train.ExperimentTests) ... ok
test_future_characters_do_not_change_previous_predictions (test_train.ModelTests) ... ok
test_training_reduces_loss_and_generates_new_tokens (test_train.ModelTests) ... ok
test_round_trip_chinese_text (test_train.TokenizerTests) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.452s

OK
```

其中特别注意 `test_future_characters_do_not_change_previous_predictions`，它严格保证了模型在预测前几个位置时，绝对不会被未来字符所影响（因果因果关系）。

---

## 🚀 第三步：运行从零预训练实验

执行 `experiment.py`，模型将随机初始化权重，并在 `data/corpus.txt` 上进行 600 次梯度更新：

```bash
python experiment.py --steps 600 --out result --prompt '问题：'
```

运行完成后，终端将打印训练报告 JSON，并将权重保存至 `result/model.pt`：

```json
{
  "initial_train_loss": 5.451095104217529,
  "final_train_loss": 0.08560040593147278,
  "training_steps": 600,
  "parameters": 26727,
  "vocab_size": 199,
  "corpus_characters": 411,
  "sample": "问题：下雨后地面为什么湿？回答：雨水落在地面上，所以地面变湿。\n问题：模型学习？回答：雨水落在地面上，所以地面变湿。\n问题：学习新知识该怎么做？回",
  "note": "仅是小语料训练集损失，不代表泛化能力"
}
```

可以看到，损失（Cross Entropy Loss）从初始的 **5.45** 迅速降到了 **0.08**，模型成功收敛！

---

## 🖨️ 第四步：离线生成采样

使用 `sample.py`，指定不同的初始 Prompt 和生成长度：

```bash
python sample.py --model result/model.pt --prompt '雨停了，' --length 50
```

输出示例：
```text
雨停了，院子里的石板还有水光。小猫沿着墙边走，闻了闻一片落叶。
清晨，学生打开书本。
```

---

## ⚙️ 实验常用参数说明

运行 `python experiment.py --help` 可查看所有可用配置：

| 参数 | 类型 | 默认值 | 作用说明 |
| :--- | :--- | :--- | :--- |
| `--corpus` | Path | `data/corpus.txt` | 训练使用的文本语料文件路径 |
| `--out` | Path | `result` | 模型权重及结果 JSON 的输出目录 |
| `--steps` | int | `350` | 梯度更新训练步数 |
| `--block-size` | int | `32` | 模型的最大上下文滑动窗口长度 |
| `--width` | int | `32` | 隐藏层向量维度（Embedding 维度） |
| `--prompt` | str | `'问题：'` | 训练完成后立即测试采样的前缀提示词 |
