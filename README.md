# 从零训练一个迷你基座模型：动手实验

这不是 ChatGPT 级别的大模型，而是**真实从随机参数开始训练**的字符级、单层、因果注意力 Transformer。适合先理解基座模型的训练闭环。项目未使用任何现成预训练权重，也没有指令微调。

## 先看现成结果

`result/result.json` 记录本机实跑的训练前后损失和生成样例，`result/model.pt` 是实际训练所得权重。示例语料 `data/corpus.txt` 为教学用途自行编写；语料很小，模型会复述/拼接训练句子，**不能当作可靠问答或真正有泛化能力的大模型**。

## 四步理解代码

1. `data/corpus.txt`：准备具有使用权的文本，当前为自编小语料。
2. `CharTokenizer`：把每个字编码为整数编号；这里采用简单字符级，不是工业级分词器。
3. `TinyTransformer`：随机初始化约 2.7 万个参数；注意力遮罩让它只能看当前位置及之前的字符。
4. `train_steps`：遮住下一字让模型猜，计算交叉熵，反向传播调整权重；`sample.py` 用训练后的权重续写。

## 🚀 统一工作流命令行 (`lab.py`)

本项目推荐使用统一入口 `lab.py`，无需在不同脚本间切换：

```bash
# 查看所有子命令说明
python lab.py --help

# 1. 训练基座模型 (可自定义 steps/width/block-size 等)
python lab.py train --steps 600 --prompt '问题：'

# 2. 单次提示词文本续写采样
python lab.py sample --prompt '问题：' --length 60

# 3. 终端打字机流式交互体验
python lab.py chat

# 4. 生成注意力矩阵与嵌入向量可视化 HTML 报告
python lab.py viz --prompt '陆向谦实验室'

# 5. 运行完整单元测试套件
python lab.py test

# 6. 一键端到端全流程闭环 (自动执行: 训练 -> 测试 -> 生成可视化)
python lab.py run

# 7. 启动本地 VitePress 交互式文档预览
python lab.py docs
```

> **注**：原有的独立脚本（`experiment.py`、`sample.py`、`chat.py`、`visualize.py`）依然完全保留并向前兼容，底层实现与单元测试完全不受影响。

## 在 Linux/macOS 上重跑环境配置

推荐 Python 3.11。在项目目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1'
.venv/bin/python -m pip install 'numpy==2.2.6'
.venv/bin/python lab.py test
.venv/bin/python lab.py run
```


## 📖 项目完整技术文档 (VitePress)

项目内置了详尽的 VitePress 交互式文档站点（位于 `docs/` 目录），涵盖网络拓扑、因果掩码数学原理、逐行代码剖析与工业级大模型演进路线：

```bash
# 进入文档目录
cd docs

# 安装文档依赖 (已就绪)
npm install

# 启动本地文档实时预览服务器
npm run dev

# 构建静态文档站点
npm run build
```

## 本机实测

PyTorch 2.7.1 CPU、Python 3.11；600 次参数更新，语料 411 字符，字符表 199 个符号，模型参数 26,727。固定训练样本损失由 5.451 降到 0.0856；4 项单元测试通过。**这是训练样本上的下降，不是独立验证集上的能力证明**。结果与随机数种子、运行环境有关。

## 以后怎么升级

换更大、合法且清洗过的语料；划分训练集/验证集；改用更合理的 tokenizer；增加模型层数和参数；配置 GPU 与分布式训练；用验证损失及独立任务评测泛化。先跑通本项目，再逐项升级，避免一开始就投入昂贵算力。
