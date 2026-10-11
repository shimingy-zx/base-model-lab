# 统一命令行与 Makefile 指令 (`lab.py` / `Makefile`)

为了避免学习者在众多独立的脚本文件之间反复切换，**Base Model Lab** 提供了工程化的统一工作流入口：
1. **`lab.py`**：基于 Python `argparse` 构建的统一多子命令入口；
2. **`Makefile`**：提供极简的一键式 `make` 宏命令，开箱即用。

两者完全向前兼容原有的底层独立脚本（`train.py`、`experiment.py`、`sample.py`、`chat.py`、`visualize.py` 与 `test_train.py`）。

---

## 🚀 统一入口：`lab.py`

在项目根目录下，直接执行 `python lab.py` 即可查看所有子命令与操作面板：

```bash
# 查看全局帮助与所有可用子命令
python lab.py --help

# 查看特定子命令的详细参数说明 (以 train 为例)
python lab.py train --help
```

### 常用子命令速查

| 子命令 | 别名 (Alias) | 说明 | 底层调用关系 |
| :--- | :--- | :--- | :--- |
| `train` | `exp`, `experiment` | 从零训练迷你基座模型并导出检查点与指标 | 调用 `experiment.run_experiment()` |
| `sample` | `gen`, `generate` | 基于已有权重进行单次前缀文本续写 | 调用 `sample.run_sample()` |
| `chat` | - | 启动终端打字机流式交互面板 | 调用 `chat.run_chat()` |
| `viz` | `visualize` | 提取注意力矩阵与词向量分布并生成 HTML 可视化报告 | 调用 `visualize.run_visualize()` |
| `test` | - | 自动发现并运行项目全套单元测试 | 基于 `unittest.TestLoader` 扫描 `test_*.py` |
| `docs` | - | 启动或构建本地 VitePress 技术文档站点 | 在 `docs/` 目录下执行 `npm run <action>` |
| `run` | `pipeline`, `all` | 一键端到端全流程闭环（训练 → 测试 → 可视化） | 串联全链路自动化执行 |

---

## 🛠️ `lab.py` 各子命令详解与调参

### 1. `lab.py train` (模型预训练)
从随机权重开始在指定语料上执行因果自注意力模型的梯度反向传播：

```bash
# 默认配置训练 (350 步，上下文 32，隐层维度 32)
python lab.py train

# 自定义调参训练示例
python lab.py train --steps 600 --width 64 --block-size 32 --prompt '问题：'
```

- `--corpus <Path>`：训练语料路径（默认：`data/corpus.txt`）；
- `--out <Path>`：检查点与评估指标保存目录（默认：`result/`）；
- `--steps <int>`：训练迭代步数（默认：`350`）；
- `--block-size <int>`：上下文最大滑动窗口长度（默认：`32`）；
- `--width <int>`：词嵌入与隐藏层维度（默认：`32`）；
- `--prompt <str>`：训练完成后立即测试采样的前缀提示词（默认：`'问题：'`）。

### 2. `lab.py sample` (单次文本续写)
加载已训练保存的模型检查点 `result/model.pt`，进行自回归采样：

```bash
python lab.py sample --prompt '雨停了，' --length 50 --temp 0.6
```

- `--model <Path>`：模型检查点权重文件路径（默认：`result/model.pt`）；
- `--prompt <str>`：输入的起始提示词（必须全部由训练集已见字符组成）；
- `--length <int>`：生成新字符的最大数量（默认：`60`）；
- `--temp <float>`：采样温度系数（默认：`0.8`；越低越确定，越高越发散）。

### 3. `lab.py chat` (终端打字机交互)
进入沉浸式的流式交互终端，支持动态调整参数与字符表检索：

```bash
python lab.py chat --temp 0.7 --length 80
```

详细交互指令（如 `:temp`、`:len`、`:vocab`）请参考 [终端交互体验](/guide/chat)。

### 4. `lab.py viz` (模型权重与注意力可视化)
提取 Transformer 内部的多头注意力掩码权重、位置编码余弦相似度与字符词嵌入的 2D PCA 投影分布，生成自包含的离线交互式 HTML 报告：

```bash
# 生成并在浏览器中自动打开可视化报告
python lab.py viz --prompt '陆向谦实验室'

# 生成但不自动弹出浏览器窗口
python lab.py viz --no-open
```

生成的 HTML 默认位于 `result/visualize.html`，无需任何外部服务器支持，双击即可本地查阅。

### 5. `lab.py test` (单元测试)
运行项目内置的所有数学与逻辑正确性单测（分词器互逆性、因果注意力防偷看、损失单调递减等）：

```bash
# 运行单元测试
python lab.py test

# 显示每个测试用例的详细执行状态
python lab.py test -v
```

### 6. `lab.py docs` (交互式文档服务)
集成 VitePress 文档站点的生命周期管理：

```bash
# 启动本地实时热重载开发服务器 (默认端口 5173)
python lab.py docs dev

# 编译生成静态站点资源文件
python lab.py docs build

# 启动静态编译产物预览服务
python lab.py docs preview
```

### 7. `lab.py run` (一键端到端闭环流水线)
面向学习者的“一键式全自动体验”。执行该命令将自动按顺序完成：
1. **训练基座模型**（随机初始化权重并在语料上训练）；
2. **运行单元测试**（断言因果掩码与数学闭环未受破坏）；
3. **导出可视化报告**（生成 `result/visualize.html`）；
4. **输出生成样例与后续操作指引**。

```bash
python lab.py run --steps 600
```

- `--skip-test`：跳过单测环节直接生成结果；
- `--no-open`：生成可视化报告后不自动调用浏览器打开。

---

## ⚡ 极简方案：`Makefile` 快捷指令

针对熟悉终端与 Make 工具的开发者，根目录的 `Makefile` 对 `lab.py` 的核心指令进行了二次精简封装：

```bash
# 查看所有可用 make 目标与说明
make help

# 1. 训练基座模型 (默认 600 步)
make train

# 2. 单次提示词采样
make sample

# 3. 启动终端打字机流式交互
make chat

# 4. 生成注意力与嵌入可视化报告
make viz

# 5. 运行完整单元测试套件
make test

# 6. 一键端到端流水线 (训练 + 测试 + 可视化)
make run

# 7. 启动本地 VitePress 文档预览
make docs

# 8. 清理临时缓存与编译产物
make clean
```

### 自定义 Python 解释器环境

`Makefile` 中预设了 `PYTHON ?= python3` 变量。如果你希望显式使用虚拟环境中的解释器执行，无需手动 `source .venv/bin/activate`，直接在命令前指定变量即可：

```bash
# 使用虚拟环境的 Python 执行流水线
PYTHON=.venv/bin/python make run

# 使用虚拟环境执行交互终端
PYTHON=.venv/bin/python make chat
```

### 清理缓存 (`make clean`)
执行 `make clean` 将自动清除项目产生的 Python 字节码缓存（`__pycache__`）以及 VitePress 文档的编译产物与缓存目录（`docs/.vitepress/dist`、`docs/.vitepress/cache`），保持代码仓库清爽整洁。
