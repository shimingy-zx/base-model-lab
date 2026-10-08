---
layout: home

hero:
  name: "Base Model Lab"
  text: "从零训练迷你基座模型"
  tagline: "从随机参数、因果自注意力到自回归生成：纯 CPU 几十秒跑通 LLM 预训练全闭环"
  actions:
    - theme: brand
      text: 快速上手 →
      link: /guide/quickstart
    - theme: alt
      text: 核心原理与架构
      link: /architecture/transformer
    - theme: alt
      text: 终端流式体验
      link: /guide/chat

features:
  - icon: 🔬
    title: 真正的从零训练 (From Scratch)
    details: 不加载任何预训练权重，不依赖任何第三方大模型 API，参数完全从高斯随机分布初始化，见证模型从乱码到学会组词造句。
  - icon: ⚡
    title: 极轻量但完整的 Transformer
    details: 约 2.7 万参数单层因果自注意力解码器（Decoder-Only），涵盖 Embedding、Causal Mask、Multi-Head Attention、FFN 与 LayerNorm。
  - icon: 💻
    title: 零显卡门槛 (纯 CPU 秒级跑通)
    details: 仅依赖 PyTorch 与 NumPy，无需高昂的 GPU/TPU 算力资源，任何普通电脑在 30 秒至 1 分钟内即可完成预训练实验。
  - icon: 💬
    title: 终端打字机流式交互 (chat.py)
    details: 内置类似 ChatGPT 的流式输出终端交互工具，支持实时温度调节、生成长度控制、字符表检查与未知字符拦截。
  - icon: 🛡️
    title: 严谨的数学验证与单测套件
    details: 提供针对因果掩码防未来偷看、分词编解码一致性、损失函数单调下降与端到端实验的全覆盖单元测试。
  - icon: 🧭
    title: 深入浅出的演进路线
    details: 全面剖析从玩具级 2.7 万参数模型到工业级 LLaMA、GPT-4 等千亿参数基座模型的技术升级路径与工程挑战。
---

<style>
:root {
  --vp-c-brand-1: #6366f1;
  --vp-c-brand-2: #4f46e5;
  --vp-c-brand-3: #4338ca;
}
</style>

## 30 秒速览训练前后对比

在这个实验中，你将亲眼目睹因果注意力机制如何通过交叉熵与反向传播在语料上调整权重：

```
🎲 训练前（随机权重，Loss ≈ 5.45）：
提示词: 问题：
模型输出: 问题：变落空界阳地想村题发影天水解嫩答化空村蓝落...（毫无意义的符号碰撞）

🎯 训练 600 步后（参数更新完毕，Loss ≈ 0.08）：
提示词: 问题：
模型输出: 问题：下雨后地面为什么湿？回答：雨水落在地面上，所以地面变湿。
```

> **重要认知**：这并非模型拥有了人类常识，而是极简模型完美“记忆并重构”了训练语料中字符序列的统计概率转移规律。通过这个最小实验，你可以透彻看清大语言模型的本质。
