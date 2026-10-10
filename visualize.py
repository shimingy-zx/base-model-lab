"""Visualize TinyTransformer model checkpoint weights, attention, and embeddings."""
import argparse
import json
import webbrowser
from pathlib import Path
import torch
import torch.nn.functional as F
from train import TinyTransformer, CharTokenizer


def extract_data(model, tokenizer, prompt, block_size, width, heads):
    model.eval()

    # 1. Parameter breakdown
    param_counts = {
        'Token Embedding': sum(p.numel() for p in model.token_embedding.parameters()),
        'Position Embedding': sum(p.numel() for p in model.position_embedding.parameters()),
        'LayerNorm 1': sum(p.numel() for p in model.norm1.parameters()),
        'Multihead Attention': sum(p.numel() for p in model.attention.parameters()),
        'LayerNorm 2': sum(p.numel() for p in model.norm2.parameters()),
        'Feed Forward (MLP)': sum(p.numel() for p in model.feed_forward.parameters()),
        'Final LayerNorm': sum(p.numel() for p in model.final_norm.parameters()),
        'Output Projection': sum(p.numel() for p in model.output.parameters()),
    }
    total_params = sum(param_counts.values())

    # 2. Attention weights for prompt
    tokens_list = [c for c in prompt if c in tokenizer.stoi]
    if not tokens_list:
        fallback = '陆向谦实验室' if all(c in tokenizer.stoi for c in '陆向谦实验室') else ''.join(tokenizer.chars[:6])
        tokens_list = list(fallback)
    tokens_list = tokens_list[:block_size]
    clean_prompt = ''.join(tokens_list)

    tokens = torch.tensor([tokenizer.encode(clean_prompt)])
    length = tokens.shape[1]
    positions = torch.arange(length)
    hidden = model.token_embedding(tokens) + model.position_embedding(positions)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    norm = model.norm1(hidden)
    _, attn_weights = model.attention(norm, norm, norm, attn_mask=mask, need_weights=True, average_attn_weights=False)
    # attn_weights shape: [1, heads, length, length]
    attn_heads = [attn_weights[0, h].detach().tolist() for h in range(heads)]
    avg_attn = attn_weights[0].mean(dim=0).detach().tolist()

    # 3. Position embedding cosine similarity
    pos_weights = F.normalize(model.position_embedding.weight.detach(), p=2, dim=1)
    pos_sim = torch.mm(pos_weights, pos_weights.t()).tolist()

    # 4. Token embedding 2D PCA projection
    emb = model.token_embedding.weight.detach()
    U, S, V = torch.pca_lowrank(emb, q=2)
    coords_2d = torch.matmul(emb - emb.mean(dim=0), V[:, :2])
    # Normalize coords to [-100, 100] for comfortable canvas scaling
    c_min = coords_2d.min(dim=0).values
    c_max = coords_2d.max(dim=0).values
    scaled = ((coords_2d - c_min) / (c_max - c_min + 1e-6) * 180 - 90).tolist()

    vocab_points = []
    for idx, char in enumerate(tokenizer.chars):
        vocab_points.append({
            'char': char,
            'id': idx,
            'x': round(scaled[idx][0], 2),
            'y': round(scaled[idx][1], 2),
        })

    return {
        'vocab_size': len(tokenizer.chars),
        'block_size': block_size,
        'width': width,
        'heads': heads,
        'total_params': total_params,
        'param_counts': param_counts,
        'prompt': clean_prompt,
        'prompt_tokens': tokens_list,
        'attn_heads': attn_heads,
        'avg_attn': avg_attn,
        'pos_sim': pos_sim,
        'vocab_points': vocab_points,
    }


def generate_html(data):
    json_data = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TinyTransformer 模型权重与注意力可视化</title>
  <style>
    :root {{
      --bg: #0a0e17;
      --card-bg: #111827;
      --card-border: #1f293d;
      --accent-indigo: #6366f1;
      --accent-cyan: #06b6d4;
      --accent-pink: #ec4899;
      --accent-emerald: #10b981;
      --text-main: #f3f4f6;
      --text-sub: #9ca3af;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text-main);
      font-family: var(--font);
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 700;
      background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .badges {{
      display: flex;
      gap: 10px;
    }}
    .badge {{
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #c7d2fe;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 500;
    }}
    .grid-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
    }}
    .metric-title {{
      font-size: 13px;
      color: var(--text-sub);
      margin-bottom: 6px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
      color: #fff;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .card-title {{
      font-size: 18px;
      font-weight: 600;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      background: #0f172a;
      padding: 4px;
      border-radius: 8px;
    }}
    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-sub);
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: var(--accent-indigo);
      color: #fff;
      box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
    }}
    /* Heatmap Styles */
    .heatmap-wrap {{
      overflow-x: auto;
      padding: 10px 0;
    }}
    .heatmap-table {{
      border-collapse: separate;
      border-spacing: 3px;
      margin: 0 auto;
    }}
    .heatmap-table th {{
      padding: 8px;
      font-size: 14px;
      font-weight: 600;
      color: var(--accent-cyan);
      text-align: center;
      min-width: 44px;
    }}
    .heatmap-table th.row-header {{
      color: var(--accent-pink);
      text-align: right;
      padding-right: 12px;
    }}
    .cell {{
      width: 44px;
      height: 44px;
      border-radius: 6px;
      text-align: center;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s, outline 0.15s;
      position: relative;
    }}
    .cell:hover {{
      transform: scale(1.18);
      z-index: 10;
      outline: 2px solid #fff;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    .tooltip {{
      position: fixed;
      background: #1e293b;
      border: 1px solid #334155;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 12px;
      color: #f8fafc;
      pointer-events: none;
      z-index: 1000;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      display: none;
      line-height: 1.4;
    }}
    /* Embedding Scatter */
    .canvas-container {{
      position: relative;
      background: #090d16;
      border-radius: 8px;
      border: 1px solid #1e293b;
      height: 480px;
      overflow: hidden;
    }}
    #scatter-canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }}
    .scatter-controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .search-input {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 6px 12px;
      color: #fff;
      font-size: 13px;
      width: 220px;
    }}
    .search-input:focus {{
      outline: none;
      border-color: var(--accent-indigo);
    }}
    /* Parameter breakdown */
    .param-bar-wrap {{
      display: flex;
      height: 20px;
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 16px;
      background: #1e293b;
    }}
    .param-segment {{
      height: 100%;
      transition: opacity 0.2s;
    }}
    .param-segment:hover {{
      opacity: 0.8;
    }}
    .param-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
    }}
    .param-item {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(15, 23, 42, 0.6);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .param-legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 8px;
    }}
    .notice {{
      font-size: 13px;
      color: var(--text-sub);
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="tooltip" id="tooltip"></div>
  <div class="container">
    <header>
      <div>
        <h1>TinyTransformer 内部权重与注意力全景可视化</h1>
        <div class="notice">基于纯 PyTorch 自研微型 Decoder-Only 架构，单层单向自回归模型</div>
      </div>
      <div class="badges">
        <span class="badge">CPU 原生训练</span>
        <span class="badge">参数量: <span id="badge-params">-</span></span>
        <span class="badge">词表大小: <span id="badge-vocab">-</span></span>
      </div>
    </header>

    <!-- Metrics -->
    <div class="grid-metrics">
      <div class="metric-card">
        <div class="metric-title">总参数量 (Parameters)</div>
        <div class="metric-value" id="val-params" style="color: #818cf8;">-</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">字符表大小 (Vocab Size)</div>
        <div class="metric-value" id="val-vocab" style="color: #38bdf8;">-</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">上下文窗口 (Block Size)</div>
        <div class="metric-value" id="val-block" style="color: #34d399;">-</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">隐层向量维度 (Width / d_model)</div>
        <div class="metric-value" id="val-width" style="color: #f472b6;">-</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">注意力头数 (Attention Heads)</div>
        <div class="metric-value" id="val-heads" style="color: #fbbf24;">-</div>
      </div>
    </div>

    <!-- 1. Self-Attention Heatmap -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>🧠 因果自注意力热力图 (Self-Attention Maps)</span>
        </div>
        <div class="tabs" id="attn-tabs">
          <button class="tab-btn active" onclick="switchHead(0)">Head 0</button>
          <button class="tab-btn" onclick="switchHead(1)">Head 1</button>
          <button class="tab-btn" onclick="switchHead(-1)">多头平均 (Avg)</button>
        </div>
      </div>
      <div class="notice" style="margin-bottom: 16px;">
        测试提示词：<strong style="color: var(--accent-cyan);" id="prompt-str"></strong>。
        横轴为被关注的历史字符，纵轴为当前自回归生成的字符。观察右上角的因果遮罩（全0）与不同注意头的注意力分布：
      </div>
      <div class="heatmap-wrap">
        <table class="heatmap-table" id="attn-table"></table>
      </div>
    </div>

    <!-- 2. Token Embedding 2D PCA -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>🌌 字符嵌入语义空间 (Token Embedding PCA 2D 投影)</span>
        </div>
        <div class="scatter-controls">
          <input type="text" id="char-search" class="search-input" placeholder="输入汉字搜索定位（如：陆、大、清）..." oninput="onSearchChar(this.value)">
        </div>
      </div>
      <div class="notice" style="margin-bottom: 12px;">
        将 426 个汉字的 32 维隐层向量通过无监督 PCA 降维至 2D 平面。鼠标悬停可查看汉字与 Token ID，搜索框可高亮定位指定字符。
      </div>
      <div class="canvas-container">
        <canvas id="scatter-canvas"></canvas>
      </div>
    </div>

    <!-- 3. Position Embedding Cosine Similarity -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>📏 位置编码相似度热力图 (Position Embedding Cosine Similarity)</span>
        </div>
      </div>
      <div class="notice" style="margin-bottom: 16px;">
        展示 32 个位置向量之间的余弦相似度（[-1, 1]）。观察模型是否自发学到了“临近位置相似度高、相对距离增大相似度平滑衰减”的先验结构：
      </div>
      <div class="heatmap-wrap">
        <canvas id="pos-canvas" width="640" height="320" style="display:block; margin: 0 auto; border-radius: 8px; border: 1px solid #1e293b;"></canvas>
      </div>
    </div>

    <!-- 4. Parameter Breakdown -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>📊 各层参数量精细构成 (Parameter Breakdown)</span>
        </div>
      </div>
      <div class="param-bar-wrap" id="param-bar"></div>
      <div class="param-grid" id="param-grid"></div>
    </div>
  </div>

  <script>
    const data = {json_data};
    let currentHead = 0;
    const tooltip = document.getElementById('tooltip');

    // Populate Metrics
    document.getElementById('badge-params').textContent = data.total_params.toLocaleString();
    document.getElementById('badge-vocab').textContent = data.vocab_size;
    document.getElementById('val-params').textContent = data.total_params.toLocaleString();
    document.getElementById('val-vocab').textContent = data.vocab_size;
    document.getElementById('val-block').textContent = data.block_size;
    document.getElementById('val-width').textContent = data.width;
    document.getElementById('val-heads').textContent = data.heads;
    document.getElementById('prompt-str').textContent = '"' + data.prompt + '"';

    // 1. Attention Heatmap Render
    function renderAttention() {{
      const table = document.getElementById('attn-table');
      table.innerHTML = '';
      const tokens = data.prompt_tokens;
      const matrix = currentHead === -1 ? data.avg_attn : data.attn_heads[currentHead];

      // Header row
      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      headerRow.appendChild(document.createElement('th')); // corner
      tokens.forEach(tok => {{
        const th = document.createElement('th');
        th.textContent = tok;
        headerRow.appendChild(th);
      }});
      thead.appendChild(headerRow);
      table.appendChild(thead);

      // Body rows
      const tbody = document.createElement('tbody');
      tokens.forEach((rowTok, r) => {{
        const tr = document.createElement('tr');
        const rth = document.createElement('th');
        rth.className = 'row-header';
        rth.textContent = rowTok;
        tr.appendChild(rth);

        tokens.forEach((colTok, c) => {{
          const td = document.createElement('td');
          td.className = 'cell';
          const val = matrix[r][c];
          
          if (c > r) {{
            // Causal mask area
            td.style.background = '#0a0f1d';
            td.style.color = '#334155';
            td.textContent = '0';
          }} else {{
            const pct = (val * 100).toFixed(1);
            const intensity = Math.min(1, val * 1.5);
            td.style.background = `rgba(99, 102, 241, ${{Math.max(0.08, intensity)}})`;
            td.style.color = intensity > 0.45 ? '#ffffff' : '#94a3b8';
            td.textContent = pct + '%';
          }}

          td.onmouseenter = (e) => {{
            tooltip.style.display = 'block';
            tooltip.innerHTML = `<strong>当前字符：</strong>「${{rowTok}}」(位置 ${{r}})<br>` +
                                `<strong>关注字符：</strong>「${{colTok}}」(位置 ${{c}})<br>` +
                                `<strong>注意力权重：</strong>${{(val * 100).toFixed(2)}}%`;
            moveTooltip(e);
          }};
          td.onmousemove = moveTooltip;
          td.onmouseleave = () => {{ tooltip.style.display = 'none'; }};
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});
      table.appendChild(tbody);
    }}

    function switchHead(headIdx) {{
      currentHead = headIdx;
      const btns = document.querySelectorAll('#attn-tabs .tab-btn');
      btns.forEach((btn, i) => {{
        if (headIdx === -1 && i === 2) btn.classList.add('active');
        else if (headIdx === i) btn.classList.add('active');
        else btn.classList.remove('active');
      }});
      renderAttention();
    }}

    function moveTooltip(e) {{
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY + 14) + 'px';
    }}

    // 2. Token Embedding Canvas
    let searchHighlight = '';
    const canvas = document.getElementById('scatter-canvas');
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      drawScatter();
    }}

    function drawScatter() {{
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Axes
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
      ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
      ctx.stroke();

      const padding = 60 * window.devicePixelRatio;
      const scaleX = (w - padding * 2) / 200;
      const scaleY = (h - padding * 2) / 200;

      data.vocab_points.forEach(pt => {{
        const cx = w / 2 + pt.x * scaleX;
        const cy = h / 2 - pt.y * scaleY;
        const isMatch = searchHighlight && pt.char === searchHighlight;

        if (isMatch) {{
          ctx.beginPath();
          ctx.arc(cx, cy, 14 * window.devicePixelRatio, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(236, 72, 153, 0.4)';
          ctx.fill();
          ctx.strokeStyle = '#ec4899';
          ctx.lineWidth = 2 * window.devicePixelRatio;
          ctx.stroke();
        }}

        // Dot
        ctx.beginPath();
        ctx.arc(cx, cy, (isMatch ? 5 : 3) * window.devicePixelRatio, 0, Math.PI * 2);
        ctx.fillStyle = isMatch ? '#ec4899' : '#38bdf8';
        ctx.fill();

        // Char text
        ctx.font = `${{(isMatch ? 16 : 11) * window.devicePixelRatio}}px sans-serif`;
        ctx.fillStyle = isMatch ? '#ffffff' : '#94a3b8';
        ctx.fillText(pt.char, cx + 5 * window.devicePixelRatio, cy - 5 * window.devicePixelRatio);
      }});
    }}

    function onSearchChar(val) {{
      searchHighlight = val.trim();
      drawScatter();
    }}

    // Canvas Hover
    canvas.addEventListener('mousemove', (e) => {{
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * window.devicePixelRatio;
      const my = (e.clientY - rect.top) * window.devicePixelRatio;
      const padding = 60 * window.devicePixelRatio;
      const scaleX = (canvas.width - padding * 2) / 200;
      const scaleY = (canvas.height - padding * 2) / 200;

      let found = null;
      for (const pt of data.vocab_points) {{
        const cx = canvas.width / 2 + pt.x * scaleX;
        const cy = canvas.height / 2 - pt.y * scaleY;
        const dist = Math.hypot(mx - cx, my - cy);
        if (dist < 12 * window.devicePixelRatio) {{
          found = pt;
          break;
        }}
      }}

      if (found) {{
        tooltip.style.display = 'block';
        tooltip.innerHTML = `<strong>字符：</strong>「${{found.char}}」<br>` +
                            `<strong>Token ID：</strong>${{found.id}}<br>` +
                            `<strong>PCA 坐标：</strong>(${{found.x}}, ${{found.y}})`;
        moveTooltip(e);
      }} else {{
        tooltip.style.display = 'none';
      }}
    }});
    canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

    // 3. Position Similarity Canvas
    function drawPosSimilarity() {{
      const pCanvas = document.getElementById('pos-canvas');
      const pCtx = pCanvas.getContext('2d');
      const size = data.block_size;
      const w = pCanvas.width;
      const h = pCanvas.height;
      const cellSize = Math.min(w, h) / size;
      const offsetX = (w - cellSize * size) / 2;

      pCtx.clearRect(0, 0, w, h);
      for (let r = 0; r < size; r++) {{
        for (let c = 0; c < size; c++) {{
          const sim = data.pos_sim[r][c]; // -1 to 1
          const norm = (sim + 1) / 2; // 0 to 1
          // Color map: cool blue to bright indigo
          pCtx.fillStyle = `rgb(${{Math.floor(20 + norm * 80)}}, ${{Math.floor(30 + norm * 70)}}, ${{Math.floor(60 + norm * 180)}})`;
          pCtx.fillRect(offsetX + c * cellSize, r * cellSize, cellSize - 1, cellSize - 1);
        }}
      }}
    }}

    // 4. Parameter Breakdown
    const colors = ['#818cf8', '#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#a78bfa', '#f87171', '#4ade80'];
    const barWrap = document.getElementById('param-bar');
    const gridWrap = document.getElementById('param-grid');
    let cIdx = 0;

    for (const [name, count] of Object.entries(data.param_counts)) {{
      const color = colors[cIdx % colors.length];
      const pct = ((count / data.total_params) * 100).toFixed(1);

      // Segment
      const seg = document.createElement('div');
      seg.className = 'param-segment';
      seg.style.width = pct + '%';
      seg.style.background = color;
      seg.title = `${{name}}: ${{count.toLocaleString()}} (${{pct}}%)`;
      barWrap.appendChild(seg);

      // Item
      const item = document.createElement('div');
      item.className = 'param-item';
      item.innerHTML = `<div><span class="param-legend-dot" style="background:${{color}}"></span>${{name}}</div>` +
                       `<div><strong>${{count.toLocaleString()}}</strong> <span style="color:#64748b; font-size:11px;">(${{pct}}%)</span></div>`;
      gridWrap.appendChild(item);
      cIdx++;
    }}

    // Initial renders
    renderAttention();
    drawPosSimilarity();
    window.addEventListener('resize', resizeCanvas);
    setTimeout(resizeCanvas, 50);
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='可视化已训练的小模型权重、注意力矩阵与词向量')
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('result') / 'model.pt')
    parser.add_argument('--prompt', default='陆向谦实验室')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('result') / 'visualize.html')
    parser.add_argument('--no-open', action='store_true', help='生成后不自动在浏览器中打开')
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f'模型权重文件不存在：{args.model}，请先运行 experiment.py 训练模型。')

    torch.set_num_threads(1)
    print(f'正在加载模型权重：{args.model} ...')
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    tokenizer = CharTokenizer(''.join(checkpoint['chars']))
    model = TinyTransformer(len(tokenizer.chars), checkpoint['block_size'], checkpoint['width'], checkpoint['heads'])
    model.load_state_dict(checkpoint['state_dict'])

    print(f'正在计算注意力权重与向量空间 (Prompt: "{args.prompt}")...')
    data = extract_data(model, tokenizer, args.prompt, checkpoint['block_size'], checkpoint['width'], checkpoint['heads'])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_html(data)
    args.out.write_text(html_content, encoding='utf-8')

    print('\n==================== 可视化报告已生成 ====================')
    print(f'总参数量：{data["total_params"]:,}')
    print(f'词表大小：{data["vocab_size"]} 字符')
    print(f'测试提示词："{data["prompt"]}" ({len(data["prompt_tokens"])} tokens)')
    print(f'报告路径：file://{args.out.resolve()}')
    print('=========================================================\n')

    if not args.no_open:
        print('正在启动默认浏览器打开可视化页面...')
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == '__main__':
    main()
