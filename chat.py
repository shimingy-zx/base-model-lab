"""Terminal interactive generator for the trained mini-transformer."""
import argparse
import sys
import time
from pathlib import Path
import torch
from train import CharTokenizer, TinyTransformer


def stream_generate(model, tokenizer, prompt, max_new_tokens=60, temperature=0.8):
    """自回归流式生成字符"""
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -model.block_size:])[:, -1] / temperature
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
            yield tokenizer.chars[next_token.item()]


def run_chat(model_path=None, length=60, temp=0.8):
    if model_path is None:
        model_path = Path(__file__).with_name('result') / 'model.pt'
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"❌ 找不到模型权重文件: {model_path}")
        print("请先运行实验训练模型: python lab.py train 或 python experiment.py")
        return 1

    torch.set_num_threads(1)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    chars = ''.join(checkpoint['chars'])
    tokenizer = CharTokenizer(chars)
    model = TinyTransformer(len(tokenizer.chars), checkpoint['block_size'],
                            checkpoint['width'], checkpoint['heads'])
    model.load_state_dict(checkpoint['state_dict'])

    max_len = length
    temperature = temp

    print("=" * 60)
    print("🤖 迷你基座语言模型 - 终端交互体验")
    print("=" * 60)

    print(f"• 模型参数量: {sum(p.numel() for p in model.parameters()):,} 个")
    print(f"• 词表大小: {len(tokenizer.chars)} 个字符")
    print(f"• 当前设置: 生成长度={max_len}, 温度={temperature}")
    print("\n💡 指令说明:")
    print("  - 输入提示词直接回车，模型将进行续写")
    print("  - 输入 ':temp 0.5' 调整采样温度 (推荐 0.2 ~ 1.0)")
    print("  - 输入 ':len 80' 调整生成长度")
    print("  - 输入 ':vocab' 查看字符表中的字符")
    print("  - 输入 'exit' 或 'quit' 退出交互")
    print("-" * 60)
    print("💡 推荐体验的开头: '问题：'、'雨停了，'、'清晨，'、'种子'、'文字模型'")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n📝 请输入提示词 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 已退出。")
            break

        if not prompt:
            continue

        if prompt.lower() in ('exit', 'quit', 'q'):
            print("👋 再见！")
            break

        # 动态修改参数
        if prompt.startswith(':temp '):
            try:
                new_t = float(prompt.split()[1])
                if new_t > 0:
                    temperature = new_t
                    print(f"✅ 采样温度已调整为: {temperature}")
                else:
                    print("⚠️ 温度必须大于 0")
            except Exception:
                print("⚠️ 格式错误，用法示例: :temp 0.7")
            continue

        if prompt.startswith(':len '):
            try:
                new_l = int(prompt.split()[1])
                if new_l > 0:
                    max_len = new_l
                    print(f"✅ 生成长度已调整为: {max_len}")
                else:
                    print("⚠️ 长度必须大于 0")
            except Exception:
                print("⚠️ 格式错误，用法示例: :len 80")
            continue

        if prompt == ':vocab':
            print("字符表包含的字符如下:")
            print(' '.join(tokenizer.chars))
            continue

        # 检查字符是否都在词表中
        unknown_chars = [c for c in prompt if c not in tokenizer.stoi]
        if unknown_chars:
            unique_unknown = ''.join(sorted(set(unknown_chars)))
            print(f"⚠️ 提示词中包含未见过的字符: [{unique_unknown}]")
            print("   由于本模型是极小字符级模型，输入的每个字都必须在训练语料中出现过。")
            print("   您可以输入 ':vocab' 查看支持的所有字，或尝试：'问题：' / '雨停了，'")
            continue

        # 流式生成并打印
        print(f"\n🤖 模型输出: {prompt}", end='', flush=True)
        try:
            for char in stream_generate(model, tokenizer, prompt,
                                       max_new_tokens=max_len,
                                       temperature=temperature):
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.015)  # 模拟打字机流式输出体验
            print()
        except Exception as e:
            print(f"\n❌ 生成时发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='在终端交互体验训练好的迷你基座模型')
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('result') / 'model.pt',
                        help='模型权重路径 (默认: result/model.pt)')
    parser.add_argument('--length', type=int, default=60, help='默认生成长度 (字符数)')
    parser.add_argument('--temp', type=float, default=0.8, help='采样温度 (temperature)')
    args = parser.parse_args()
    sys.exit(run_chat(args.model, length=args.length, temp=args.temp) or 0)


if __name__ == '__main__':
    main()

