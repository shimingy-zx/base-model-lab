"""Run a small from-scratch decoder-only Transformer pretraining exercise."""
import argparse
import json
from pathlib import Path

import torch
from train import CharTokenizer, TinyTransformer, train_steps, generate


def main():
    parser = argparse.ArgumentParser(description='用 CPU 从随机参数训练一个教学用字符级基座模型')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).parent / 'data' / 'corpus.txt')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('result'))
    parser.add_argument('--steps', type=int, default=350)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--prompt', default='问题：')
    args = parser.parse_args()
    if args.steps < 1 or args.width % 2 or args.block_size < 2:
        parser.error('steps >= 1, width must be even and block-size >= 2')
    torch.set_num_threads(1)
    torch.manual_seed(9)
    text = args.corpus.read_text(encoding='utf-8')
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    model = TinyTransformer(len(tokenizer.chars), block_size=args.block_size, width=args.width, heads=2)
    before, after = train_steps(model, data, steps=args.steps, block_size=args.block_size,
                                batch_size=8, learning_rate=0.003)
    sample = generate(model, tokenizer, args.prompt, max_new_tokens=70)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'chars': tokenizer.chars,
                'block_size': args.block_size, 'width': args.width, 'heads': 2}, args.out / 'model.pt')
    report = {'initial_train_loss': before, 'final_train_loss': after,
              'training_steps': args.steps, 'parameters': sum(p.numel() for p in model.parameters()),
              'vocab_size': len(tokenizer.chars), 'corpus_characters': len(text),
              'sample': sample, 'note': '仅是小语料训练集损失，不代表泛化能力'}
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
