"""Generate a sample using the checkpoint trained by experiment.py."""
import argparse
from pathlib import Path
import torch
from train import CharTokenizer, TinyTransformer, generate


def main():
    parser = argparse.ArgumentParser(description='从已训练的小模型生成文字')
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('result') / 'model.pt')
    parser.add_argument('--prompt', default='问题：')
    parser.add_argument('--length', type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    tokenizer = CharTokenizer(''.join(checkpoint['chars']))
    model = TinyTransformer(len(tokenizer.chars), checkpoint['block_size'], checkpoint['width'], checkpoint['heads'])
    model.load_state_dict(checkpoint['state_dict'])
    try:
        print(generate(model, tokenizer, args.prompt, max_new_tokens=args.length))
    except KeyError as error:
        parser.error(f'提示词含训练语料未见过的字符：{error.args[0]!r}，请换用语料中的字。')


if __name__ == '__main__':
    main()
