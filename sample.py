"""Generate a sample using the checkpoint trained by experiment.py."""
import argparse
from pathlib import Path
import torch
from train import CharTokenizer, TinyTransformer, generate


def run_sample(model, prompt='问题：', length=60, temperature=0.8):
    model_path = Path(model)
    if not model_path.exists():
        raise FileNotFoundError(f'模型权重文件不存在：{model_path}，请先训练模型。')
    torch.set_num_threads(1)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    tokenizer = CharTokenizer(''.join(checkpoint['chars']))
    m = TinyTransformer(len(tokenizer.chars), checkpoint['block_size'], checkpoint['width'], checkpoint['heads'])
    m.load_state_dict(checkpoint['state_dict'])
    return generate(m, tokenizer, prompt, max_new_tokens=length, temperature=temperature)


def main():
    parser = argparse.ArgumentParser(description='从已训练的小模型生成文字')
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('result') / 'model.pt')
    parser.add_argument('--prompt', default='问题：')
    parser.add_argument('--length', type=int, default=60)
    parser.add_argument('--temp', type=float, default=0.8)
    args = parser.parse_args()
    try:
        print(run_sample(args.model, args.prompt, args.length, args.temp))
    except KeyError as error:
        parser.error(f'提示词含训练语料未见过的字符：{error.args[0]!r}，请换用语料中的字。')
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

