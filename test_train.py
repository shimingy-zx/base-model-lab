import unittest
import subprocess
import tempfile
import pathlib
import json
import sys
import torch
from train import CharTokenizer, TinyTransformer, train_steps, generate


class TokenizerTests(unittest.TestCase):
    def test_round_trip_chinese_text(self):
        text = '春天来了，花儿开了。\n'
        tok = CharTokenizer(text)
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(len(tok.encode(text)), len(text))


class ModelTests(unittest.TestCase):
    def test_future_characters_do_not_change_previous_predictions(self):
        torch.manual_seed(7)
        model = TinyTransformer(vocab_size=16, block_size=8, width=16, heads=2)
        model.eval()
        a = torch.tensor([[1, 2, 3, 4, 5]])
        b = torch.tensor([[1, 2, 3, 8, 9]])
        with torch.no_grad():
            logits_a = model(a)
            logits_b = model(b)
        self.assertEqual(tuple(logits_a.shape), (1, 5, 16))
        torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], rtol=0, atol=1e-6)

    def test_training_reduces_loss_and_generates_new_tokens(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        text = '春天来了，花儿开了。\n' * 25
        tok = CharTokenizer(text)
        model = TinyTransformer(len(tok.chars), block_size=12, width=16, heads=2)
        losses = train_steps(model, torch.tensor(tok.encode(text)), steps=30, block_size=12, batch_size=8, learning_rate=0.01)
        self.assertLess(losses[-1], losses[0])
        sampled = generate(model, tok, '春天', max_new_tokens=5)
        self.assertTrue(sampled.startswith('春天'))
        self.assertEqual(len(sampled), 7)


class ExperimentTests(unittest.TestCase):
    def test_command_saves_checkpoint_and_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = pathlib.Path(directory) / 'input.txt'
            corpus.write_text('春天来了，花儿开了。\n' * 25, encoding='utf-8')
            out = pathlib.Path(directory) / 'result'
            p = subprocess.run([sys.executable, 'experiment.py', '--corpus', str(corpus), '--out', str(out),
                                '--steps', '5', '--block-size', '12', '--width', '16', '--prompt', '春天'],
                               cwd=pathlib.Path(__file__).parent, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            report = json.loads((out / 'result.json').read_text(encoding='utf-8'))
            self.assertLess(report['final_train_loss'], report['initial_train_loss'])
            self.assertTrue(report['sample'].startswith('春天'))
            self.assertTrue((out / 'model.pt').exists())
            sampled = subprocess.run([sys.executable, 'sample.py', '--model', str(out / 'model.pt'),
                                      '--prompt', '春天', '--length', '5'],
                                     cwd=pathlib.Path(__file__).parent, capture_output=True, text=True)
            self.assertEqual(sampled.returncode, 0, sampled.stderr)
            self.assertTrue(sampled.stdout.startswith('春天'))


if __name__ == '__main__':
    unittest.main()
