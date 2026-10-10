#!/usr/bin/env python3
"""Unified CLI entrypoint for Base Model Lab."""
import argparse
import subprocess
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

BANNER = """\
======================================================================
🔬 Base Model Lab: 从零训练迷你基座模型 · 统一工作流命令行
======================================================================
常用指令速查:
  python lab.py train       从零训练迷你基座模型并导出权重与指标
  python lab.py sample      单次提示词文本续写采样
  python lab.py chat        启动终端打字机流式交互体验
  python lab.py viz         生成并查看权重、因果注意力与嵌入可视化报告
  python lab.py test        运行全套自动化测试验证模型与闭环
  python lab.py docs        启动本地 VitePress 深度技术文档预览
  python lab.py run         一键全流程闭环 (训练 -> 验证 -> 可视化 -> 生成)
======================================================================\
"""


def cmd_train(args):
    from experiment import run_experiment
    print(f"🚀 [Train] 开始训练基座模型 (steps={args.steps}, block_size={args.block_size}, width={args.width})...")
    try:
        report = run_experiment(
            corpus=args.corpus,
            out=args.out,
            steps=args.steps,
            block_size=args.block_size,
            width=args.width,
            prompt=args.prompt,
            verbose=True,
        )
        print("\n✅ 训练完成！")
        print(f"• 初始损失: {report['initial_train_loss']:.4f} -> 最终损失: {report['final_train_loss']:.4f}")
        print(f"• 权重已保存至: {Path(args.out) / 'model.pt'}")
        print("👉 接下来您可以运行: python lab.py chat 进行实时交互")
        return 0
    except (ValueError, KeyError, FileNotFoundError) as err:
        print(f"❌ 训练失败: {err}", file=sys.stderr)
        return 1


def cmd_sample(args):
    from sample import run_sample
    try:
        output = run_sample(
            model=args.model,
            prompt=args.prompt,
            length=args.length,
            temperature=args.temp,
        )
        print(f"\n📝 提示词: {args.prompt}")
        print(f"🤖 续写结果:\n{output}\n")
        return 0
    except (KeyError, FileNotFoundError, ValueError) as err:
        print(f"❌ 生成失败: {err}", file=sys.stderr)
        return 1


def cmd_chat(args):
    from chat import run_chat
    return run_chat(model_path=args.model, length=args.length, temp=args.temp) or 0


def cmd_viz(args):
    from visualize import run_visualize
    try:
        out_path = run_visualize(
            model_path=args.model,
            prompt=args.prompt,
            out_path=args.out,
            no_open=args.no_open,
        )
        return 0
    except (FileNotFoundError, KeyError, ValueError) as err:
        print(f"❌ 可视化生成失败: {err}", file=sys.stderr)
        return 1


def cmd_test(args):
    print("🧪 [Test] 运行单元测试套件...")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT_DIR), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def cmd_docs(args):
    docs_dir = ROOT_DIR / "docs"
    if not docs_dir.exists():
        print(f"❌ 文档目录不存在: {docs_dir}", file=sys.stderr)
        return 1

    action = args.docs_action or "dev"
    cmd = ["npm", "run", action]
    print(f"📖 [Docs] 正在执行: {' '.join(cmd)} (目录: {docs_dir})...")
    try:
        res = subprocess.run(cmd, cwd=docs_dir)
        return res.returncode
    except FileNotFoundError:
        print("❌ 未检测到 npm 命令，请确认已安装 Node.js 与 npm 环境。", file=sys.stderr)
        return 1


def cmd_run(args):
    print("\n=======================================================")
    print("🚀 [Pipeline] 开始执行一键端到端全流程闭环实验")
    print("=======================================================")

    # Step 1: 训练
    print("\n[Step 1/3] 训练基座模型...")
    from experiment import run_experiment
    out_dir = ROOT_DIR / "result"
    corpus_file = ROOT_DIR / "data" / "corpus.txt"
    report = run_experiment(
        corpus=corpus_file,
        out=out_dir,
        steps=args.steps,
        prompt=args.prompt,
        verbose=False,
    )
    print(f"✓ 训练完成: 损失由 {report['initial_train_loss']:.4f} 降至 {report['final_train_loss']:.4f}")

    # Step 2: 自动化测试
    if not args.skip_test:
        print("\n[Step 2/3] 验证单元测试...")
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(ROOT_DIR), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=1)
        test_result = runner.run(suite)
        if not test_result.wasSuccessful():
            print("❌ 单元测试未全部通过，流程中断！", file=sys.stderr)
            return 1
        print("✓ 全部单元测试通过")
    else:
        print("\n[Step 2/3] 跳过单元测试验证 (--skip-test)")

    # Step 3: 可视化报告
    print("\n[Step 3/3] 生成注意力与权重空间可视化报告...")
    from visualize import run_visualize
    viz_file = out_dir / "visualize.html"
    run_visualize(
        model_path=out_dir / "model.pt",
        prompt="陆向谦实验室",
        out_path=viz_file,
        no_open=args.no_open,
    )

    print("\n=======================================================")
    print("🎉 全流程实验成功完成！")
    print(f"• 模型权重: {out_dir / 'model.pt'}")
    print(f"• 评估报告: {out_dir / 'result.json'}")
    print(f"• 可视化页: file://{viz_file.resolve()}")
    print("• 生成样句: " + report["sample"])
    print("\n💡 接下来您可以直接运行交互终端:")
    print("   python lab.py chat")
    print("=======================================================\n")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lab",
        description="Base Model Lab 统一命令行管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=BANNER,
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="子命令", metavar="<command>")

    # 1. train
    p_train = subparsers.add_parser("train", aliases=["exp", "experiment"], help="训练迷你基座模型并导出检查点与结果")
    p_train.add_argument("--corpus", type=Path, default=ROOT_DIR / "data" / "corpus.txt", help="语料文件路径")
    p_train.add_argument("--out", type=Path, default=ROOT_DIR / "result", help="输出文件夹路径")
    p_train.add_argument("--steps", type=int, default=350, help="训练迭代步数 (默认: 350)")
    p_train.add_argument("--block-size", type=int, default=32, help="上下文窗口长度 (默认: 32)")
    p_train.add_argument("--width", type=int, default=32, help="词嵌入与隐层维度 (默认: 32)")
    p_train.add_argument("--prompt", default="问题：", help="评估生成所用的初始提示词")
    p_train.set_defaults(func=cmd_train)

    # 2. sample
    p_sample = subparsers.add_parser("sample", aliases=["gen", "generate"], help="从已训练好的权重采样续写文本")
    p_sample.add_argument("--model", type=Path, default=ROOT_DIR / "result" / "model.pt", help="模型权重文件")
    p_sample.add_argument("--prompt", default="问题：", help="输入提示词 (必须为语料中见过的字符)")
    p_sample.add_argument("--length", type=int, default=60, help="续写字符长度 (默认: 60)")
    p_sample.add_argument("--temp", type=float, default=0.8, help="采样温度 (默认: 0.8)")
    p_sample.set_defaults(func=cmd_sample)

    # 3. chat
    p_chat = subparsers.add_parser("chat", help="在终端开启打字机流式交互体验")
    p_chat.add_argument("--model", type=Path, default=ROOT_DIR / "result" / "model.pt", help="模型权重文件")
    p_chat.add_argument("--length", type=int, default=60, help="单次生成最大字符数 (默认: 60)")
    p_chat.add_argument("--temp", type=float, default=0.8, help="采样温度 (默认: 0.8)")
    p_chat.set_defaults(func=cmd_chat)

    # 4. viz
    p_viz = subparsers.add_parser("viz", aliases=["visualize"], help="生成注意力矩阵、位置编码与词向量空间可视化报告")
    p_viz.add_argument("--model", type=Path, default=ROOT_DIR / "result" / "model.pt", help="模型权重文件")
    p_viz.add_argument("--prompt", default="陆向谦实验室", help="测试注意力所用的提示词")
    p_viz.add_argument("--out", type=Path, default=ROOT_DIR / "result" / "visualize.html", help="HTML 输出路径")
    p_viz.add_argument("--no-open", action="store_true", help="生成后不自动在系统浏览器中打开")
    p_viz.set_defaults(func=cmd_viz)

    # 5. test
    p_test = subparsers.add_parser("test", help="运行自动化测试套件")
    p_test.add_argument("-v", "--verbose", action="store_true", help="输出更详细的测试过程")
    p_test.set_defaults(func=cmd_test)

    # 6. docs
    p_docs = subparsers.add_parser("docs", help="启动或构建 VitePress 交互式文档站点")
    p_docs.add_argument(
        "docs_action",
        nargs="?",
        default="dev",
        choices=["dev", "build", "preview"],
        help="文档操作: dev (实时预览，默认) | build (静态构建) | preview (预览产物)",
    )
    p_docs.set_defaults(func=cmd_docs)

    # 7. run (pipeline)
    p_run = subparsers.add_parser("run", aliases=["pipeline", "all"], help="一键全流程闭环：训练 -> 测试 -> 可视化")
    p_run.add_argument("--steps", type=int, default=350, help="训练迭代步数 (默认: 350)")
    p_run.add_argument("--prompt", default="问题：", help="测试提示词")
    p_run.add_argument("--no-open", action="store_true", help="不自动弹出浏览器查看可视化")
    p_run.add_argument("--skip-test", action="store_true", help="跳过单元测试验证")
    p_run.set_defaults(func=cmd_run)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        print(BANNER)
        print("\n提示: 请输入具体的子命令，例如 'python lab.py --help' 查看详细参数。")
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        sys.exit(args.func(args) or 0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
