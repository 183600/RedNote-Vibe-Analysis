# -*- coding: utf-8 -*-
"""两个分析管线共用的步骤运行器（论文管线 / fork 深潜套件）。

`run_paper_pipeline.py` 与 `run_fork_pipeline.py` 此前各自复制了一份完全相同的
参数解析与顺序执行逻辑；这里把「步骤选择 + CLI + 依次运行」收敛到单一实现，
两个入口脚本只保留各自的 STEPS 清单与说明文字。

步骤定义为 `(脚本名, 说明)` 元组列表；统一支持：

  --list            只打印步骤（不执行）
  --from N          从第 N 步开始（1-based）
  --only SCRIPT...  只运行指定脚本（未知名单直接报错）

被运行的脚本统一用当前解释器（`sys.executable`）顺序执行，任一步非零退出即
以 `subprocess.CalledProcessError` 中止，保证管线不会在失败后继续跑后续步骤。
"""
from __future__ import annotations

import argparse
import subprocess
import sys

Step = tuple[str, str]


def select_steps(steps: list[Step], only: list[str] | None = None, *,
                 from_step: int = 1) -> list[Step]:
    """按 `--only` / `--from` 选择要运行的步骤子集。

    only 非空时按其筛选（保持 STEPS 中的原始顺序）；未知脚本名抛 SystemExit。
    from_step 为 1-based；越界时不报错，返回空列表（由调用方决定行为）。
    """
    if only:
        wanted = set(only)
        unknown = wanted - {script for script, _ in steps}
        if unknown:
            raise SystemExit(f"未知步骤: {', '.join(sorted(unknown))}")
        return [(s, d) for s, d in steps if s in wanted]
    return steps[from_step - 1:]


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--list", action="store_true", help="只列出步骤")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        metavar="N", help="从第 N 步开始（1-based）")
    parser.add_argument("--only", nargs="+", metavar="SCRIPT", help="只运行指定脚本")
    return parser


def print_steps(steps: list[Step]) -> None:
    for i, (script, desc) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {script}  --  {desc}")


def run_steps(steps: list[Step], *, python: str = sys.executable,
              runner=subprocess.run) -> None:
    """顺序执行每个步骤；任一步失败立即抛出（不执行后续步骤）。

    `runner` 可在测试中替换为记录调用的假实现。
    """
    for i, (script, _) in enumerate(steps, 1):
        print(f"\n===== [{i}/{len(steps)}] running {script} =====", flush=True)
        runner([python, script], check=True)


def run_pipeline(steps: list[Step], description: str, done_message: str,
                 argv: list[str] | None = None, *, python: str = sys.executable,
                 runner=subprocess.run) -> int:
    """完整的 CLI 流程：解析参数 -> 打印步骤 -> （可选）执行 -> 完成提示。"""
    args = build_parser(description).parse_args(argv)
    selected = select_steps(steps, args.only, from_step=args.from_step)
    print_steps(selected)
    if args.list:
        return 0
    run_steps(selected, python=python, runner=runner)
    print(f"\n{done_message}")
    return 0
