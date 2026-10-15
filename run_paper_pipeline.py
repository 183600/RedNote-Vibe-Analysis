# -*- coding: utf-8 -*-
"""按论文附录 A 的顺序依次运行 paper 管线。

用法：
  python3 run_paper_pipeline.py                  # 运行全部 8 步
  python3 run_paper_pipeline.py --list           # 只打印步骤
  python3 run_paper_pipeline.py --from 4         # 从第 4 步开始
  python3 run_paper_pipeline.py --only paper_detector.py paper_robustness.py
"""
import argparse
import subprocess
import sys

STEPS = [
    ("paper_lexicon.py", "词表发现 -> paper_lexicon_results.json"),
    ("paper_features.py", "特征抽取 -> paper_features_{human,aigc,exploring}.csv.gz"),
    ("paper_model_fingerprint.py", "模型指纹 -> paper_model_results.json + 2 图"),
    ("paper_detector.py", "检测器与检验 -> paper_detector_{results,model}.json + 2 图"),
    ("paper_drift.py", "时序漂移 -> paper_drift_results.json、逐帖得分 + 3 图"),
    ("paper_extra_figs.py", "补充图 -> 修正时间线 / 差中差 / 词表图"),
    ("paper_engagement.py", "互动分析 -> paper_engagement_results.json + 2 图"),
    ("paper_robustness.py", "稳健性检验 -> paper_robustness_results.json + 5 图"),
]


def select_steps(args) -> list[tuple[str, str]]:
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {script for script, _ in STEPS}
        if unknown:
            raise SystemExit(f"未知步骤: {', '.join(sorted(unknown))}")
        return [(s, d) for s, d in STEPS if s in wanted]
    return STEPS[args.from_step - 1:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="只列出步骤")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        metavar="N", help="从第 N 步开始（1-based）")
    parser.add_argument("--only", nargs="+", metavar="SCRIPT", help="只运行指定脚本")
    args = parser.parse_args()

    steps = select_steps(args)
    for i, (script, desc) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {script}  --  {desc}")

    if args.list:
        return 0

    for i, (script, _) in enumerate(steps, 1):
        print(f"\n===== [{i}/{len(steps)}] running {script} =====", flush=True)
        subprocess.run([sys.executable, script], check=True)
    print("\n论文管线全部步骤已完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
