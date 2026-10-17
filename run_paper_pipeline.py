# -*- coding: utf-8 -*-
"""按论文附录 A 的顺序依次运行 paper 管线。

用法：
  python3 run_paper_pipeline.py                  # 运行全部 8 步
  python3 run_paper_pipeline.py --list           # 只打印步骤
  python3 run_paper_pipeline.py --from 4         # 从第 4 步开始
  python3 run_paper_pipeline.py --only paper_detector.py paper_robustness.py
"""
import sys

from pipeline_common import run_pipeline

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


def main() -> int:
    return run_pipeline(STEPS, __doc__ or "paper pipeline",
                        "论文管线全部步骤已完成。")


if __name__ == "__main__":
    sys.exit(main())
