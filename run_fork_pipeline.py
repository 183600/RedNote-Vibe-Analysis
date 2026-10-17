# -*- coding: utf-8 -*-
"""按推荐顺序运行本 fork 的「深潜分析」套件（输出统一落在 insights/）。

论文管线（paper_*.py）由 run_paper_pipeline.py 负责；本运行器只编排 fork
新增的深潜分析：全部复用 paper_features_*.csv.gz / paper_detector_model.json
或全量扫描原始 JSONL，互不写坏对方输出，可以任意单独重跑。

用法：
  python3 run_fork_pipeline.py                     # 运行全部 12 步
  python3 run_fork_pipeline.py --list              # 只打印步骤
  python3 run_fork_pipeline.py --from 3            # 从第 3 步开始
  python3 run_fork_pipeline.py --only detector_diagnostics.py family_lexicon_analysis.py

注意：完整运行需要数分钟到数十分钟（deep_dive / framing / title_style /
diversity 为全量扫描），建议先跑 --list 查看各步骤输出。
"""
import sys

from pipeline_common import run_pipeline

STEPS = [
    ("deep_dive_analysis.py",
     "发布节奏 / 词表扩散 / 句子节奏 / 重复内容 -> deep_dive_results.{json,md} + 4 图"),
    ("framing_analysis.py",
     "开篇/收尾模板化与梯度 -> framing_results.{json,md} + 2 图"),
    ("title_style_analysis.py",
     "标题端 AI 味（消融/像人度/互动/模板）-> title_style_results.{json,md} + 3 图"),
    ("diversity_analysis.py",
     "句长 burstiness / MTLD / Heaps β -> diversity_results.{json,md} + 2 图"),
    ("detector_diagnostics.py",
     "概率校准 / 阈值 / 错分切片 / 不确定性 -> detector_diagnostics_results.{json,md} + 3 图"),
    ("family_lexicon_analysis.py",
     "17 个 LLM 的家族/单模型词表指纹 -> family_lexicon_results.{json,md} + 1 图"),
    ("engagement_structure_analysis.py",
     "AI 味惩罚的互动结构分解（零互动/基尼/构成比）-> engagement_structure_results.{json,md} + 1 图"),
    ("engagement_drivers_analysis.py",
     "互动惩罚的逐特征载体（单元内秩相关 + 固定效应多元回归）-> engagement_drivers_results.{json,md} + 1 图"),
    ("author_style_analysis.py",
     "作者层分解（作者间 vs 人内斜率、ICC、人内配对）-> author_style_results.{json,md} + 2 图"),
    ("user_clustering.py",
     "探索集用户风格聚类 -> user_clustering_results.md"),
    ("symbol_analysis.py",
     "标签/表情生态（普及率、AI 专属符号、梯度、互动）-> symbol_results.{json,md} + 3 图"),
    ("near_duplicate_analysis.py",
     "字符 5-gram Jaccard 近重复/模板复用（bottom-k 候选+精确校验）-> near_duplicate_results.{json,md} + 1 图"),
]


def main() -> int:
    return run_pipeline(STEPS, __doc__ or "fork deep-dive pipeline",
                        "fork 深潜分析全部步骤已完成。")


if __name__ == "__main__":
    sys.exit(main())
