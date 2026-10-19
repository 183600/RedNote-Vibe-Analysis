# -*- coding: utf-8 -*-
"""按推荐顺序运行本 fork 的「深潜分析」套件（输出统一落在 insights/）。

论文管线（paper_*.py）由 run_paper_pipeline.py 负责；本运行器只编排 fork
新增的深潜分析：全部复用 paper_features_*.csv.gz / paper_detector_model.json
或全量扫描原始 JSONL，互不写坏对方输出，可以任意单独重跑。

用法：
  python3 run_fork_pipeline.py                     # 运行全部 22 步
  python3 run_fork_pipeline.py --list              # 只打印步骤
  python3 run_fork_pipeline.py --from 3            # 从第 3 步开始
  python3 run_fork_pipeline.py --only detector_diagnostics.py family_lexicon_analysis.py

注意：完整运行需要数分钟到数十分钟（deep_dive / framing / title_style /
diversity 为全量扫描），建议先跑 --list 查看各步骤输出。
"""
import sys

from pipeline_common import run_pipeline

STEPS = [
    ("derived_cache.py",
     "为五个 JSONL 构建共享派生记录缓存（后续步骤免重复解析；缺失时自动回退）"),
    ("deep_dive_analysis.py",
     "发布节奏 / 词表扩散 / 句子节奏 / 重复内容 -> deep_dive_results.{json,md} + 4 图"),
    ("framing_analysis.py",
     "开篇/收尾模板化与梯度 -> framing_results.{json,md} + 2 图"),
    ("title_style_analysis.py",
     "标题端 AI 味（消融/像人度/互动/模板）-> title_style_results.{json,md} + 3 图"),
    ("title_body_congruence_analysis.py",
     "标题×正文 AI 味组合（四象限/时间演化/模型一致性/象限互动）-> title_body_congruence_results.{json,md} + 2 图"),
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
    ("family_transfer_analysis.py",
     "家族分类器迁移：野外高 AI 味帖的风格家族构成与互动回报 -> family_transfer_results.{json,md} + 2 图"),
    ("stance_analysis.py",
     "语气情态：谨慎表达/断言强调/程度强化/社区呼语的梯度与互动偏效应 -> stance_results.{json,md} + 2 图"),
    ("concreteness_analysis.py",
     "具体性：数字/单位量值/模糊量词/地名的梯度、长度控制与互动 -> concreteness_results.{json,md} + 2 图"),
    ("cta_analysis.py",
     "互动引导语：评论/点赞收藏/关注转发/催购四类 CTA 的梯度、互动与 p_ai 调节 -> cta_results.{json,md} + 2 图"),
    ("typography_analysis.py",
     "标点与排版：12 个标点族/34 个标记的跨子集密度、P(AI) 梯度、互动偏效应、家族指纹与季度趋势 -> typography_results.{json,md} + 3 图"),
    ("discourse_analysis.py",
     "篇章连接：因果/转折/递进/序列/举例/总结 6 类连接词 + 口语-书面语域，跨子集密度、P(AI) 梯度、互动偏效应、家族指纹与季度趋势 -> discourse_results.{json,md} + 3 图"),
    ("axis_synthesis_analysis.py",
     "跨轴综合：五轴 30 类别的共现结构、联合 P(AI) 独立信号、轴空间可分性与堆叠互动 -> axis_synthesis_results.{json,md} + 2 图"),
    ("redundancy_analysis.py",
     "正文自重复：片段复读率/句间相似度/近重复句/复读句首的跨子集长度匹配差异、P(AI) 梯度、互动偏效应与季度趋势 -> redundancy_results.{json,md} + 2 图"),
]


def main() -> int:
    return run_pipeline(STEPS, __doc__ or "fork deep-dive pipeline",
                        "fork 深潜分析全部步骤已完成。")


if __name__ == "__main__":
    sys.exit(main())
