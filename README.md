# RedNote-Vibe Dataset

[English](README.md) | [中文](README_zh.md)

**A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Social Media** [arxiv](https://arxiv.org/abs/2509.22055)

![Dataset Overview](overview.png)

## Abstract

The proliferation of Large Language Models (LLMs) has led to widespread AI-Generated Text (AIGT) on social media platforms, creating unique challenges where content dynamics are driven by user engagement and evolve over time. However, existing datasets mainly depict static AIGT detection. In this work, we introduce **RedNote-Vibe**, the first longitudinal (5-years) dataset for social media AIGT analysis. This dataset is sourced from Xiaohongshu platform, containing user engagement metrics (e.g., likes, comments) and timestamps spanning from the pre-LLM period to July 2025, which enables research into the temporal dynamics and user interaction patterns of AIGT.


## Dataset Structure

All data files are available at the following link:

**[Download from Google Drive](https://drive.google.com/drive/folders/1T8JV-DmIo7SI8pBOPeuiQvXzFwe8ns--?usp=sharing)**

The directory contains:

- `training_set_human.jsonl` — Human-authored posts (pre-LLM period, before Nov 2022)
- `training_set_aigc.jsonl` — AI-generated posts using 17 LLMs
- `exploration_set.jsonl` — Post-LLM period posts (2023-2025) for temporal analysis

If you would like to receive notifications about updates, please **STAR to subscribe**.

### Dataset Statistics

The dataset spans 11 content domains. The table below summarizes per-domain statistics: number of posts (#), average text length (Length), average number of tags (#Tags), and average user engagement (Likes, Comments, Collections).

| Domain    |     # | Length | #Tags |  Likes |  Comm. |  Colls. |
|-----------|------:|-------:|------:|-------:|-------:|--------:|
| Health    | 19.2k |  333.6 |   7.6 | 2814.3 |   97.1 |  2468.8 |
| Fashion   | 17.7k |  191.2 |  10.9 | 1850.5 |   62.5 |   681.3 |
| Food      | 10.5k |  398.4 |   3.0 |   66.2 |   11.9 |    45.0 |
| Career    | 11.7k |  406.9 |  10.0 | 2047.2 |  102.6 |  1445.9 |
| Pets      |  3.9k |  433.2 |   2.7 |   57.5 |   12.6 |    34.1 |
| Education |  5.9k |  496.0 |   3.7 |   55.9 |    6.1 |    38.3 |
| Sports    | 17.7k |  528.8 |   3.2 |   60.1 |    9.8 |    36.4 |
| Relation. | 22.5k |  308.2 |   7.1 | 2663.6 |  275.8 |   593.9 |
| Travel    | 14.8k |  452.8 |  10.7 | 1102.4 |   86.6 |   662.0 |
| Wellness  | 16.0k |  396.5 |  10.5 | 2348.6 |  217.7 |  1025.8 |
| Others    | 16.9k |  220.9 |   8.9 |  208.8 | 1266.2 |  3225.0 |


### Data Format

Each entry contains the following fields:

```json
{
  "note_title": "Post title",
  "local_time": "Publication timestamp (YYYYMMDDHH)",
  "note_content": "Main text content",
  "likes": 123,
  "collections": 45,
  "comments": 67,
  "domain": "Content category",
  "model_family": "AI provider (for AIGC data)",
  "model": "Specific AI model (for AIGC data)"
}
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
# optional: only needed by quick_start.py (BERT baseline demo)
pip install torch transformers datasets
```

### Basic Usage

Run the quick start example to train a BERT model for AIGT detection:

```bash
python quick_start.py
```

This script will:
1. Load the human and AI-generated datasets
2. Prepare balanced training/validation/test splits
3. Fine-tune a BERT model for binary classification
4. Evaluate performance and show prediction examples
5. Save the trained model for future use


## 🗂 Repository Layout

| Path | Description |
|------|-------------|
| `training_set_human.jsonl`, `training_set_aigc.jsonl`, `exploring_set.jsonl` | Research datasets (never modify) |
| `paper_*.py` | Paper pipeline: lexicon → features → model fingerprint → detector → drift → extra figures → engagement → robustness |
| `analysis_utils.py` | Shared helpers: streaming JSONL reader, record/content/engagement/timestamp helpers, canonical body cleaning (`paper_features` lexicon), Chinese fonts, unified plot style, detector scoring, entropy/coverage, lexical-diversity tools (burstiness, MTLD, Heaps β, prefix n-gram stats, bootstrap median CI), stats (bootstrap CI, Cliff's delta, Wilson CI) |
| `paper_figs/` | Figures produced by the paper pipeline |
| `deep_dive_analysis.py` | Full-scan deep dive (publish rhythm / lexicon diffusion / sentence rhythm / duplicate structure) |
| `insights/` + `deep_dive_results.{json,md}` | Figures and result files produced by `deep_dive_analysis.py` |
| `framing_analysis.py` | Framing analysis: opening/closing template concentration, AI-specific template gradient/diffusion, ending-symbol composition |
| `insights/framing_*.png` + `framing_results.{json,md}` | Figures and result files produced by `framing_analysis.py` |
| `title_style_analysis.py` | Title stylometry: title-feature ablation, per-model title human-likeness, title-level engagement & templates |
| `insights/title_*.png` + `title_style_results.{json,md}` | Figures and result files produced by `title_style_analysis.py` |
| `diversity_analysis.py` | Sentence burstiness / MTLD / Heaps β analysis (full-scan sentence metrics + per-dataset sampling) |
| `insights/diversity_*.png` + `diversity_results.{json,md}` | Figures and result files produced by `diversity_analysis.py` |
| `detector_diagnostics.py` | Detector diagnostics: probability calibration (prior correction), threshold sensitivity, error slices by length/domain, exploring-set confidence bands |
| `insights/detector_*.png` + `detector_diagnostics_results.{json,md}` | Figures and result files produced by `detector_diagnostics.py` |
| `family_lexicon_analysis.py` | Per-family / per-model lexical fingerprints inside the AIGC subset (domain-reweighted log-odds) |
| `insights/family_lexicon.png` + `family_lexicon_results.{json,md}` | Figure and result files produced by `family_lexicon_analysis.py` |
| `engagement_structure_analysis.py` | Engagement-structure decomposition of the AI-style penalty: zero-engagement density, Gini / top-10% concentration, collection-to-like ratio, within-cell rank correlations |
| `insights/engagement_structure.png` + `engagement_structure_results.{json,md}` | Figure and result files produced by `engagement_structure_analysis.py` |
| `engagement_drivers_analysis.py` | Feature-level drivers of the AI-style engagement penalty: within-cell rank correlations for all 37 features + month×domain fixed-effects OLS with cluster-bootstrap CIs, plus P(AI) net-of-drivers models |
| `insights/engagement_drivers.png` + `engagement_drivers_results.{json,md}` | Figure and result files produced by `engagement_drivers_analysis.py` |
| `author_style_analysis.py` | Author-level decomposition of the penalty (`user_id` subset): selection check, author concentration, ICC of P(AI), between-author vs within-author FE slopes with cluster-bootstrap CIs, within-author paired comparison |
| `insights/author_*.png` + `author_style_results.{json,md}` | Figures and result files produced by `author_style_analysis.py` |
| `symbol_analysis.py` | Hashtag/emoji symbol ecology: structure & adoption, generator-specific tags, per-tag AI-ness, P(AI) gradient, quarterly diffusion, tag count vs engagement |
| `insights/symbol_*.png` + `symbol_results.{json,md}` | Figures and result files produced by `symbol_analysis.py` |
| `near_duplicate_analysis.py` | Near-duplicate / template-reuse scan: char 5-gram rolling hashes + bottom-k (K=128) candidate generation + exact Jaccard verification, within/cross-subset shares, P(AI) gradient, quarterly trend, cluster structure, brute-force recall self-check |
| `insights/near_duplicate.png` + `near_duplicate_results.{json,md}` | Figure and result files produced by `near_duplicate_analysis.py` |
| `run_fork_pipeline.py` | Runner for the 12-step deep-dive suite (`--list` / `--from N` / `--only SCRIPT`) |
| `pipeline_common.py` | Shared step-runner behind `run_paper_pipeline.py` / `run_fork_pipeline.py` (step selection, CLI, sequential execution) |
| `*.py` (other) | Exploratory fork analyses: word frequency, particles, topics, sentiment, time, domains, user clustering, … |
| `output/` | Figures from the joint exploratory analysis |
| `tests/` | Unit tests for `analysis_utils.py` (incl. calibration/threshold/within-cell helpers), framing helpers, the title/diversity/symbol/near-duplicate/engagement-driver/author-decomposition pure functions, and the shared pipeline runner (`python3 -m unittest discover -s tests`) |
| `ci_gate.py` | CI quality gate (syntax + research-data integrity) |

All analyses are deterministic (fixed random seeds) and can be re-run independently; the paper pipeline order is documented in [paper.md](paper.md), Appendix A.

## 📄 Paper

A full research paper based on the analyses in this fork is available at **[paper.md](paper.md)** (in Chinese): *Anatomy, Diffusion, and Cost of the "AI Flavor": A Stylometric Study of AI-Generated Text on Chinese Social Media*. Key findings: an interpretable 35-feature detector reaches AUC 0.940; human-likeness ranking of 17 LLMs (GLM-3 and Qwen2.5 most human-like, Gemini least); a step-change (not gradual drift) in AI-style prevalence after ChatGPT (high-confidence share 0.6–1.9% → 14.6–22.2%, sampling-stratified); and a stable "AI-flavor penalty" on engagement (median likes 785 vs 106.5 for the most-human vs most-AI deciles).

Robustness checks (length matching, within-domain effect consistency, leave-one-domain-out transfer, bootstrap confidence intervals and threshold sensitivity) are in `paper_robustness.py` / `paper_robustness_results.json` and §5.6 of the paper: AUC 95% CI 0.937–0.943, cross-domain AUC 0.914–0.949, prevalence step holds at every threshold in {0.5, 0.7, 0.9, 0.95}.

Run the full pipeline in order with `python3 run_paper_pipeline.py` (or `--list` to inspect the 8 steps).

## 📊 Fork 分析

本 fork 对 RedNote-Vibe 数据集进行了全面的数据分析，主要发现和可视化如下：

### 数据集概览

| 数据集 | 记录数 | 平均内容长度 | 平均标题长度 |
|--------|--------|-------------|-------------|
| training_set_human | 51,878 | 488.1 | 15.7 |
| training_set_aigc | 7,254 | 347.7 | 18.3 |
| exploring_set | 91,517 | 322.6 | 15.0 |

### 词频相关性矩阵

**正文内容 (note_content)：**

|                    | 人类 | AIGC | 探索集 |
|--------------------|------|------|--------|
| training_set_human | 1.00 | 0.67 | 0.83   |
| training_set_aigc  | 0.67 | 1.00 | 0.64   |
| exploring_set      | 0.83 | 0.64 | 1.00   |

**标题内容 (note_title)：**

|                    | 人类 | AIGC | 探索集 |
|--------------------|------|------|--------|
| training_set_human | 1.00 | 0.46 | 0.53   |
| training_set_aigc  | 0.46 | 1.00 | 0.45   |
| exploring_set      | 0.53 | 0.45 | 1.00   |

### 关键发现

1. **标题是强区分特征**：人类与AIGC在标题上的相关性仅为0.46，远低于正文(0.67)
2. **内容长度**：人类帖子最长(488字符)，AIGC中等(348)，探索集最短(323)
3. **标题长度**：AIGC标题最长(18.3字符)，更正式详细
4. **高频词差异**：人类使用更多emoji；AIGC常使用"姐妹们"

### 常见词（助词/虚词）分析

除了实词外，我们也分析了像"的"、"了"、"是"这种在各话题都有的常见虚词/助词的使用模式：

**1. 常见词使用排名（均值比率，前10）：**

| 词语 | Human | AIGC | Exploring |
|------|-------|------|--------|
| 的 | 2.64% | 2.35% | 1.93% |
| 是 | 1.08% | 0.85% | 0.72% |
| 了 | 0.95% | 0.85% | 0.59% |
| 有 | 0.73% | 0.50% | 0.51% |
| 我 | 0.90% | 0.79% | 0.67% |
| 在 | 0.46% | 0.43% | 0.33% |

**2. 共现关系（重要成对词相关性）：**

| 词对 | Human | Exploring |
|------|-------|-----------|
| 的+是 | 0.69 | 0.80 |
| 的+了 | 0.37 | 0.60 |
| 有+没有 | 0.56 | 0.66 |
| 是+在 | 0.47 | 0.66 |

**3. 关键差异：**
- **Human文本更长**：平均502字 vs AIGC 367字 vs exploring 334字
- **"的"使用率**：Human最高（2.64%），说明表达更口语化/情感化
- **领域差异**：心理类"的"使用率最高（~3%），穿搭最低（~1.3%）
- **AIGC特点**：与Human相关性高但与exploring差异大，表现出混合特征

### 深潜分析（`deep_dive_analysis.py`）

全量扫描四个新维度：发布节奏 / 词表扩散 / 句子节奏 / 重复内容。结果图在 `insights/`，数据在 `deep_dive_results.json` 与 `deep_dive_results.md`。前三项按 note_id 去重后统计（每篇笔记一次），重复内容保留逐记录口径：

1. **发布节奏**：探索集发布峰值在 18 时（8.66%），18–23 时合计 42.28%，周末占 26.64%；人类集 `local_time` 小时字段恒为占位值 08，小时分布不可用；两集星期分布形态相关 ρ=0.571。
2. **词表扩散**：受控 AIGC 参考 AI 味词密度 48.0/万字符；人类集干净窗口（≤2022.06）AI 味词使用率随时间显著下降（ρ=-0.851，p=3.9e-06），探索集全窗口平稳（ρ=0.089，p=0.64；2023H1 14.95 → 2025H1 16.34 每万字符）。
3. **句子节奏**：平均句长中位数 human 46.7 / aigc 25.6 / exploring 22.1；人类 vs AIGC Cliff's δ=-0.522，人类 vs 探索集 δ=-0.632（p≈0）。
4. **重复内容与数据完整性**：探索集逐记录正文重复占比 68.11%（人类 1.05%、AIGC 0.00%）；91,517 条记录仅 60,161 个唯一 note_id，重复绝大部分是同一笔记「带完整元数据的副本 + 缺 user_id/互动字段的副本」；去重到 note_id 后正文重复率降至 1.52%；另有 63.70% 的记录缺 user_id 或领域为「未知」。

### 框架模板化分析（`framing_analysis.py`）

全量扫描三个新维度：开篇/收尾模板集中度、AI 专属模板梯度与扩散、收尾符号构成。结果图在 `insights/framing_*.png`，数据在 `framing_results.json` 与 `framing_results.md`。探索集按 note_id 去重，p_ai 与 `paper_exploring_scores.csv.gz` 按行序对齐（content_len 逐条校验，mismatch=0）：

1. **开篇公式化**：AIGC 首 4 字 Top-20 覆盖率为 14.8%（人类 3.1%、探索集 3.2%），开篇熵 11.86 bit（人类 15.55、探索集 15.37）；收尾的集中度差异则很小（4.7% vs 6.0%/7.3%）。
2. **梯度**：探索集内 P(AI) 十分位与 AI 专属开篇命中率显著上升（0.6%→2.3%，Cliff's δ=0.22），与人类专属开篇命中率显著下降（1.8%→0.3%，δ=-0.23）。
3. **扩散**：AI 专属开篇使用率 AIGC 16.4% / 探索集 1.3% / 人类集 0.8%，探索集季度趋势不显著（ρ=0.21，p=0.57），说明框架层面的扩散弱于词汇层面；小时分布几乎平坦（极差 0.094）。
4. **收尾符号**：AIGC 以感叹号（26.8%）、emoji（16.8%）、波浪号（9.4%）收尾的比例远高于人类（5.4%/1.9%/1.9%），探索集（10.4%/6.8%/3.5%）介于两者之间；探索集另有 17.2% 为纯标签帖（人类 0%、AIGC 0.3%）。

### 标题端「AI 味」分析（`title_style_analysis.py`）

对论文特征中的 7 个标题特征做检测器消融、模型像人度、时间动态与互动分析。结果图在 `insights/title_*.png`，数据在 `title_style_results.json` 与 `title_style_results.md`。消融与 `paper_detector.py` 完全同口径（同一 5 折 CV、随机种子）；标题模板统计的人类/探索集按 note_id 去重：

1. **标题单独可分性**：仅 7 个标题特征 5 折 CV AUC=0.742（正文 29 特征 0.929、全部 37 特征 0.940），达到正文模型的 79.9%；标题得分与正文得分秩相关仅 ρ=0.017——标题与正文是两个近乎独立的 AI 风格信号，只看正文会漏掉标题端信息。
2. **模型像人度（标题端）**：gpt-o4 标题最像人（均值 0.446），gemini-2.5 最不像（0.794）；排序与正文端（GLM-3/Qwen2.5 最像人、Gemini 全系最不像）不同，模型差异在标题端有独立表现。
3. **标题端无互动惩罚**：标题得分与点赞总体 ρ=+0.082、控制月份×领域后仅 +0.006（正文对照 ρ=-0.139）；「AI 味惩罚」主要由正文承载，标题的营销式写法不直接压低互动。
4. **标题模板化弱于正文**：AIGC 标题首 4 字 Top-20 覆盖率仅 3.8%（人类 2.8%、探索集 4.8%），远低于正文开篇的 14.8%（熵 12.46 vs 11.86 bit）；AI 专属标题前缀以 `姐妹们！`（+0.82pp）、`夏日必备`、`秋冬必备` 为代表；标题半年度高置信（≥0.9）占比仅 0.6%–1.2%，标题端风格跃升明显弱于正文端。

### 文本多样性与节奏深潜（`diversity_analysis.py`）

全量句法指标（去重口径同上）+ 每子集 12,000 帖固定种子抽样的词汇多样性。结果图在 `insights/diversity_*.png`，数据在 `diversity_results.json` 与 `diversity_results.md`：

1. **句长 burstiness**：人类帖长短句交错显著更强——burstiness 中位数人类 0.668 vs AIGC 0.540 / 探索集 0.579（human vs AIGC Cliff's δ=+0.339，p≈0）；句长分布人类长尾更重（≥180 字截断桶仍有约 6%）。
2. **词汇多样性（MTLD）**：AIGC 每帖 MTLD 中位数 189.9，显著高于人类 122.2 与探索集 126.7（human vs AIGC δ=-0.507）——单帖内用词更丰富，与 `paper_features` 的独特字/二元组比率方向一致。
3. **Heaps 定律（统一 100 万 token 预算）**：词汇增长斜率 β 人类 0.639 / 探索集 0.613 / AIGC 0.583——AIGC 跨帖词汇增长最慢；单帖内「词汇量大」与跨帖「模板复用多」并存，与 `framing_analysis` 的模板集中度结论互洽。

### 检测器诊断（`detector_diagnostics.py`）

对论文检测器做四项诊断：概率校准、阈值敏感性、错分切片与探索集不确定性。结果图在 `insights/detector_calibration.png`、`detector_errors.png`、`detector_uncertainty.png`，数据在 `detector_diagnostics_results.json` 与 `detector_diagnostics_results.md`。CV 口径与 `paper_detector.py` 完全一致（复算 AUC=0.9402/0.9294，逐位吻合）：

1. **概率校准**：平衡类权重得分不能直接当概率用——按自然先验（P(AI)=12.3%）做 logit 平移校正后，ECE 从 0.1446 降到 0.0072、Brier 从 0.0911 降到 0.0482，校正后校准良好。
2. **阈值敏感性**：τ=0.5 时精确率 0.49/召回率 0.86（人类 12.5% 被误标）；τ=0.9 时精确率升至 0.873/召回率 0.56（人类仅 1.1% 被误标）；τ=0.95 精确率 0.940——高置信子集分析用高阈值是安全的。
3. **错分切片**：错误率随长度单调变化——最短十分位（~208 字）人类假阳性率 28.0%，最长十分位（~950 字）AIGC 假阴性率 38.7%（该十位 AUC 降至 0.898）；领域间差异小得多（人类假阳性率 9.1%–15.6%），说明检测器不是靠领域词作弊。错分画像镜像对称：人类假阳性帖在独特字比率、AI 味词、感叹号上更像 AIGC，AIGC 假阴性帖则反之。
4. **探索集不确定性**：高置信占比在 31 个月内的月度趋势不显著（ρ=0.295，p=0.12，与 §5.4 平台期一致），而"拿不准"（0.4–0.6）占比不升反降（ρ=-0.549，p=0.002）——排除"高置信上升只是检测器越来越含糊"的解释。

### 模型家族词表指纹（`family_lexicon_analysis.py`）

在 AIGC 子集内部做家族/单模型 one-vs-rest 的 informative Dirichlet log-odds（口径同 `paper_lexicon.py`，并额外把各家族领域构成重加权到全语料分布以消除话题混杂）。结果图在 `insights/family_lexicon.png`，数据在 `family_lexicon_results.json` 与 `family_lexicon_results.md`：

1. **家族口癖差异显著**：GPT 偏好"其实/特别/变得"类书面连接词；Gemini 高度偏好"姐妹/感觉/真的/们/冲"的社区话术；DeepSeek 偏好"直接/突然/救命/居然"的情绪化口语；Qwen 偏好"不仅/更加/非常/成为"的正式递进词；GLM 特征词偏叙事化（探索/探秘/探店）；Claude 介于正式与口语之间。
2. **共享 AI 味词表（25 词）密度**：家族间相差 1.9 倍——Gemini 70.8/万字符最高，GLM 37.2 最低，人类基线仅 14.0；这与 `paper_model_fingerprint.py` 的"Gemini 最不像人、GLM-3 最像人"结论方向一致。
3. **单模型粒度**：如 gemini-2.0 的"姐妹"、deepseek-v3 的"绝绝子/yyds"、claude-sonnet-4 的"很/真的/但是"、glm-3 的"雪山/本科"，为 §5.2 的风格指纹补充了可直接引用的语言学证据。

### 互动结构与「AI 味惩罚」分解（`engagement_structure_analysis.py`）

把 `paper_engagement.py` 的"AI 味惩罚"拆成互动结构：零互动密度、长尾集中度（基尼/前 10% 占比）与互动构成比。结果图在 `insights/engagement_structure.png`，数据在 `engagement_structure_results.json` 与 `engagement_structure_results.md`。口径与 `paper_engagement.py` 一致（2023.01 起、同一十分位切分、月份×领域单元内秩相关），十分位点赞中位数与 `paper_engagement_results.json` 逐位吻合：

1. **惩罚首先体现在"沉默"**：零点赞占比从最像人十分位的 2.0% 升到最像 AI 十分位的 8.3%（零收藏 6.7%→22.2%、零评论 11.5%→18.5%；零点赞占比随十分位的 Spearman ρ=+0.939, p=5.5e-05）。
2. **点赞长尾更集中**：点赞基尼从 0.839 升到 0.903，前 10% 帖子的点赞占比 75.1%→85.0%；收藏/评论本身接近最大集中（基尼≈0.98–1.00），变化主要发生在点赞端。
3. **收藏/点赞比倒 U**：中位比从 0.387 爬到第 5 十分位峰值 0.450 后跌到 0.264——最像 AI 的帖子要么无人问津、要么只剩点赞，收藏价值坍缩。
4. **方向稳健**：控制月份×领域后（141 个单元），点赞/收藏的单元内加权 ρ≈-0.09，82%/82% 的单元为负；评论 ρ=-0.019（60% 为负）——互动惩罚在所有指标上方向一致，点赞与收藏最敏感。

### 互动惩罚的逐特征载体（`engagement_drivers_analysis.py`）

把论文 §5.5 的「AI 味惩罚」再拆一层：对检测器的 37 个可解释特征做「月份×领域」单元内的单变量秩相关，以及控制其余风格特征后的固定效应多元回归（特征先 z-score 再单元内去均值；95% CI 来自按单元整群 bootstrap 300 次）。结果图在 `insights/engagement_drivers.png`，数据在 `engagement_drivers_results.json` 与 `engagement_drivers_results.md`。P(AI) 的单元内加权 ρ=-0.0907 与 `paper_engagement_results.json` 逐位一致（n=85,690、141 个单元、每单元 ≥100 条）：

1. **强负载体是「高密度标点 + 规整句式」，而不是某个 AI 专属词**：单变量最强负关联为独特二元组比率（ρ=-0.135，92% 单元为负）、逗号密度（-0.123）、独特字比率（-0.119）、句号密度（-0.095）；多元控制后最强负偏效应为逗号密度（β=-0.104）、不同 emoji 数（-0.086）、句号密度（-0.062）、标题含问号（-0.051）、平均行长（-0.048）。
2. **正向载体与既有结论互洽**：标签数（单变量 +0.116 / 多元 +0.088）与正文长度（多元 +0.081）是最大的正驱动——与 `symbol_analysis` 的「标签越多互动越好」、`title_style_analysis` 的「标题端无惩罚」方向一致。
3. **P(AI) 惩罚是独立成分**：单独回归 β=-0.094（95% CI [-0.109, -0.081]），控制正文长度与标签数后仍为 -0.079（[-0.090, -0.068]）——不是长度或标签数量的间接效应。
4. **跨指标高度一致**：点赞与收藏的 37 个偏效应 Pearson r=0.951（收藏侧 23/37 显著）——惩罚结构不是点赞独有。

> 方法注记：单变量与多元回答的问题不同，个别特征会出现符号反转（如正文长度单变量 ρ 略负、偏效应显著为正），这是特征间强相关下的抑制定量（suppression）；结果文件同时给出两种口径，避免误读。

### 作者层分解：作者间 vs 人内（`author_style_analysis.py`）

前面的互动分析都在逐帖层面，无法回答「惩罚来自同一作者写了更 AI 风格的帖子，还是来自长期写 AI 风格的作者整体互动更低」。本分析在探索集带 `user_id` 的去重笔记子集（33,219 篇 / 26,740 位作者，占去重笔记 55.2%）上做作者间/人内分解。结果图在 `insights/author_structure.png`、`insights/author_within.png`，数据在 `author_style_results.json` 与 `author_style_results.md`。口径与 `paper_engagement.py` 一致（≥2023.01），并复算总体 ρ=-0.1392、单元内加权 ρ=-0.0907 与 `paper_engagement_results.json` 逐位吻合：

1. **选择检查**：作者子集 2025 年占 71.0%、点赞中位 597（无 user_id 记录 28）、P(AI) 更低（均值 0.439 vs 0.504，高置信 15.6% vs 22.4%）——作者层结论只在此采样层内成立。
2. **风格是半作者属性**：≥2 帖作者（3,362 位 / 9,841 帖）的 P(AI) ICC=0.535；作者发文量并不更 AI（ρ=-0.027），高置信帖在前 10% 高产作者中占 21.7%（低于其笔记占比 25.5%）。
3. **惩罚主要在作者之间**：精确分解 β_pooled=-0.324 = 0.91×β_between(-0.355) + 0.09×β_within(-0.006)；作者间 WLS（≥2 帖）β=-0.141（95% 整群 bootstrap CI [-0.298, -0.011]），人内（作者 FE）β=-0.006（[-0.070, +0.067]）、再加月份 FE 为 -0.001（[-0.068, +0.062]）。
4. **人内配对无差异**：1,266 位同时有「像人」（P(AI)<0.5）与「像 AI」（≥0.5）帖的作者，平均 Δ(log10 点赞)=-0.003（CI [-0.050, +0.042]，乘数 ×0.99；48.1% 为负，Wilcoxon p=0.85）；同作者同月份单元（517 个）Δ=-0.016（p=0.63）；≥4 帖作者（616 位）人内 ρ 加权均值 -0.003。
5. **解读**：横截面负相关主要由作者间差异承载，人内点估计接近零但 CI 较宽（无法排除小幅惩罚）；这不是对 `paper_engagement.py` 结论的否定，而是把「AI 味惩罚」的位置定位到作者层。

### 话题标签与表情生态（`symbol_analysis.py`）

把标签与表情当作「符号系统」全量扫描六个维度：标签结构与普及率、生成器专属标签、标签级 AI 味、P(AI) 梯度与季度扩散、emoji/[xxR] 表情生态、标签数×互动。结果图在 `insights/symbol_*.png`，数据在 `symbol_results.json` 与 `symbol_results.md`。标签口径与 `paper_features.py` 的 `n_hashtags` 同源（兼容 `#标签[话题]#` 与 BOM 残留），表情正则与其完全相同（另剔除变体选择符的重复计数）；人类/探索集按 note_id 去重，探索集 p_ai 按行序对齐（content_len 校验 mismatch=0）：

1. **标签普及，却不是高 P(AI) 帖的特征**：标签使用率随语料时代上升（人类 54.9%/2.96 个 → AIGC 71.2%/3.94 个 → 探索集 91.9%/6.61 个），但探索集内每帖标签数随 P(AI) 显著回落（ρ=-0.213；剔除纯标签帖后 -0.258；控制长度分层后加权 -0.268）。
2. **AIGC 标签词表最「一次性」**：AIGC 的 65.6% 标签用量是仅出现一次的标签（人类 36.4%），生成器临场自造 `#健康生活`/`#ootd`/`#健康饮食` 等 AI 专属话题词；人类则复用 `#笔记灵感`/`#减肥减脂吃这些` 等社区既定标签。
3. **专属标签没有迁移到真实流量**：探索集内 AI 专属标签命中率随 P(AI) 分位从 12.2% 降到 3.7%（δ=-0.154），人类专属标签 4.1%→2.3%——高 P(AI) 帖是整体少用标签，而非在标签层面更像生成集。
4. **标签层面的季度扩散不显著**：AI 专属标签季度 ρ=+0.588（p=0.07）、AIGC 高频 Top-50 ρ=-0.042（p=0.91）；对照词汇层面的显著扩散，标签没有出现同样的传染曲线。
5. **表情偏好分化**：AIGC 更多用 Unicode emoji（58.5% vs 人类 40.8%）且偏好 `➕⭐❤❗` 一类清单/要点符号；人类更常用 `[爱心R][开心R][比心R]` 社区表情（29.0% vs 13.9%）与 `✨🌟😭💪` 情绪 emoji；探索集 emoji 帖占比随 P(AI) 上升（ρ=+0.169）。
6. **标签越多互动越好（方向与 AI 味相反）**：控制月份×领域后（100 个单元），标签数与 log 点赞的单元内加权 ρ=+0.118（仅 20% 单元为负），点赞中位数从 0 标签的 10 升到 10+ 标签的 179。

### 近重复内容与模板复用（`near_duplicate_analysis.py`）

对全量清洗正文做字符 5-gram 滚动哈希 + bottom-k（K=128）候选生成 + 完整集合精确 Jaccard 校验，度量子集内近重复率、跨子集模板迁移、P(AI) 梯度、季度趋势与近重复簇结构。结果图在 `insights/near_duplicate.png`，数据在 `near_duplicate_results.json` 与 `near_duplicate_results.md`。清洗后不足 20 个 unique 5-gram 的「空正文/纯标签」帖（探索集 15,312 篇）不参与匹配并单独计数；召回自检（1,000 篇抽样暴力基准，J ≥ 0.5）召回率 100%（73/73）：

1. **近重复集中在探索集，且以「近似改写」为主**：探索集 3.49% 的帖子存在 J ≥ 0.7 孪生（J ≥ 0.9 仍有 1.47%），远高于人类集（0.90%）；而两帖 5-gram 集完全相同的整帖重复只有 0.67%——近重复主要来自同模板换词改写而非直接搬运。
2. **AIGC 子集内部零自重复，但存在人类→生成的模板迁移**：7,175 篇生成帖两两之间 J ≥ 0.5 对数 0；但生成集与人类集有 33 对 J ≥ 0.5、9 对 J ≥ 0.7 的近重复（示例均为换词/改标点/续写级改写），其中 GLM 家族 2.82% 的帖子命中——与「生成集以真实人类帖为底本」的构造方式一致；生成集与探索集的迁移反而极少（5 对 ≥ 0.5）。
3. **近重复与 AI 味基本正交**：探索集近重复占比沿 P(AI) 十分位基本持平（Spearman ρ=0.042, p=0.907）——高 AI 味并不主要来自复制粘贴/套模板。
4. **季度趋势不显著**：J ≥ 0.7 占比从 2023Q1 的 5.13% 回落到 2024Q3 的 1.35%、2025Q2 回到 3.26%，整体无显著单调趋势（ρ=-0.297, p=0.405）。
5. **簇结构**：J ≥ 0.7 连通分量 657 个（最大 26 篇，为 2024Q1 的三元锂电池安全科普改写群；另有 2025-03 手机发布文案 17 篇）。

### 生成的可视化图表

| 文件 | 描述 |
|------|------|
| `content_length_distribution.png` | 内容长度分布 |
| `title_top_words.png` | 标题高频词 |
| `desc_top_words.png` | 正文高频词 |
| `domain_domain_distribution.png` | 领域分布 |
| `scatter_*.png` | 词频相关性散点图 |
| `particle_distribution_comparison.png` | 常见词分布对比 |
| `particle_mean_comparison.png` | 常见词平均使用率 |
| `particle_heatmap.png` | 常见词热力图 |
| `particle_all_comparison.png` | 全量常见词对比 |
| `emoji_distribution.png` / `punctuation_heatmap.png` | emoji 与标点使用 |
| `sentiment_comparison.png` / `sentiment_box.png` | 情感对比与分布 |
| `time_distribution.png` / `month_distribution.png` | 发布时间分布 |
| `insights/publish_rhythm.png`、`lexicon_diffusion.png`、`sentence_rhythm.png`、`duplicate_content.png` | 深潜分析图表（发布节奏、词表扩散、句子节奏、重复内容） |
| `insights/framing_formulaicity.png`、`framing_gradient.png` | 框架模板化图表（开篇/收尾集中度、收尾符号构成、P(AI) 梯度、季度扩散） |
| `insights/title_detector.png`、`title_dynamics.png`、`title_formulaicity.png` | 标题端分析图表（特征权重消融、半年度动态与互动、标题前缀集中度） |
| `insights/diversity_burstiness.png`、`diversity_lexical.png` | 文本多样性图表（句长分布/burstiness ECDF、MTLD 小提琴图与 Heaps 曲线） |
| `insights/detector_calibration.png`、`detector_errors.png`、`detector_uncertainty.png` | 检测器诊断图（可靠性曲线/阈值权衡、长度与领域错误切片、置信度分层时间线） |
| `insights/family_lexicon.png` | 模型家族词表指纹（词×家族 log-odds 热力图、AI 味词密度） |
| `insights/engagement_structure.png` | 互动结构分解（零互动密度、基尼集中度、收藏/点赞比与十分位趋势） |
| `insights/engagement_drivers.png` | 互动惩罚的逐特征载体（单变量秩相关 ρ、多元固定效应 β 与整群 bootstrap CI、点赞-收藏偏效应一致性） |
| `insights/author_structure.png`、`author_within.png` | 作者层分解图（选择检查 ECDF、作者发文集中度、作者间梯度、斜率森林图、人内配对散点与 Δ 分布） |
| `insights/symbol_hashtag.png`、`symbol_gradient.png`、`symbol_emoji.png` | 标签/表情生态图（标签结构、P(AI) 梯度与专属标签命中、emoji 与 [xxR] 偏好） |
| `insights/near_duplicate.png` | 近重复/模板复用图（各子集最近邻相似度生存曲线、子集内近重复率、跨子集对矩阵、P(AI) 梯度、季度趋势、簇规模分布） |
| `output/` | 领域、话题、互动等探索性图表合集 |
| `paper_figs/` | 论文管线图表（检测器、漂移、互动、稳健性） |

### 分析脚本

**论文管线（顺序执行，见 [paper.md](paper.md) 附录 A）：**
`paper_lexicon.py` → `paper_features.py` → `paper_model_fingerprint.py` → `paper_detector.py` → `paper_drift.py` → `paper_extra_figs.py` → `paper_engagement.py` → `paper_robustness.py`

**探索性分析脚本：**

- `complete_analysis.py` / `word_freq_analysis.py` - 数据集概览与词频分析
- `particle_analysis.py` - 常见词（助词/虚词）分析
- `word_correlation.py` / `word_correlation_extended.py` - 词频相关性
- `analyze_domains.py` / `analyze_per_domain.py` - 领域分布与分领域词频
- `sentiment_analysis.py` / `stat_tests.py` - 情感与统计检验
- `emoji_analysis.py` / `time_analysis.py` / `extended_analysis.py` - emoji、时间与扩展特征
- `topic_comparison_analysis.py` / `advanced_topic_comparison.py` / `new_topic_comparison.py` - 话题对比
- `unique_comparison.py` / `internal_detailed_analysis.py` / `internal_diff_analysis.py` - 独特性与内部差异
- `deep_dive_analysis.py` - 发布节奏 / 词表扩散 / 句子节奏 / 重复内容深潜分析（输出 `insights/`）
- `framing_analysis.py` - 开篇/收尾框架模板化与 AI 风格关联（输出 `insights/framing_*.png`、`framing_results.json/md`）
- `title_style_analysis.py` - 标题端 AI 味：标题特征消融、模型标题像人度、标题互动与模板（输出 `insights/title_*.png`、`title_style_results.json/md`）
- `diversity_analysis.py` - 句长 burstiness / MTLD / Heaps β 多样性与节奏分析（输出 `insights/diversity_*.png`、`diversity_results.json/md`）
- `detector_diagnostics.py` - 检测器概率校准 / 阈值敏感性 / 错分切片 / 探索集不确定性（输出 `insights/detector_*.png`、`detector_diagnostics_results.json/md`）
- `family_lexicon_analysis.py` - AIGC 子集内家族/单模型词表指纹（输出 `insights/family_lexicon.png`、`family_lexicon_results.json/md`）
- `engagement_structure_analysis.py` - 「AI 味惩罚」的互动结构分解：零互动密度 / 基尼集中度 / 互动构成比 / 单元内秩相关（输出 `insights/engagement_structure.png`、`engagement_structure_results.json/md`）
- `engagement_drivers_analysis.py` - 「AI 味惩罚」的逐特征载体：37 个特征的单元内秩相关 + 月份×领域固定效应多元回归（整群 bootstrap CI）+ P(AI) 净效应（输出 `insights/engagement_drivers.png`、`engagement_drivers_results.json/md`）
- `author_style_analysis.py` - 作者层分解：作者集中度与 ICC、作者间/人内固定效应斜率（整群 bootstrap CI）、同作者配对对照（输出 `insights/author_*.png`、`author_style_results.json/md`）
- `user_clustering.py` - 用户风格聚类（输出 `user_clustering_results.md`）
- `symbol_analysis.py` - 标签/表情生态：结构普及率、AI 专属符号、P(AI) 梯度与季度扩散、标签×互动（输出 `insights/symbol_*.png`、`symbol_results.json/md`）
- `near_duplicate_analysis.py` - 近重复/模板复用：字符 5-gram Jaccard 全量扫描（bottom-k 候选 + 精确校验）、跨子集模板迁移、P(AI) 梯度、季度与簇结构（输出 `insights/near_duplicate.png`、`near_duplicate_results.json/md`）
- `quick_start.py` - BERT 基线训练

深潜分析套件可一键顺序执行：`python3 run_fork_pipeline.py`（`--list` / `--from N` / `--only 脚本名`，编排 deep_dive / framing / title_style / diversity / detector_diagnostics / family_lexicon / engagement_structure / engagement_drivers / author_style / user_clustering / symbol / near_duplicate 共 12 步，输出统一落在 `insights/`）。两个管线运行器共用 `pipeline_common.py` 的步骤选择与顺序执行实现（含未知脚本名校验），其单元测试见 `tests/test_pipeline_common.py`。

所有脚本使用 `analysis_utils.py` 中的统一数据读取、中文字体与检测器打分工具（含概率校准 `reliability_bins` / `expected_calibration_error` / `prior_corrected_proba` 与阈值指标 `threshold_metrics`）；其单元测试见 `tests/`。

中文字体由 `setup_chinese_fonts()` 自动解析：依次扫描 `REDNOTE_FONT_DIR`、`assets/fonts/`、系统字体目录（Noto Sans CJK/SC、思源黑体、文泉驿等），找不到时回退下载 Noto Sans SC（另含单色 emoji 字体）并缓存到 `~/.cache/rednote-vibe/fonts`；也可用 `REDNOTE_FONT_PATH` 直接指定字体文件，或设 `REDNOTE_FONT_DOWNLOAD=0` 禁用联网下载。写回 CSV 时使用内容比较（`write_csv_gz_if_changed`），内容未变则不触碰文件；`paper_drift.py --figs-only` 可只重绘图表、完全不写 `paper_exploring_scores.csv.gz`。

---

## Supported Tasks

### 1. AIGT Classification (Binary)
Distinguish between human-written and AI-generated content.

### 2. AI Provider Identification (6-way)
Identify the source among six major AI providers:
- OpenAI
- Google
- Anthropic
- DeepSeek
- Qwen
- GLM


### 3. Model Identification (17-way)
Fine-grained identification among 17 specific AI models.



## Citation

If you use this dataset in your research, please cite our paper:

```bibtex
@misc{li2025rednote,
      title={RedNote-Vibe: A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Social Media}, 
      author={Yudong Li and Yufei Sun and Yuhan Yao and Peiru Yang and Wanyue Li and Jiajun Zou and Yongfeng Huang and Linlin Shen},
      year={2025},
      eprint={2509.22055},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22055}, 
}
```

## License

This dataset is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.
