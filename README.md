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
pip install torch transformers scikit-learn pandas numpy datasets
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

### 生成的可视化图表

| 文件 | 描述 |
|------|------|
| `content_length_distribution.png` | 内容长度分布 |
| `title_top_words.png` | 标题高频词 |
| `desc_top_words.png` | 正文高频词 |
| `domain_distribution.png` | 域名分布 |
| `scatter_*.png` | 词频相关性散点图 |
| `particle_distribution_comparison.png` | 常见词分布对比 |
| `particle_mean_comparison.png` | 常见词平均使用率 |
| `particle_heatmap.png` | 常见词热力图 |
| `particle_all_comparison.png` | 全量常见词对比 |

### 分析脚本

- `complete_analysis.py` - 完整数据集分析
- `word_freq_analysis.py` - 词频分析
- `particle_analysis.py` - 常见词（助词/虚词）分析
- `quick_start.py` - 基线模型训练

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
