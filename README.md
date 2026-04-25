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


## 📊 Fork Analysis

This fork includes comprehensive data analysis of the RedNote-Vibe dataset. Key findings and visualizations are provided below.

### Dataset Overview

| Dataset | Records | Avg Content Length | Avg Title Length |
|---------|---------|-------------------|------------------|
| training_set_human | 51,878 | 488.1 | 15.7 |
| training_set_aigc | 7,254 | 347.7 | 18.3 |
| exploring_set | 91,517 | 322.6 | 15.0 |

### Word Frequency Correlation Matrix

**Content (note_content):**

|                    | human | aigc | exploring |
|--------------------|-------|------|------------|
| training_set_human | 1.00  | 0.67 | 0.83       |
| training_set_aigc  | 0.67  | 1.00 | 0.64       |
| exploring_set      | 0.83  | 0.64 | 1.00       |

**Title (note_title):**

|                    | human | aigc | exploring |
|--------------------|-------|------|------------|
| training_set_human | 1.00  | 0.46 | 0.53       |
| training_set_aigc  | 0.46  | 1.00 | 0.45       |
| exploring_set      | 0.53  | 0.45 | 1.00       |

### Key Findings

1. **Title is a strong discriminator**: Human vs AIGC correlation in titles is only 0.46, much lower than in content (0.67)
2. **Content length**: Human posts are longest (488 chars), AIGC medium (348), exploring shortest (323)
3. **Title length**: AIGC titles are longest (18.3 chars), more formal and detailed
4. **High-frequency words**: Human uses more emojis; AIGC frequently uses "姐妹们" (sisters)

### Generated Visualizations

| File | Description |
|------|-------------|
| `content_length_distribution.png` | Content length distribution across datasets |
| `title_top_words.png` | Top words in titles |
| `desc_top_words.png` | Top words in content |
| `domain_distribution.png` | Domain distribution |
| `scatter_*.png` | Word frequency correlation scatter plots |

### Analysis Scripts

- `complete_analysis.py` - Full dataset analysis
- `word_freq_analysis.py` - Word frequency analysis
- `quick_start.py` - Baseline model training

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
