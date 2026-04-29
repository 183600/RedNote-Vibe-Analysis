# 基于RedNote-Vibe 数据集的分析

[English](README.md) | [中文](README_zh.md)

## 数据集

**[数据集github](https://github.com/ydli-ai/RedNote-Vibe.git)**

所有数据文件均可通过以下链接获取：

**[从 Google Drive 下载](https://drive.google.com/drive/folders/1T8JV-DmIo7SI8pBOPeuiQvXzFwe8ns--?usp=sharing)**

目录内容包括：

- `training_set_human.jsonl` — 人类撰写的帖子（LLM 出现前，2022 年 11 月之前）
- `training_set_aigc.jsonl` — 由 17 个 LLM 生成的帖子
- `exploration_set.jsonl` — LLM 出现后的帖子（2023–2025），用于时序分析

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

## 参考资料

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

## 许可证

基于 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 协议发布。
