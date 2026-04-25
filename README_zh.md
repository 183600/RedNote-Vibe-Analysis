# RedNote-Vibe 数据集

[English](README.md) | [中文](README_zh.md)

**社交媒体中 AI 生成文本时序动态的数据集** [arxiv](https://arxiv.org/abs/2509.22055)

![数据集概览](overview.png)

## 摘要

大语言模型（LLMs）的快速发展使得社交媒体平台上出现了大量的 AI 生成文本（AIGT），由此带来了独特的研究挑战：这些内容的动态变化由用户互动驱动，并随时间不断演化。然而，现有数据集主要面向静态的 AIGT 检测任务。为此，我们构建了 **RedNote-Vibe**——首个面向社交媒体 AIGT 分析的纵向（跨越 5 年）数据集。该数据集采集自小红书平台，包含用户互动指标（如点赞、评论）以及从 LLM 出现之前直至 2025 年 7 月的时间戳，可支持对 AIGT 的时序动态及用户互动模式开展深入研究。

## 数据集结构

所有数据文件均可通过以下链接获取：

**[从 Google Drive 下载](https://drive.google.com/drive/folders/1T8JV-DmIo7SI8pBOPeuiQvXzFwe8ns--?usp=sharing)**

目录内容包括：

- `training_set_human.jsonl` — 人类撰写的帖子（LLM 出现前，2022 年 11 月之前）
- `training_set_aigc.jsonl` — 由 17 个 LLM 生成的帖子
- `exploration_set.jsonl` — LLM 出现后的帖子（2023–2025），用于时序分析

如希望获取后续更新通知，请 **Star 本仓库订阅**。

### 数据统计

数据集涵盖 11 个内容领域。下表汇总了各领域的统计信息：帖子数量（#）、平均文本长度（Length）、平均标签数量（#Tags），以及平均用户互动指标（点赞 Likes、评论 Comm.、收藏 Colls.）。

| 领域       |    数量 |  长度 | 标签数 |  点赞 |   评论 |   收藏 |
|------------|------:|------:|------:|------:|------:|------:|
| 健康       | 19.2k | 333.6 |   7.6 | 2814.3 |   97.1 | 2468.8 |
| 时尚       | 17.7k | 191.2 |  10.9 | 1850.5 |   62.5 |  681.3 |
| 美食       | 10.5k | 398.4 |   3.0 |   66.2 |   11.9 |   45.0 |
| 职场       | 11.7k | 406.9 |  10.0 | 2047.2 |  102.6 | 1445.9 |
| 宠物       |  3.9k | 433.2 |   2.7 |   57.5 |   12.6 |   34.1 |
| 教育       |  5.9k | 496.0 |   3.7 |   55.9 |    6.1 |   38.3 |
| 运动       | 17.7k | 528.8 |   3.2 |   60.1 |    9.8 |   36.4 |
| 情感       | 22.5k | 308.2 |   7.1 | 2663.6 |  275.8 |  593.9 |
| 旅行       | 14.8k | 452.8 |  10.7 | 1102.4 |   86.6 |  662.0 |
| 心理       | 16.0k | 396.5 |  10.5 | 2348.6 |  217.7 | 1025.8 |
| 其他       | 16.9k | 220.9 |   8.9 |  208.8 | 1266.2 | 3225.0 |

### 数据格式

每条数据包含以下字段：

```json
{
  "note_title": "帖子标题",
  "local_time": "发布时间戳 (YYYYMMDDHH)",
  "note_content": "正文内容",
  "likes": 123,
  "collections": 45,
  "comments": 67,
  "domain": "内容分类",
  "model_family": "AI 提供方（仅 AIGC 数据）",
  "model": "具体 AI 模型（仅 AIGC 数据）"
}
```

## 🚀 快速开始

### 环境依赖

```bash
pip install torch transformers scikit-learn pandas numpy datasets
```

### 基本用法

运行快速开始示例，训练一个用于 AIGT 检测的 BERT 模型：

```bash
python quick_start.py
```

该脚本将完成以下步骤：
1. 加载人类与 AI 生成的数据集
2. 构建均衡的训练 / 验证 / 测试集划分
3. 微调 BERT 模型进行二分类
4. 评估性能并展示预测样例
5. 保存训练好的模型以供后续使用

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

### 生成的可视化图表

| 文件 | 描述 |
|------|------|
| `content_length_distribution.png` | 内容长度分布 |
| `title_top_words.png` | 标题高频词 |
| `desc_top_words.png` | 正文高频词 |
| `domain_distribution.png` | 域名分布 |
| `scatter_*.png` | 词频相关性散点图 |

### 分析脚本

- `complete_analysis.py` - 完整数据集分析
- `word_freq_analysis.py` - 词频分析
- `quick_start.py` - 基线模型训练

---

## 支持的任务

### 1. AIGT 二分类
区分人类撰写与 AI 生成的内容。

### 2. AI 提供方识别（6 类）
识别下列主要 AI 提供方：
- OpenAI
- Google
- Anthropic
- DeepSeek
- Qwen

### 3. 模型识别（17 类）
对 17 个具体 AI 模型进行细粒度识别。

## 引用

如果本数据集对您的研究有所帮助，请引用我们的论文：

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

本数据集基于 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 协议发布。
