# RedNote Vibe Dataset 综合分析报告

## 1. 数据集概览

| 数据集 | 记录数 |
|--------|--------|
| training_set_human.jsonl | 51,878 |
| training_set_aigc.jsonl | 7,254 |
| exploring_set.jsonl | 91,517 |

## 2. 词频相关性矩阵

### 2.1 正文内容 (desc/note_content)

|                    | training_set_hu | training_set_ai | exploring_set |
|--------------------|-----------------|-----------------|---------------|
| training_set_human | 1.0000          | 0.6682          | 0.8339        |
| training_set_aigc  | 0.6682          | 1.0000          | 0.6389        |
| exploring_set       | 0.8339          | 0.6389          | 1.0000        |

### 2.2 标题内容 (note_title)

|                    | training_set_hu | training_set_ai | exploring_set |
|--------------------|-----------------|-----------------|---------------|
| training_set_human | 1.0000          | 0.4620          | 0.5261        |
| training_set_aigc  | 0.4620          | 1.0000          | 0.4468        |
| exploring_set       | 0.5261          | 0.4468          | 1.0000        |

### 2.3 关键发现
- **正文**：Human与exploring相关性最高(0.83)，AIGC与其他两个相关性较低(0.64-0.67)
- **标题**：各数据集间相关性均较低(0.44-0.53)，human vs aigc仅0.46
- 标题的区分度更高，是识别AI生成内容的重要特征

## 3. 高频词分析

### 3.1 正文高频词 Top 10

| 排名 | training_set_human | training_set_aigc | exploring_set |
|------|-------------------|-------------------|---------------|
| 1 | 赞r | 姐妹们 | 哭惹r |
| 2 | 哭惹r | 地址 | 笑哭r |
| 3 | 偷笑r | 赞r | 害羞r |
| 4 | 笑哭r | 偷笑r | 赞r |
| 5 | 派对r | 哭惹r | 偷笑r |
| 6 | 害羞r | 捂脸r | 派对r |
| 7 | 萌萌哒r | 笑哭r | 一r |
| 8 | 哇r | look | 失望r |
| 9 | 失望r | 记住 | 哇r |
| 10 | 捂脸r | 薯队长 | 星r |

### 3.2 标题高频词 Top 10

| 排名 | training_set_human | training_set_aigc | exploring_set |
|------|-------------------|-------------------|---------------|
| 1 | ootd | 姐妹们 | 极氪 |
| 2 | 好物分享 | 救命 | 一衣多穿 |
| 3 | 有道词典笔打卡 | ootd | ootd |
| 4 | 早餐 | 姐妹们冲鸭 | 正念打卡 |
| 5 | kg | 夏日必备 | 空心病 |
| 6 | plog | 谁懂啊 | 每日正念 |
| 7 | 第天 | 哭惹r | 正念日记 |
| 8 | vlog | 啊啊啊 | oppo |
| 9 | 探店 | 冲鸭 | 救命 |
| 10 | no | 爱心 | bm |

## 4. 内容长度统计

### 4.1 正文长度

| 数据集 | 平均长度 | 中位数 |
|--------|---------|--------|
| training_set_human | 488.1 | 419.0 |
| training_set_aigc | 347.7 | 318.0 |
| exploring_set | 322.6 | 187.0 |

发现：Human内容最长，exploring最短

### 4.2 标题长度

| 数据集 | 平均长度 | 中位数 |
|--------|---------|--------|
| training_set_human | 15.7 | 16.0 |
| training_set_aigc | 18.3 | 18.0 |
| exploring_set | 15.0 | 15.0 |

发现：AIGC标题最长(18.3)，更正式详细

## 5. 生成的图表文件

### 5.1 散点图
- scatter_desc_only_training_set_human_vs_training_set_aigc.png
- scatter_desc_only_training_set_human_vs_exploring_set.png
- scatter_desc_only_training_set_aigc_vs_exploring_set.png
- scatter_title_only_training_set_human_vs_training_set_aigc.png
- scatter_title_only_training_set_human_vs_exploring_set.png
- scatter_title_only_training_set_aigc_vs_exploring_set.png

### 5.2 高频词图
- desc_top_words.png (正文)
- title_top_words.png (标题)

### 5.3 长度分布图
- content_length_distribution.png (正文)
- title_length_distribution.png (标题)

### 5.4 其他
- domain_domain_distribution.png (域名分布)
- correlation_heatmaps.png (相关性热力图)

## 6. 总结

1. **正��内容**：AIGC与human相关性0.67，exploring与human相关性最高0.83
2. **标题内容**：各数据集间相关性低(0.44-0.53)，是区分AI内容的关键特征
3. **内容长度**：Human最长(488字符)，AIGC次之(348)，exploring最短(323)
4. **标题长度**：AIGC最长(18字符)，human和exploring相近(15-16字符)
5. **高频词差异**：Human使用更多emoji(AIGC常用"姐妹们")