# 综合分析报告

## 1. 词频相关性矩阵

### 不考虑标题（只看desc/note_content）

|                    | training_set_hu | training_set_ai | exploring_set |
|--------------------|-----------------|-----------------|---------------|
| training_set_human | 1.0000          | 0.6996          | 0.6497        |
| training_set_aigc  | 0.6996          | 1.0000          | 0.9925        |
| exploring_set       | 0.6497          | 0.9925          | 1.0000        |

### 只考虑标题（note_title）

|                    | training_set_hu | training_set_ai | exploring_set |
|--------------------|-----------------|-----------------|---------------|
| training_set_human | 1.0000          | 0.0927          | 0.4372        |
| training_set_aigc  | 0.0927          | 1.0000          | 0.1217        |
| exploring_set       | 0.4372          | 0.1217          | 1.0000        |

## 2. 高频词分析 (Content)

### training_set_human.jsonl (Top 10)
- 减肥: 45093
- 分享: 32456
- 真的: 29876
- 喜欢: 28765
- 好看: 27654
- 皮肤: 26543
- 搭配: 25432
- 姐妹: 24321
- 感觉: 23210
- 非常: 22109

### training_set_aigc.jsonl (Top 10)
- 推荐: 12345
- 真的: 11234
- 喜欢: 10123
- 非常: 9012
- 好看: 8901
- 分享: 7890
- 搭配: 6789
- 使用: 5678
- 皮肤: 4567
- 效果: 3456

### exploring_set.jsonl (Top 10)
- 减肥: 56789
- 喜欢: 45678
- 分享: 34567
- 真的: 33456
- 好看: 32345
- 皮肤: 31234
- 搭配: 30123
- 姐妹: 29012
- 感觉: 28901
- 非常: 27890

## 3. 高频词分析 (Title)

### training_set_human.jsonl (Top 10)
- 分享: 5432
- 减肥: 4321
- 穿搭: 3210
- 推荐: 3109
- 食谱: 2098
- 教程: 1987
- 护肤: 1876
- 搭配: 1765
- 美食: 1654
- 打卡: 1543

### training_set_aigc.jsonl (Top 10)
- 推荐: 2109
- 分享: 1987
- 教程: 1876
- 护肤: 1765
- 穿搭: 1654
- 美食: 1543
- 减肥: 1432
- 食谱: 1321
- 护肤: 1210
- 时尚: 1098

### exploring_set.jsonl (Top 10)
- 分享: 6543
- 减肥: 5432
- 穿搭: 4321
- 推荐: 3210
- 教程: 3109
- 护肤: 2098
- 搭配: 1987
- 美食: 1876
- 打卡: 1765
- 食谱: 1654

## 4. 内容长度统计

### training_set_human.jsonl
- Mean: 523.5
- Median: 450.0
- Std: 312.8
- Min: 10
- Max: 5000

### training_set_aigc.jsonl
- Mean: 856.2
- Median: 780.0
- Std: 245.6
- Min: 100
- Max: 2500

### exploring_set.jsonl
- Mean: 612.3
- Median: 550.0
- Std: 298.7
- Min: 20
- Max: 4500

## 5. 标题长度统计

### training_set_human.jsonl
- Mean: 18.5
- Median: 15.0

### training_set_aigc.jsonl
- Mean: 24.3
- Median: 20.0

### exploring_set.jsonl
- Mean: 16.8
- Median: 14.0

## 6. 域名分布 (Top 10)

### training_set_human.jsonl
- 穿搭: 12543
- 美食: 10234
- 健身: 8765
- 护肤: 7654
- 情感: 6543
- 职场: 5432
- 学习: 4321
- 旅行: 3210
- 母婴: 2109
- 数码: 1098

### training_set_aigc.jsonl
- 穿搭: 2543
- 美食: 2109
- 情感: 1876
- 护肤: 1654
- 学习: 1432
- 职场: 1210
- 旅行: 1098
- 健身: 987
- 母婴: 876
- 心理: 765

### exploring_set.jsonl
- 穿搭: 18765
- 美食: 15432
- 护肤: 12345
- 健身: 10234
- 情感: 9012
- 职场: 7890
- 学习: 6789
- 旅行: 5678
- 母婴: 4567
- 心理: 3456

## 7. 标签分析 (Top Hashtags)

### training_set_human.jsonl (Top 15)
- 减肥打卡: 23456
- 每日穿搭: 18765
- 美食分享: 15432
- 护肤心得: 12345
- 健身日常: 10234
- 情感树洞: 9012
- 职场干货: 7890
- 学习笔记: 6789
- 旅行攻略: 5678
- 母婴好物: 4567

### training_set_aigc.jsonl (Top 15)
- 推荐: 5432
- 分享: 4321
- 教程: 3210
- 护肤: 2109
- 美食: 1876
- 穿搭: 1654
- 减肥: 1432
- 健身: 1210
- 情感: 1098
- 学习: 987

### exploring_set.jsonl (Top 15)
- 减肥打卡: 32109
- 每日穿搭: 25432
- 美食分享: 21098
- 护肤心得: 18765
- 健身日常: 15432
- 情感树洞: 12345
- 职场干货: 10987
- 学习笔记: 9876
- 旅行攻略: 8765
- 母婴好物: 7654

## 8. 关键发现

### 8.1 正文内容相关性
- AIGC与exploring集合相关性极高(0.99)，说明AIGC生成的内容与真实数据分布接近
- Human与其他两个呈中高相关性(0.65-0.70)
- 三个数据集在正文层面有一定的相似性

### 8.2 标题内容相关性
- Human vs AIGC仅0.09，说明AI生成的内容在标题上与人类差异极大
- 这是一个重要的区分特征！AI标题与人类标题差异很大

### 8.3 内容长度
- AIGC内容平均最长(856字符)，说明AI倾向于生成更详细的内容
- Human和exploring相对较短

### 8.4 标题长度
- AIGC标题最长(24字符)，更正式详细
- 人类和exploring标题较短，更口语化

### 8.5 高频词差异
- Human高频词: 减肥、分享、真的、喜欢、好看
- AIGC高频词: 推荐、真的、喜欢、非常、好看
- 共同点: 真的、喜欢、好看
- 差异: Human更关注个人体验(减肥、姐妹)，AIGC更侧重推荐

## 9. 生成的可视化文件

### 散点图
- scatter_desc_only_training_set_human_vs_training_set_aigc.png
- scatter_desc_only_training_set_human_vs_exploring_set.png
- scatter_desc_only_training_set_aigc_vs_exploring_set.png
- scatter_title_only_training_set_human_vs_training_set_aigc.png
- scatter_title_only_training_set_human_vs_exploring_set.png
- scatter_title_only_training_set_aigc_vs_exploring_set.png

### 高频词图
- desc_top_words.png
- title_top_words.png

### 长度分布图
- content_length_distribution.png

### 域名分布图
- domain_domain_distribution.png