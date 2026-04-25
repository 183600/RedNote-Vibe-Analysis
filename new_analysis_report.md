# 新增分析报告

## 1. 标题模式分析

### 1.1 标题长度

| 数据集 | 平均标题长度 |
|--------|-------------|
| training_set_human | 15.14 chars |
| training_set_aigc | 18.31 chars |
| exploring_set | 14.45 chars |

**发现**: AIGC标题最长(18.31)，更正式详细；exploring最短(14.45)

### 1.2 标题模式分布

| 模式 | training_set_human | training_set_aigc | exploring_set |
|------|-------------------|-------------------|---------------|
| other | 81.4% | 78.1% | 66.5% |
| possessive (的...) | 17.2% | 20.4% | 29.0% |
| numbered | 1.0% | 0.2% | 0.4% |
| question_end | 0.4% | 1.0% | 0.3% |
| mbti | 0% | 0.2% | 3.8% |

**发现**: 
- exploring_set大量使用"女性成长"、"自我提升"等所有格标题(29%)
- exploring_set有明显MBTI相关标题(3.8%)
- AIGC更正式，疑问句标题略多(1.0%)

---

## 2. URL/链接分析

### 2.1 链接数量

| 数据集 | 总链接数 |
|--------|---------|
| training_set_human | 17 |
| training_set_aigc | 3 |
| exploring_set | 4 |

**发现**: Human数据包含更多外部链接

### 2.2 顶级域名分布

| training_set_human | training_set_aigc | exploring_set |
|-------------------|-------------------|---------------|
| www.ucl.ac.uk (2) | visa.nadra.gov.pk (1) | mbti.xiaguhou.com (1) |
| www.shanghairanking.cn (1) | invol.co (1) | www.16personalities.com (1) |
| www.sheffield.ac.uk (1) | shope.ee (1) | www.jungus.cn (1) |
| youtu.be (1) | | totypes.com (1) |
| b23.tv (1) | | |

**发现**: Human内容包含更多学术/教育类链接

---

## 3. 数字使用分析

| 数据集 | 数字总数 | 含价格信息 | 含数量词 |
|--------|---------|-----------|---------|
| training_set_human | 24,570 | 39 | 3,506 |
| training_set_aigc | 18,459 | 9 | 2,922 |
| exploring_set | 9,796 | 3 | 1,825 |

**发现**: Human内容数字信息更丰富，价格和数量词更多

---

## 4. 结构特征分析

### 4.1 段落与列表

| 数据集 | 平均段落数 | 有编号列表 | 有项目符号 |
|--------|-----------|-----------|-----------|
| training_set_human | 5.91 | 4.0% | 9.1% |
| training_set_aigc | 6.79 | 5.2% | 8.5% |
| exploring_set | 10.97 | 7.5% | 6.3% |

**发现**: 
- exploring段落最多(10.97)，内容最长
- AIGC有更多结构化列表(5.2%)

---

## 5. 句子模式分析

| 数据集 | 平均词/句 | 词汇丰富度 |
|--------|----------|-----------|
| training_set_human | 14.91 | 0.973 |
| training_set_aigc | 12.19 | 0.975 |
| exploring_set | 14.95 | 0.977 |

**发现**: AIGC句子较短(12.19词/句)，词汇丰富度相近

---

## 6. 词汇重复率

| 数据集 | 平均重复率 |
|--------|-----------|
| training_set_human | 0.113 |
| training_set_aigc | 0.145 |
| exploring_set | 0.120 |

**发现**: AIGC词汇重复率最高(0.145)，Human最低(0.113)

---

## 7. 语言特征分析

| 特征 | training_set_human | training_set_aigc | exploring_set |
|------|-------------------|-------------------|---------------|
| 感叹号(!) | 469 | 169 | 129 |
| 问号(?) | 178 | 44 | 63 |
| 省略号(...) | 335 | 501 | 193 |
| 大写字母 | 24,073 | 18,110 | 14,445 |
| 引号 | 5,744 | 3,679 | 13,621 |

**发现**:
- Human使用更多感叹号(情感表达)
- AIGC使用更多省略号(...)
- exploring_set引号使用最多(可能包含更多引用)

---

## 8. 总结：新发现的关键差异

1. **标题**: AIGC标题更长(18.3字)，exploring有大量"成长"类所有格标题
2. **链接**: Human包含更多外部链接(学术/教育类)
3. **数字**: Human数字信息更丰富
4. **结构**: exploring段落最多，AIGC结构化列表最多
5. **句子**: AIGC句子更短，词汇重复率更高(0.145 vs 0.113)
6. **情感**: Human更多感叹号，AIGC更多省略号
7. **引用**: exploring引号使用最多

这些新特征可作为区分AI生成内容与人类内容的辅助指标。