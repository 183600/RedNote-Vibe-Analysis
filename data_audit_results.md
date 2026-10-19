# 数据质量审计结果（Data Audit）

> 由 `data_audit.py` 全量运行生成；只读研究资产，不做任何修改。行序校验使用原始 month/domain/likes/collections/comments，`content_len` 差异作为清洗版本漂移单独报告。

## 1 文件清单

| 资产 | 存在 | 大小 (MB) |
|---|---:|---:|
| `training_set_human.jsonl` | 是 | 76.20 |
| `training_set_aigc.jsonl` | 是 | 7.69 |
| `exploring_set.jsonl` | 是 | 94.50 |
| `paper_features_human.csv.gz` | 是 | 2.68 |
| `paper_features_aigc.csv.gz` | 是 | 0.36 |
| `paper_features_exploring.csv.gz` | 是 | 3.78 |
| `paper_exploring_scores.csv.gz` | 是 | 2.25 |
| `paper_detector_model.json` | 是 | 0.00 |

## 2 语料质量

| 语料 | 记录数 | 空正文 | 短正文 | 缺 note_id | 重复 note_id 行 | 重复正文行 | 缺领域 | 时间缺失/无效 | 负互动 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 人类(前LLM) | 51,878 | 0 | 0 | 0 | 206 (206 组) | 283 | 0 | 0 | 0 |
| AIGC(17模型) | 7,254 | 0 | 301 | 7,254 | 0 (0 组) | 0 | 0 | 7,254 | 0 |
| 探索集(后LLM) | 91,517 | 831 | 36,229 | 0 | 31,356 (31,016 组) | 31,578 | 16,978 | 0 | 0 |

> 「重复正文」按原始内容精确哈希匹配；AIGC 无 note_id，重复 note_id 仅统计有人类/探索集字段的记录。

## 3 顶层字段覆盖率

| 字段 | 人类(前LLM) | AIGC(17模型) | 探索集(后LLM) |
|---|---:|---:|---:|
| `note_title` | 100.0% | 100.0% | 100.0% |
| `desc` | 100.0% | — | 54.9% |
| `domain` | 100.0% | 100.0% | 81.4% |
| `local_time` | 100.0% | — | 100.0% |
| `liked_count` | 100.0% | — | 54.9% |
| `collected_count` | 100.0% | — | 54.9% |
| `comments_count` | 100.0% | — | 54.9% |
| `note_id` | 100.0% | — | 100.0% |
| `note_content` | — | 100.0% | 45.1% |
| `model_family` | — | 100.0% | — |
| `model` | — | 100.0% | — |
| `time` | — | — | 54.9% |
| `likes` | — | — | 45.1% |
| `collections` | — | — | 45.1% |
| `comments` | — | — | 45.1% |
| `user_id` | — | — | 36.3% |

## 4 派生文件与行序对齐

| 语料 | JSONL 行 | features 行 | 行数一致 | 月份不一致 | 领域不一致 | 点赞不一致 | 收藏不一致 | 评论不一致 | 文本长度 drift (|差|均值) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 人类(前LLM) | 51,878 | 51,878 | 是 | 0 | 0 | 0 | 0 | 0 | 0.3 |
| AIGC(17模型) | 7,254 | 7,254 | 是 | 0 | 0 | 0 | 0 | 0 | 2.9 |
| 探索集(后LLM) | 91,517 | 91,517 | 是 | 0 | 0 | 0 | 0 | 0 | 18.1 |

- `paper_exploring_scores.csv.gz` 行数 91,517，与 features 行数一致：是；scores/features content_len 不一致 0 条。
- P(AI) 范围 [0.0000, 1.0000]，均值 0.4578，越界 0 条，缺失 0 条。

## 5 features 缺失概览

| 文件 | 行 × 列 | 有缺失的列 |
|---|---:|---|
| `paper_features_human.csv.gz` | 51,878 × 45 | `model_family` 51,878 (100.0%)；`model` 51,878 (100.0%) |
| `paper_features_aigc.csv.gz` | 7,254 × 45 | `month` 7,254 (100.0%) |
| `paper_features_exploring.csv.gz` | 91,517 × 45 | `model_family` 91,517 (100.0%)；`model` 91,517 (100.0%) |

## 6 审计结论

- 三个语料与对应 features 表行数一致，原始月份/领域/互动字段逐条对齐。
- 探索集 scores 与 features 的 `content_len` 完全一致；JSONL 当前 `clean_body` 长度与派生 `content_len` 存在系统性漂移，属于清洗正则版本差异，不是行序错位。
- 短正文主要集中在探索集；重复 note_id/正文在人类与探索集均存在，深潜分析按既有协议去重。
- 本审计只报告观察到的质量事实，不对数据做截断、填补或删除。
