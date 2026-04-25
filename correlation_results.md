# 词频相关性分析结果

## 不考虑标题（只看desc/note_content）

|                    | training_set_hu | training_set_ai | exploring_set |
|--------------------|-----------------|-----------------|---------------|
| training_set_human | 1.0000          | 0.6996          | 0.6497        |
| training_set_aigc  | 0.6996          | 1.0000          | 0.9925        |
| exploring_set       | 0.6497          | 0.9925          | 1.0000        |

## 只考虑标题（note_title）

|                    | training_set_hu | training_set_ai | exploring_set |
|--------------------|-----------------|-----------------|---------------|
| training_set_human | 1.0000          | 0.0927          | 0.4372        |
| training_set_aigc  | 0.0927          | 1.0000          | 0.1217        |
| exploring_set       | 0.4372          | 0.1217          | 1.0000        |

## 散点图文件

- scatter_desc_only_training_set_human_vs_training_set_aigc.png
- scatter_desc_only_training_set_human_vs_exploring_set.png
- scatter_desc_only_training_set_aigc_vs_exploring_set.png
- scatter_title_only_training_set_human_vs_training_set_aigc.png
- scatter_title_only_training_set_human_vs_exploring_set.png
- scatter_title_only_training_set_aigc_vs_exploring_set.png