# -*- coding: utf-8 -*-
"""topic_comparison_analysis 情感计数预筛的等价性测试（不触碰研究数据）。

count_sentiment_words 在跑正则前先做「片段不含任何情感词用字则跳过」的
预筛；本测试用无预筛的参考实现逐一对照，保证预筛不改变任何计数。
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from topic_comparison_analysis import (_SENTIMENT_CHARS, NEGATIVE_WORDS,
                                        POSITIVE_WORDS, SENTIMENT_PATTERN,
                                        clean_tokenize, count_sentiment_words)


def reference_count(token):
    """无预筛的历史实现。"""
    pos = neg = 0
    for m in SENTIMENT_PATTERN.finditer(token):
        if m.lastgroup == "pos":
            pos += 1
        else:
            neg += 1
    return pos, neg


class TestSentimentPrefilter(unittest.TestCase):
    def test_prefilter_char_set_covers_all_words(self):
        for w in POSITIVE_WORDS | NEGATIVE_WORDS:
            for ch in w:
                self.assertIn(ch, _SENTIMENT_CHARS)

    def test_equivalent_to_reference_on_battery(self):
        tokens = [
            "好", "喜欢", "不好", "超级喜欢", "今天好开心啊", "美美的",
            "难受", "太累了", "差评", "完美", "细心", "焦虑", "无感",
            "abc", "ABC", "hello世界", "12345", "？？？", "………", "话题",
            "喜欢喜欢喜欢", "好好好", "难过死了", "又开心又难过",
            "今天天气真不错", "不错不错", "有点烦", "十分满意", "感动哭了",
        ]
        for token in tokens:
            with self.subTest(token=token):
                self.assertEqual(count_sentiment_words(token),
                                 reference_count(token))

    def test_clean_tokenize_drops_stopwords_and_short_runs(self):
        joined = " ".join(clean_tokenize("今天好开心啊，#话题 好！"))
        self.assertNotIn("好", joined.split())  # 单字片段（含停用词）不保留
        self.assertIn("今天好开心啊", joined)


if __name__ == "__main__":
    unittest.main()
