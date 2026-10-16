#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import re
import codecs
from collections import Counter

import jieba

from analysis_utils import iter_jsonl_codecs, load_records_with_total


SAMPLE_LIMIT = 5000

jieba.setLogLevel(60)
TOKEN_KEEP = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def title_tokens(title):
    return [t for t in (tok.strip().lower() for tok in jieba.lcut(str(title)))
            if t and TOKEN_KEEP.search(t)]


def load_jsonl(filepath, limit=SAMPLE_LIMIT, seed=42):
    return load_records_with_total(filepath, sample=limit, seed=seed,
                                   iterator=iter_jsonl_codecs)[0]


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]
    print("=" * 70)
    print("TITLE PATTERN ANALYSIS")
    print("=" * 70)

    all_features = {}

    for f in files:
        data = load_jsonl(f)
        print("\n" + "=" * 50)
        print("Dataset: " + f)
        print("=" * 50)

        title_lengths = []
        title_words = []
        title_patterns = Counter()
        features = {"exclaim": 0, "question": 0, "ellipsis": 0, "caps": 0, "quotes": 0}

        for item in data:
            title = item.get("note_title", "") or item.get("title", "")
            if title:
                title_lengths.append(len(title))

                words = title_tokens(title)
                title_words.extend(words)

                if re.match(r"^\d+[\u3001.\s]", str(title)):
                    title_patterns["numbered"] += 1
                elif re.match(r"^[\u4e00-\u9fa5]+[\u7684]", str(title)):
                    title_patterns["possessive"] += 1
                elif re.search(r"[?？]$", str(title)):
                    title_patterns["question_end"] += 1
                elif re.match(r"^#", str(title)):
                    title_patterns["hashtag_start"] += 1
                elif re.match(r"^\d+$", str(title)):
                    title_patterns["number_only"] += 1
                elif re.search(r"mbti|MBTI", str(title), re.I):
                    title_patterns["mbti"] += 1
                else:
                    title_patterns["other"] += 1

            text = str(item.get("desc", "")) + " " + str(item.get("note_content", ""))
            features["exclaim"] += text.count("!") + text.count("！")
            features["question"] += text.count("?") + text.count("？")
            features["ellipsis"] += (
                text.count("...") + text.count("…") + text.count("。。。"))
            features["caps"] += sum(1 for c in text if c.isupper())
            features["quotes"] += len(re.findall(r'["\u201c\u201d]', text))

        all_features[f] = features

        avg_len = sum(title_lengths) / len(title_lengths) if title_lengths else 0
        unique_words = len(set(title_words))

        print("\n1. TITLE LENGTH:")
        print("   Avg title length: " + str(round(avg_len, 2)) + " chars")

        print("\n2. TITLE VOCAB:")
        print("   Total words: " + str(len(title_words)))
        print("   Unique words: " + str(unique_words))

        print("\n3. TITLE PATTERNS:")
        for pattern, count in title_patterns.most_common(15):
            pct = round(100 * count / len(title_lengths), 1) if title_lengths else 0
            print("   " + pattern + ": " + str(count) + " (" + str(pct) + "%)")

    print("\n" + "=" * 70)
    print("LINGUISTIC FEATURES ANALYSIS")
    print("=" * 70)

    for f in files:
        features = all_features[f]

        print("\n" + f + ":")
        print("   Exclamation marks: " + str(features["exclaim"]))
        print("   Question marks: " + str(features["question"]))
        print("   Ellipsis (...): " + str(features["ellipsis"]))
        print("   Uppercase chars: " + str(features["caps"]))
        print("   Quotes: " + str(features["quotes"]))

    print("\n" + "=" * 70)
    print("Analysis Complete")


if __name__ == "__main__":
    main()
