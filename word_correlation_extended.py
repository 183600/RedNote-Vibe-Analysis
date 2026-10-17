# -*- coding: utf-8 -*-
import json
import random
import re
import math
from collections import defaultdict, Counter
import numpy as np

from analysis_utils import BASE, clean_body


exclude = set([u"话题", u"时尚薯", u"生活薯", u"知识薯", u"美食薯", u"校园薯",
               u"薯队长", u"薯条小助手", u"小红书创作学院"])

# 全文件共用一个预编译 CJK 正则
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

PATH = BASE + "exploring_set_fixed.jsonl"


def _record_words(item):
    text = clean_body(item.get("note_title", "") + " " + (
        item.get("desc") or item.get("note_content") or ""))
    return [w for w in _CJK_RUN_RE.findall(text)
            if len(w) >= 2 and w not in exclude]


def main():
    print("Loading...")
    user_word_total = defaultdict(int)
    word_doc_freq = Counter()

    with open(PATH, "rb") as f:
        for line in f:
            try:
                s = line.decode("utf-8")
                item = json.loads(s)
                user_id = item.get("user_id")
                if not user_id:
                    continue
                words = _record_words(item)
                if words:
                    user_word_total[user_id] += len(words)
                    word_doc_freq.update(set(words))
            except:
                pass

    print("Users: " + str(len(user_word_total)))

    MIN_COUNT = 20
    active_users = {u for u, c in user_word_total.items() if c >= MIN_COUNT}
    print("Active: " + str(len(active_users)))

    MIN_USERS = 30
    top_words = [w for w, c in word_doc_freq.most_common(100) if c >= MIN_USERS]
    print("Words: " + str(len(top_words)))

    top_word_set = set(top_words)
    note_by_user = defaultdict(lambda: defaultdict(int))
    with open(PATH, "rb") as f:
        for line in f:
            try:
                s = line.decode("utf-8")
                item = json.loads(s)
                user_id = item.get("user_id")
                if not user_id or user_id not in active_users:
                    continue
                counts = note_by_user[user_id]
                for w in _record_words(item):
                    if w in top_word_set:
                        counts[w] += 1
            except:
                pass

    word_to_idx = {w: i for i, w in enumerate(top_words)}
    user_ids = random.Random(42).sample(
        sorted(active_users), min(3000, len(active_users))
    )
    n_users = len(user_ids)
    n_words = len(top_words)

    print("Matrix: " + str(n_users) + " x " + str(n_words))

    matrix = []
    for uid in user_ids:
        total = float(user_word_total[uid])
        row = [note_by_user[uid].get(w, 0) / total if total > 0 else 0
               for w in top_words]
        matrix.append(row)

    cols = np.ascontiguousarray(np.array(matrix, dtype=np.float64).T)

    valid = [j for j in range(n_words) if cols[j].std() > 0]
    print("Valid: " + str(len(valid)))

    if len(valid) < 2:
        print("Not enough!")
        raise SystemExit(1)

    print("Computing...")
    # 均值/中心化/范数只依赖单列，预先算一次；原先每对词都重复 O(n) 的
    # 向量运算（中心化与范数各算两遍）。逐对的乘-求和表达式保持不变，
    # 因此相关系数与原实现逐位一致。
    centered = np.empty((len(valid), n_users), dtype=np.float64)
    norms = np.empty(len(valid), dtype=np.float64)
    for i, j in enumerate(valid):
        xc = cols[j] - cols[j].mean()
        centered[i] = xc
        norms[i] = math.sqrt(float((xc * xc).sum()))

    all_results = []
    for j1i in range(len(valid)):
        xc = centered[j1i]
        sx = norms[j1i]
        for j2i in range(j1i + 1, len(valid)):
            yc = centered[j2i]
            sy = norms[j2i]
            r = float((xc * yc).sum()) / (sx * sy) if sx != 0 and sy != 0 else 0.0
            if not math.isnan(r):
                all_results.append((top_words[valid[j1i]], top_words[valid[j2i]], r))

    all_results.sort(key=lambda x: x[2])

    pos = [(w1, w2, r) for w1, w2, r in all_results if r > 0.15]
    pos.sort(key=lambda x: x[2], reverse=True)
    neg = [(w1, w2, r) for w1, w2, r in all_results if r < -0.05]

    print("Pos: " + str(len(pos)) + ", Neg: " + str(len(neg)))

    with open("word_correlation_word_pairs.md", "wb") as f:
        f.write((u"# 跨话题文风词词频相关性分析\n\n").encode('utf-8'))
        f.write((u"分析对象：至少 30 名用户使用过的高频词（跨用户词频占比相关性）\n").encode('utf-8'))
        f.write((u"用户词频归一化为该词在用户总词频中的比例\n\n").encode('utf-8'))
    
        f.write((u"## 词频正相关词组 (r > 0.15)\n\n").encode('utf-8'))
        f.write((u"含义：当用户多用词A时，也倾向多用词B\n\n").encode('utf-8'))
        for w1, w2, r in pos[:30]:
            f.write((u"- **" + w1 + u"** <-> **" + w2 + u"**: " + str(round(r, 3)) + u"\n").encode('utf-8'))
    
        f.write((u"\n## 词频负相关词组 (r < -0.05)\n\n").encode('utf-8'))
        f.write((u"含义：当用户多用词A时，倾向少用词B\n\n").encode('utf-8'))
        for w1, w2, r in neg[:30]:
            f.write((u"- **" + w1 + u"** <-> **" + w2 + u"**: " + str(round(r, 3)) + u"\n").encode('utf-8'))

    print("Done!")


if __name__ == "__main__":
    main()
