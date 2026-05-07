# -*- coding: utf-8 -*-
import json
import re
import math
from collections import defaultdict, Counter


def mean(lst):
    return sum(lst) / len(lst) if lst else 0


def pearsonr(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    mx = mean(x)
    my = mean(y)
    cov = sum((x[i] - mx) * (y[i] - my) for i in xrange(n))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


exclude = set([u"话题", u"时尚薯", u"生活薯", u"知识薯", u"美食薯", u"校园薯",
               u"薯队长", u"薯条小助手", u"小红书创作学院"])

print("Loading...")
note_by_user = defaultdict(lambda: defaultdict(int))
word_doc_freq = Counter()

with open("exploring_set_fixed.jsonl", "rb") as f:
    for line in f:
        try:
            s = line.decode("utf-8").strip()
            item = json.loads(s)
            user_id = item.get("user_id")
            if not user_id:
                continue
            text = item.get("note_title", "") + " " + item.get("desc", "")
            words = re.findall(ur"[\u4e00-\u9fff]+", text)
            for w in set(words):
                if len(w) >= 2 and w not in exclude:
                    note_by_user[user_id][w] += 1
                    word_doc_freq[w] += 1
        except:
            pass

print("Users: " + str(len(note_by_user)))

MIN_COUNT = 20
active_users = {u: c for u, c in note_by_user.items() if sum(c.values()) >= MIN_COUNT}
print("Active: " + str(len(active_users)))

MIN_USERS = 30
top_words = [w for w, c in word_doc_freq.most_common(100) if c >= MIN_USERS]
print("Words: " + str(len(top_words)))

word_to_idx = {w: i for i, w in enumerate(top_words)}
user_ids = list(active_users.keys())[:3000]
n_users = len(user_ids)
n_words = len(top_words)

print("Matrix: " + str(n_users) + " x " + str(n_words))

matrix = []
for uid in user_ids:
    total = float(sum(active_users[uid].values()))
    row = [active_users[uid].get(w, 0) / total if total > 0 else 0 for w in top_words]
    matrix.append(row)

def get_col(col_idx):
    return [matrix[row][col_idx] for row in xrange(n_users)]

def std(lst):
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))

valid = [j for j in xrange(n_words) if std(get_col(j)) > 0]
print("Valid: " + str(len(valid)))

if len(valid) < 2:
    print("Not enough!")
    exit(1)

print("Computing...")
all_results = []
for j1i in xrange(len(valid)):
    for j2i in xrange(j1i + 1, len(valid)):
        j1, j2 = valid[j1i], valid[j2i]
        r = pearsonr(get_col(j1), get_col(j2))
        if not math.isnan(r):
            all_results.append((top_words[j1], top_words[j2], r))

all_results.sort(key=lambda x: x[2])

pos = [(w1, w2, r) for w1, w2, r in all_results if r > 0.15]
neg = [(w1, w2, r) for w1, w2, r in all_results if r < -0.05]

print("Pos: " + str(len(pos)) + ", Neg: " + str(len(neg)))

with open("word_correlation_extended_results.md", "wb") as f:
    f.write((u"# 跨话题文风词词频相关性分析\n\n").encode('utf-8'))
    f.write((u"分析对象：在多个话题（domain）中都出现的词\n").encode('utf-8'))
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