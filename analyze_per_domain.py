# -*- coding: utf-8 -*-
import json
import re
from collections import defaultdict, Counter

exclude = set([u'话题', u'时尚薯', u'生活薯', u'知识薯', u'美食薯', u'校园薯',
               u'薯队长', u'薯条小助手', u'小红书创作学院', u'笔记灵感'])

# 36个跨话题词
cross_topic_words = [
    u'分钟', u'小时', u'女性成长', u'哭惹', u'笑哭', u'偷笑', u'失望', u'所以',
    u'害羞', u'左右', u'派对', u'捂脸', u'比如', u'姐妹们', u'周末去哪儿', 
    u'浪漫生活的记录者', u'萌萌哒', u'我的日常', u'日常', u'大大方方做自己', 
    u'最后', u'记录吧就现在', u'运动', u'汗颜', u'种草', u'飞吻', u'记住',
    u'大学生', u'总结', u'职场', u'穿搭', u'上海', u'但是', u'宝子们', u'个月'
]

# 主要话题
major_domains = [u'\u60c5\u611f', u'\u7a7f\u642d', u'\u5fc3\u7406', u'\u5065\u5eb7', u'\u804c\u573a', u'\u65c5\u884c']
domain_names = {u'\u60c5\u611f': '情感', u'\u7a7f\u642d': '穿搭', u'\u5fc3\u7406': '心理',
                u'\u5065\u5eb7': '健康', u'\u804c\u573a': '职场', u'\u65c5\u884c': '旅行'}

print("Loading...")
# 每个话题的词频
domain_word_freq = {d: Counter() for d in major_domains}
domain_total = {d: 0 for d in major_domains}

with open("exploring_set_fixed.jsonl", "rb") as f:
    for line in f:
        try:
            s = line.decode("utf-8").strip()
            item = json.loads(s)
            domain = item.get("domain", "unknown")
            if domain not in major_domains:
                continue
            
            text = item.get("note_title", "") + " " + item.get("desc", "")
            words = re.findall(ur"[\u4e00-\u9fff]+", text)
            
            domain_total[domain] += 1
            for w in words:
                if w in cross_topic_words:
                    domain_word_freq[domain][w] += 1
        except:
            pass

print("Domain totals:")
for d in major_domains:
    print("  " + domain_names[d] + ": " + str(domain_total[d]) + " notes")

print("\n=== 每个话题Top10词 ===")
for d in major_domains:
    print("\n" + domain_names[d] + ":")
    for w, c in domain_word_freq[d].most_common(10):
        pct = c * 100.0 / domain_total[d]
        print("  " + w.encode('utf-8') + ": " + str(c) + " (" + str(round(pct, 2)) + "%)")

# 计算每个词在每个话题的相对频率
print("\n=== 各话题词频占比矩阵 ===")
results = []
for w in cross_topic_words:
    row = []
    for d in major_domains:
        freq = domain_word_freq[d].get(w, 0)
        pct = freq * 100.0 / domain_total[d] if domain_total[d] > 0 else 0
        row.append(pct)
    avg_pct = sum(row) / len(row)
    min_pct = min(row)
    results.append((w, row, avg_pct, min_pct))

# 按平均词频排序
results.sort(key=lambda x: x[2], reverse=True)

print("\n词频占比(%) | 情感 | 穿搭 | 心理 | 健康 | 职场 | 旅行 | 平均")
print("-" * 70)
for w, row, avg, min_pct in results[:25]:
    print(w.encode('utf-8').ljust(8) + " | " + " | ".join([str(round(r, 2)).rjust(5) for r in row]) + " | " + str(round(avg, 2)))

# 找出在所有话题词频都较高的词
print("\n=== 在所有话题都大量存在的词 ===")
high_freq_words = []
for w, row, avg, min_pct in results:
    if min_pct > 0.05:  # 在每个话题词频都超过0.05%
        high_freq_words.append((w, row, min_pct))

high_freq_words.sort(key=lambda x: x[2], reverse=True)
print("词频都超过0.05%的词: " + str(len(high_freq_words)))
for w, row, min_pct in high_freq_words:
    print("  " + w.encode('utf-8') + ": min=" + str(round(min_pct, 2)) + "%")

# 保存结果
with open("word_correlation_extended_results.md", "wb") as f:
    f.write((u"# 各话题中的跨话题文风词分析\n\n").encode('utf-8'))
    f.write((u"## 36个跨话题词在6个主要话题的词频分布\n\n").encode('utf-8'))
    f.write((u"词频占比(%)\n\n").encode('utf-8'))
    
    header = u"词语 | 情感 | 穿搭 | 心理 | 健康 | 职场 | 旅行 | 平均"
    f.write((header + u"\n").encode('utf-8'))
    f.write((u"-" * len(header) + u"\n").encode('utf-8'))
    
    for w, row, avg, min_pct in results[:30]:
        line = w + u" | " + u" | ".join([str(round(r, 1)) for r in row]) + u" | " + str(round(avg, 1))
        f.write((line + u"\n").encode('utf-8'))
    
    f.write((u"\n## 在所有话题都大量存在的词\n\n").encode('utf-8'))
    f.write((u"以下词在6个主要话题的词频都超过0.05%:\n\n").encode('utf-8'))
    for w, row, min_pct in high_freq_words:
        f.write((u"- " + w + u": 最低 " + str(round(min_pct, 2)) + u"%\n").encode('utf-8'))

print("\nDone!")