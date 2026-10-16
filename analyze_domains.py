# -*- coding: utf-8 -*-
import json
import re
from collections import Counter

from analysis_utils import BASE


exclude = set([u'话题', u'时尚薯', u'生活薯', u'知识薯', u'美食薯', u'校园薯',
               u'薯队长', u'薯条小助手', u'小红书创作学院', u'笔记灵感', u'无标题'])

# 单个词的全部状态打包进一个 int：高位是出现该词的笔记数，低 _MASK_BITS 位是
# 话题 bitmask。避免为每个词同时维护 Counter 计数 + 一个 set 对象（原先每个词
# 一个 set，数百万个 set 对象是内存占用的主要来源）。
_COUNT_SHIFT = 16
_COUNT_ONE = 1 << _COUNT_SHIFT
_MASK_BITS = _COUNT_ONE - 1


def _bump(info, domain_bit):
    """词频计数 +1 并并入话题 bit（info=None 表示首次出现）。"""
    if info is None:
        return _COUNT_ONE | domain_bit
    return (info + _COUNT_ONE) | domain_bit


def main():
    print('Loading...')
    word_info = {}
    domain_counts = Counter()
    domain_bits = {}

    with open(BASE + 'exploring_set_fixed.jsonl', 'rb') as f:
        for line in f:
            try:
                item = json.loads(line)
                domain = item.get('domain', 'unknown')
                domain_counts[domain] += 1
                bit = domain_bits.setdefault(domain, 1 << len(domain_bits))
                text = item.get('note_title', '') + ' ' + (
                    item.get('desc', '') or item.get('note_content', ''))
                words = re.findall(r'[\u4e00-\u9fff]+', text)
                for w in set(words):
                    if len(w) >= 2 and w not in exclude:
                        word_info[w] = _bump(word_info.get(w), bit)
            except:
                pass

    domain_display = {
        u'\u60c5\u611f': '情感', u'\u7a7f\u642d': '穿搭', u'\u5fc3\u7406': '心理',
        u'\u5065\u5eb7': '健康', u'\u804c\u573a': '职场', u'\u65c5\u884c': '旅行',
        u'\u672a\u77e5': '未知', u'\u5ba0\u7269': '宠物', u'\u7f8e\u98df': '美食',
        u'\u5b66\u4e60': '学习', u'\u8fd0\u52a8': '运动'
    }

    MIN_USERS = 30
    # 等价于原先 Counter.most_common(100)：按计数降序，并列时保持首次出现顺序
    # （info 低 16 位是话题 bitmask，排序时必须右移，否则并列会按 bitmask 值排序）。
    ranked = sorted(word_info.items(),
                    key=lambda kv: kv[1] >> _COUNT_SHIFT, reverse=True)

    results = []
    for w, info in ranked[:100]:
        count = info >> _COUNT_SHIFT
        if count < MIN_USERS:
            continue
        mask = info & _MASK_BITS
        results.append((w, bin(mask).count("1"), mask, count))

    results.sort(key=lambda x: x[1], reverse=True)

    # Count distribution
    dist = Counter()
    for w, n, m, c in results:
        dist[n] += 1

    with open("domain_coverage_results.md", "wb") as f:
        f.write((u"# 100个跨话题常用词的话题分布分析\n\n").encode('utf-8'))
    
        f.write((u"## 实际话题分布（按笔记数排序）\n\n").encode('utf-8'))
        for d, c in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            name = domain_display.get(d, d)
            f.write((u"- " + name + u": " + str(c) + u" 笔记\n").encode('utf-8'))
    
        f.write((u"\n## 100个词的话题覆盖分布\n\n").encode('utf-8'))
        for n in sorted(dist.keys(), reverse=True):
            f.write((u"- " + str(n) + u"个话题: " + str(dist[n]) + u"个词\n").encode('utf-8'))
    
        f.write((u"\n## 真正的跨话题词（在全部6个主要话题出现）\n\n").encode('utf-8'))
        major_domains = [u'\u60c5\u611f', u'\u7a7f\u642d', u'\u5fc3\u7406', u'\u5065\u5eb7', u'\u804c\u573a', u'\u65c5\u884c']
        in_all_6 = [(w, c) for w, n, m, c in results
                    if all(m & domain_bits.get(d, 0) for d in major_domains)]
        f.write((u"共 " + str(len(in_all_6)) + u" 个词在6个主要话题都有出现:\n\n").encode('utf-8'))
        for w, c in in_all_6:
            f.write((u"- " + w + u": " + str(c) + u" 笔记\n").encode('utf-8'))
    
        f.write((u"\n## 只在少量话题出现的词\n\n").encode('utf-8'))
        f.write((u"以下词只在1-3个话题出现，不能算真正的\"跨话题\"词:\n\n").encode('utf-8'))
        few_topics = [(w, n, c) for w, n, m, c in results if n <= 3]
        for w, n, c in few_topics[:15]:
            f.write((u"- " + w + u": " + str(n) + u"个话题, " + str(c) + u"笔记\n").encode('utf-8'))
    
        f.write((u"\n## 结论\n\n").encode('utf-8'))
        n_all6 = len(in_all_6)
        n_dom = len(domain_counts)
        n_all = dist.get(n_dom, 0)
        n_mode = max(dist, key=lambda x: dist[x]) if dist else 0
        n_mode_cnt = dist.get(n_mode, 0)
        n_5 = dist.get(5, 0)
        n_few = sum(c for n, c in dist.items() if n <= 4)
        f.write((u"1. 100个词中有 " + str(n_all6) + u" 个词真正在6个主要话题（情感、穿搭、心理、健康、职场、旅行）都出现\n").encode('utf-8'))
        f.write((u"2. 话题覆盖数的众数是 " + str(n_mode) + u" 个话题（" + str(n_mode_cnt) + u" 个词）；覆盖全部 " + str(n_dom) + u" 个话题（含\"未知\"分类）的词有 " + str(n_all) + u" 个\n").encode('utf-8'))
        f.write((u"3. 有 " + str(n_5) + u" 个词只出现在5个话题，" + str(n_few) + u"个词出现在4个及以下话题，不能算真正的跨话题词\n").encode('utf-8'))
        f.write((u"4. \"未知\"分类包含大量笔记，是分类系统未命名的内容\n").encode('utf-8'))

    print("Done!")


if __name__ == "__main__":
    main()
