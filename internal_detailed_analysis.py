# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
深入分析：Human vs AIGC vs Exploring 内部差异和相关性
避免与已有分析重复，增加新的分析维度
"""

import json
import re
from collections import Counter, defaultdict
from functools import partial

from analysis_utils import engagement_of, run_parallel

RE_CJK = re.compile(r"[\u4e00-\u9fff]+")
RE_SENT_SPLIT = re.compile(r"[。！？\n]")
RE_LIST_ITEM = re.compile(r"^\d+[.、]", re.MULTILINE)
RE_BULLET = re.compile(r"^[、•·]", re.MULTILINE)
# 与 paper_features.HASHTAG 同口径：拉丁/数字标签同样计入（原正则只匹配
# 汉字，含英文的 #OOTD 等会被漏计），长度为 1-30 个非空白非 # 字符。
RE_HASHTAG = re.compile(r"#[^#\s]{1,30}")
RE_EMOJI = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0"
    r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F]"
)
RE_DIGIT = re.compile(r"\d")

_DATA_KEYS = (
    "note_title",
    "domain",
    "local_time",
    "user_id",
    "time",
    "liked_count",
    "collected_count",
    "comments_count",
    "likes",
    "collections",
    "comments",
)


def iter_records(path):
    """流式读取并只保留下游用到的字段，避免把整份数据集常驻内存。

    口径与原 load_jsonl 完全一致：跳过空行与解析失败的行；正文统一放在
    note_content（若源记录有）或 desc 字段。
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = {}
            if "note_content" in item:
                rec["note_content"] = item["note_content"]
            else:
                rec["desc"] = item.get("desc", "")
            for key in _DATA_KEYS:
                if key in item:
                    rec[key] = item[key]
            yield rec


def extract_chinese(text):
    if not text:
        return []
    return RE_CJK.findall(text)


def get_text(item):
    """Get text field from different data formats"""
    if "note_content" in item:
        return item.get("note_content", "")
    return item.get("desc", "")


positive_words = [
    "喜欢",
    "爱",
    "开心",
    "快乐",
    "幸福",
    "美好",
    "推荐",
    "赞",
    "棒",
    "太好了",
    "好看",
    "满意",
    "惊喜",
    "感动",
    "治愈",
    "温暖",
    "完美",
    "感谢",
    "加油",
    "优秀",
    "成功",
    "进步",
    "成长",
]
negative_words = [
    "焦虑",
    "难过",
    "痛苦",
    "累",
    "烦",
    "失望",
    "伤",
    "哭",
    "难",
    "怕",
    "担心",
    "恐惧",
    "抑郁",
    "崩溃",
    "糟糕",
    "无力",
    "压抑",
    "自责",
    "内耗",
    "后悔",
]

_POSITIVE_SET = set(positive_words)
_NEGATIVE_SET = set(negative_words)
# 最长优先的合并模式：避免 "伤心" 同时命中 "伤"、"难过" 同时命中 "难"
# 这类嵌套词导致的同一处情感被重复计数。
_SENTIMENT_PATTERN = re.compile("|".join(
    sorted((re.escape(w) for w in _POSITIVE_SET | _NEGATIVE_SET),
           key=len, reverse=True)))


def sentiment_hits(text):
    """非重叠、最长优先的词典匹配，返回 (正向命中数, 负向命中数)。

    每个词至多计一次（保持原实现的「存在性」口径），但嵌套的短词不再与
    其所属的长词在同一位置被重复计入。
    """
    if not text:
        return 0, 0
    hits = set(_SENTIMENT_PATTERN.findall(text))
    return len(hits & _POSITIVE_SET), len(hits & _NEGATIVE_SET)


def scan(path):
    """单次流式扫描数据集，一次性累计 13 个分析维度所需的全部中间量。

    每个维度都沿用原先独立遍历时的同一套累加口径，因此各输出数值不变，
    但解析与遍历只做一遍，且不再把全量记录留在内存中。
    """
    n = 0
    domains = Counter()
    # 情感（正文 / 标题）
    pos_c = neg_c = pos_t = neg_t = 0
    # 互动
    likes_sum = collects_sum = comments_sum = 0
    has_engagement = False
    # 结构（正文 / 标题）
    para_c = list_c = bullet_c = emoji_c = hashtag_c = 0
    para_t = list_t = bullet_t = emoji_t = hashtag_t = 0
    # 复杂度（正文 / 标题）
    chars_sum_c = chars_cnt_c = ratio_sum_c = sent_sum_c = sent_cnt_c = 0
    chars_sum_t = chars_cnt_t = ratio_sum_t = sent_sum_t = sent_cnt_t = 0
    # 标题风格
    title_q = title_ex = title_num = title_emo = title_empty = 0
    # 词表多样性（汉字串集合）
    ttr_total = 0
    seen_runs = set()
    # 元数据
    meta_user = meta_time = meta_lt = meta_like = meta_collect = meta_comment = 0
    # 领域-情感（普通 dict，便于结果 pickle 后由并行 worker 返回）
    domain_sent = {}
    # 一致性
    empty_desc = empty_title = short = 0
    titles_c = Counter()
    # 时间分布
    months = Counter()
    # 标点（正文 / 标题）
    punct_c = {"!": 0, "?": 0, "...": 0}
    punct_t = {"!": 0, "?": 0, "...": 0}
    # 领域-长度（保留首次出现顺序）
    domain_len_sum = defaultdict(int)
    domain_len_cnt = defaultdict(int)

    for item in iter_records(path):
        n += 1
        title = item.get("note_title", "")
        content = get_text(item)

        # 1. 领域分布
        domains[item.get("domain", "Unknown")] += 1

        # 2. 情感（正文 / 标题）
        pos_i, neg_i = sentiment_hits(content)
        pos_c += pos_i
        neg_c += neg_i
        pos_ti, neg_ti = sentiment_hits(title)
        pos_t += pos_ti
        neg_t += neg_ti

        # 3. 互动（AIGC 语料没有互动字段：缺失不等于 0，单独标记，避免
        #    把 0 当成真实互动参与率打印/对比）
        if any(k in item for k in ("liked_count", "likes", "collected_count",
                                   "collections", "comments_count", "comments")):
            has_engagement = True
        like_i, coll_i, comm_i = engagement_of(item)
        likes_sum += like_i
        collects_sum += coll_i
        comments_sum += comm_i

        # 4. 结构（正文）
        para_c += sum(1 for p in content.split("\n") if p.strip())
        list_c += len(RE_LIST_ITEM.findall(content))
        bullet_c += len(RE_BULLET.findall(content))
        emoji_c += len(RE_EMOJI.findall(content))
        hashtag_c += len(RE_HASHTAG.findall(content))
        # 4b. 结构（标题）
        para_t += sum(1 for p in title.split("\n") if p.strip())
        list_t += len(RE_LIST_ITEM.findall(title))
        bullet_t += len(RE_BULLET.findall(title))
        emoji_t += len(RE_EMOJI.findall(title))
        hashtag_t += len(RE_HASHTAG.findall(title))

        # 5. 复杂度（正文）
        words = extract_chinese(content)
        if words:
            total = sum(map(len, words))
            unique = len(set("".join(words)))
            chars_sum_c += total
            chars_cnt_c += 1
            ratio_sum_c += unique / total if total > 0 else 0
        for s in RE_SENT_SPLIT.split(content):
            s = s.strip()
            if s:
                sent_sum_c += len(s)
                sent_cnt_c += 1
        # 5b. 复杂度（标题）
        chars = extract_chinese(title)
        if chars:
            total = sum(map(len, chars))
            unique = len(set("".join(chars)))
            chars_sum_t += total
            chars_cnt_t += 1
            ratio_sum_t += unique / total if total > 0 else 0
        for s in RE_SENT_SPLIT.split(title):
            s = s.strip()
            if s:
                sent_sum_t += len(s)
                sent_cnt_t += 1

        # 6. 标题风格
        if not title:
            title_empty += 1
        else:
            if "?" in title or "？" in title:
                title_q += 1
            if "!" in title or "！" in title:
                title_ex += 1
            if RE_DIGIT.search(title):
                title_num += 1
            if RE_EMOJI.search(title):
                title_emo += 1

        # 7. 词表多样性（汉字串，复用第 5 步的正文汉字串）
        ttr_total += len(words)
        seen_runs.update(words)

        # 8. 元数据
        if "user_id" in item:
            meta_user += 1
        if "time" in item:
            meta_time += 1
        if "local_time" in item:
            meta_lt += 1
        if "liked_count" in item:
            meta_like += 1
        if "collected_count" in item:
            meta_collect += 1
        if "comments_count" in item:
            meta_comment += 1

        # 9. 领域-情感（复用第 2 步已算好的本条正文情感计数）
        key = item.get("domain", "Unknown")
        slot = domain_sent.get(key)
        if slot is None:
            slot = domain_sent[key] = [0, 0]
        slot[0] += pos_i
        slot[1] += neg_i

        # 10. 一致性
        stripped = content.strip()
        if not stripped:
            empty_desc += 1
        if not title:
            empty_title += 1
        if content and len(stripped) < 20:
            short += 1
        # 空标题不是「重复标题」：不计入重复标题统计（否则人类/探索集会
        # 因大量缺失标题而凭空多出一个重复类别）。
        if title:
            titles_c[title] += 1

        # 11. 时间分布
        lt = item.get("local_time", "")
        if lt and len(lt) >= 6:
            months[lt[:6]] += 1

        # 12. 标点（与 internal_diff_analysis / paper_features 同口径：
        #     省略号同时统计 ASCII "..."、Unicode "…" 与中文 "。。。"）
        punct_c["!"] += content.count("!") + content.count("！")
        punct_c["?"] += content.count("?") + content.count("？")
        punct_c["..."] += (content.count("...") + content.count("…")
                           + content.count("。。。"))
        punct_t["!"] += title.count("!") + title.count("！")
        punct_t["?"] += title.count("?") + title.count("？")
        punct_t["..."] += (title.count("...") + title.count("…")
                           + title.count("。。。"))

        # 13. 领域-长度
        d = item.get("domain", "Unknown")
        domain_len_sum[d] += len(content)
        domain_len_cnt[d] += 1

    return {
        "n": n,
        "domains": domains,
        "sent": (pos_c, neg_c, pos_t, neg_t),
        "engagement": (likes_sum, collects_sum, comments_sum),
        "has_engagement": has_engagement,
        "structure_content": (para_c, list_c, bullet_c, emoji_c, hashtag_c),
        "structure_title": (para_t, list_t, bullet_t, emoji_t, hashtag_t),
        "complexity_content": (chars_sum_c, chars_cnt_c, ratio_sum_c,
                               sent_sum_c, sent_cnt_c),
        "complexity_title": (chars_sum_t, chars_cnt_t, ratio_sum_t,
                             sent_sum_t, sent_cnt_t),
        "title_style": (title_q, title_ex, title_num, title_emo, title_empty),
        "ttr": (ttr_total, len(seen_runs)),
        "metadata": (meta_user, meta_time, meta_lt, meta_like, meta_collect,
                     meta_comment),
        "domain_sent": domain_sent,
        "consistency": (empty_desc, empty_title, short,
                        sum(1 for c in titles_c.values() if c > 1)),
        "months": months,
        "punct_content": punct_c,
        "punct_title": punct_t,
        "domain_len_sum": domain_len_sum,
        "domain_len_cnt": domain_len_cnt,
    }


def analyze_domain(result, name):
    domains = result["domains"]
    total = sum(domains.values())
    print(f"\n{name}:")
    for domain, count in domains.most_common(12):
        print(f"  {domain}: {count} ({count / total * 100:.1f}%)")


def sentiment(result, name, use_title=False):
    pos_c, neg_c, pos_t, neg_t = result["sent"]
    pos, neg = (pos_t, neg_t) if use_title else (pos_c, neg_c)
    label = "Title" if use_title else "Content"
    ratio = pos / neg if neg > 0 else float("inf")
    print(
        f"\n{name} ({label}): Positive={pos}, Negative={neg}, Ratio={ratio:.2f}"
    )


def engagement(result, name):
    if not result.get("has_engagement", True):
        print(f"\n{name}: N/A (no engagement fields)")
        return
    likes_sum, collects_sum, comments_sum = result["engagement"]
    n = result["n"]
    print(
        f"\n{name}: Likes={likes_sum / n:.1f}, Collects={collects_sum / n:.1f}, Comments={comments_sum / n:.1f}"
    )


def structure(result, name, use_title=False):
    if use_title:
        para, lists, bullet, emoji, hashtag = result["structure_title"]
    else:
        para, lists, bullet, emoji, hashtag = result["structure_content"]
    n = result["n"]
    label = "Title" if use_title else "Content"
    print(
        f"\n{name} ({label}): Para={para / n:.1f}, List={lists / n:.1f}, Emoji={emoji / n:.1f}, Tag={hashtag / n:.1f}"
    )


def complexity(result, name, use_title=False):
    if use_title:
        chars_sum, chars_cnt, ratio_sum, sent_sum, sent_cnt = \
            result["complexity_title"]
    else:
        chars_sum, chars_cnt, ratio_sum, sent_sum, sent_cnt = \
            result["complexity_content"]

    avg_chars = chars_sum / chars_cnt if chars_cnt else 0
    avg_ratio = ratio_sum / chars_cnt if chars_cnt else 0
    avg_sent = sent_sum / sent_cnt if sent_cnt else 0
    label = "Title" if use_title else "Content"
    print(
        f"\n{name} ({label}): AvgChars={avg_chars:.0f}, UniqueRatio={avg_ratio:.3f}, AvgSentLen={avg_sent:.0f}"
    )


def title_style(result, name):
    q, ex, num, emo, empty = result["title_style"]
    n = result["n"]
    print(
        f"\n{name}: Question={q / n * 100:.1f}%, Exclaim={ex / n * 100:.1f}%, Number={num / n * 100:.1f}%, Emoji={emo / n * 100:.1f}%, Empty={empty / n * 100:.1f}%"
    )


def vocab_ttr(result, name):
    total, unique = result["ttr"]
    ttr = unique / total if total > 0 else 0
    print(f"\n{name}: Total={total}, Unique={unique}, TTR={ttr:.4f}")


def metadata(result, name):
    (user, tm, lt, like, collect, comment) = result["metadata"]
    n = result["n"]
    print(
        f"\n{name}: user_id={user / n * 100:.1f}%, time={tm / n * 100:.1f}%, local_time={lt / n * 100:.1f}%, likes={like / n * 100:.1f}%"
    )


def domain_style(result, name):
    print(f"\n{name}:")
    top_domains = sorted(
        result["domain_sent"].items(),
        key=lambda kv: kv[1][0] + kv[1][1],
        reverse=True,
    )[:5]
    for d, (p, n) in top_domains:
        ratio = p / n if n > 0 else float("inf")
        print(f"  {d}: pos={p}, neg={n}, ratio={ratio:.2f}")


def consistency(result, name):
    empty_desc, empty_title, short, dup = result["consistency"]
    n = result["n"]
    print(
        f"\n{name}: EmptyDesc={empty_desc / n * 100:.1f}%, EmptyTitle={empty_title / n * 100:.1f}%, Short<20={short / n * 100:.1f}%, DuplicateTitle={dup}"
    )


def time_dist(result, name):
    print(f"\n{name}:")
    for m, c in result["months"].most_common(8):
        print(f"  {m}: {c}")


def char_patterns(result, name, use_title=False):
    punct = result["punct_title"] if use_title else result["punct_content"]
    n = result["n"]
    label = "Title" if use_title else "Content"
    print(
        f"\n{name} ({label}): Exclaim={punct['!'] / n:.2f}, Question={punct['?'] / n:.2f}, Ellipsis={punct['...'] / n:.2f}"
    )


def domain_len(result, name):
    domain_lens = {
        d: (result["domain_len_sum"][d], result["domain_len_cnt"][d])
        for d in result["domain_len_sum"]
    }
    print(f"\n{name}:")
    for d, (total, cnt) in sorted(
        domain_lens.items(), key=lambda x: x[1][0] / x[1][1], reverse=True
    )[:5]:
        avg = total / cnt
        print(f"  {d}: avg_len={avg:.0f}")


def main():
    print("Loading data...")
    # 三个数据集互不依赖，进程池并行扫描（池不可用时 run_parallel 自动串行，
    # 结果与串行逐位一致）。
    datasets = [
        ("Human", "training_set_human.jsonl"),
        ("AIGC", "training_set_aigc.jsonl"),
        ("Exploring", "exploring_set.jsonl"),
    ]
    results = run_parallel([partial(scan, path) for _, path in datasets])
    stats = {name: result for (name, _), result in zip(datasets, results)}

    print(
        f"Human: {stats['Human']['n']}, AIGC: {stats['AIGC']['n']}, Exploring: {stats['Exploring']['n']}"
    )

    print("\n=== 1. Internal Domain Distribution ===")
    for name, result in stats.items():
        analyze_domain(result, name)

    print("\n=== 2. Sentiment Analysis ===")
    print("\n=== 2. Sentiment Analysis (Content) ===")
    for name, result in stats.items():
        sentiment(result, name, False)

    print("\n=== 2b. Sentiment Analysis (Title) ===")
    for name, result in stats.items():
        sentiment(result, name, True)

    print("\n=== 3. Engagement Stats ===")
    for name, result in stats.items():
        engagement(result, name)

    print("\n=== 4. Content Structure ===")
    print("\n=== 4. Content Structure (Content) ===")
    for name, result in stats.items():
        structure(result, name, False)

    print("\n=== 4b. Content Structure (Title) ===")
    for name, result in stats.items():
        structure(result, name, True)

    print("\n=== 5. Text Complexity ===")
    print("\n=== 5. Text Complexity (Content) ===")
    for name, result in stats.items():
        complexity(result, name, False)

    print("\n=== 5b. Text Complexity (Title) ===")
    for name, result in stats.items():
        complexity(result, name, True)

    print("\n=== 6. Title Style ===")
    for name, result in stats.items():
        title_style(result, name)

    print("\n=== 7. Vocabulary Diversity (TTR) ===")
    for name, result in stats.items():
        vocab_ttr(result, name)

    print("\n=== 8. Metadata Fields ===")
    for name, result in stats.items():
        metadata(result, name)

    print("\n=== 9. Domain-Style Correlation ===")
    for name, result in stats.items():
        domain_style(result, name)

    print("\n=== 10. Data Consistency ===")
    for name, result in stats.items():
        consistency(result, name)

    print("\n=== 11. Time Distribution ===")
    for name, result in stats.items():
        time_dist(result, name)

    print("\n=== 12. Character Patterns ===")
    print("\n=== 12. Character Patterns (Content) ===")
    for name, result in stats.items():
        char_patterns(result, name, False)

    print("\n=== 12b. Character Patterns (Title) ===")
    for name, result in stats.items():
        char_patterns(result, name, True)

    print("\n=== 13. Domain vs Content Length ===")
    for name, result in stats.items():
        domain_len(result, name)

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
