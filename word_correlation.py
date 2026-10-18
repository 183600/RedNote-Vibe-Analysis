import json
import re
from collections import defaultdict, Counter
import numpy as np

from analysis_utils import BASE


RE_TOPIC_TAG = re.compile(r"#\w+\[话题\]")
RE_HASHTAG = re.compile(r"#\w+")
# 去掉话题标签后就只剩下汉字是有效字符：所有非汉字字符（含空白/标点/字母数字）
# 一律作为分隔符切词，等价于原先依次把 astral/alnum/非汉字替换为空格再 split。
RE_NON_CJK = re.compile(r"[^\u4e00-\u9fff]+")


def clean_text(text):
    text = str(text)
    text = RE_TOPIC_TAG.sub(" ", text)
    text = RE_HASHTAG.sub(" ", text)
    return [w for w in RE_NON_CJK.split(text) if len(w) >= 2]


PATH = BASE + "exploring_set_fixed.jsonl"


def _record_words(item):
    text = item.get("note_title", "") + " " + (
        item.get("desc") or item.get("note_content") or "")
    return clean_text(text)


def main():
    note_counts = defaultdict(int)
    user_token_total = defaultdict(int)
    word_doc_freq = Counter()
    # 第一遍顺手记下每条可用记录的 (user_id, 行起始字节偏移)，第二遍只回读
    # 目标用户的少量行，不再整份重解析大文件。
    record_offsets = []

    print("Loading...")
    # 二进制模式读取：文本模式逐行 tell() 需要反复处理解码器状态，代价高；
    # 数据文件无 CR，二进制按 LF 切行与文本通用换行口径一致，json.loads
    # 可直接解析 bytes。
    with open(PATH, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                item = json.loads(line)
                user_id = item.get("user_id")
                if not user_id:
                    continue
                words = _record_words(item)
                note_counts[user_id] += 1
                user_token_total[user_id] += len(words)
                record_offsets.append((user_id, pos))
                for w in set(words):
                    word_doc_freq[w] += 1
            except:
                continue

    MIN_NOTES = 2
    user_ids = [u for u, n in note_counts.items() if n >= MIN_NOTES]

    MIN_USERS = 3
    topWords = [w for w, c in word_doc_freq.most_common(200) if c >= MIN_USERS]

    print("Counting top words...")
    top_word_set = set(topWords)
    wanted_users = set(user_ids)
    user_totals = defaultdict(Counter)
    with open(PATH, "rb") as f:
        for user_id, pos in record_offsets:
            if user_id not in wanted_users:
                continue
            try:
                f.seek(pos)
                item = json.loads(f.readline())
                user_totals[user_id].update(
                    w for w in _record_words(item) if w in top_word_set)
            except:
                continue

    all_words = topWords
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    n_users = len(user_ids)
    n_words = len(all_words)

    print(f"{n_users} users, {n_words} words")

    matrix = np.zeros((n_users, n_words))
    for i, uid in enumerate(user_ids):
        uc = user_totals[uid]
        # 按用户总词频归一化：原始计数会让"发帖多/写得长"的用户对所有词
        # 都有高计数，把活跃度混进词间相关；占比口径与
        # word_correlation_extended.py 一致。
        total = float(user_token_total[uid])
        row = matrix[i]
        for j, word in enumerate(all_words):
            row[j] = uc.get(word, 0) / total if total > 0 else 0.0

    valid = [j for j in range(n_words) if np.std(matrix[:, j]) > 0]
    print(f"Valid: {len(valid)}")

    cols = np.ascontiguousarray(matrix[:, valid].T)
    zs = np.empty_like(cols)
    for i in range(cols.shape[0]):
        xm = cols[i] - cols[i].mean()
        xmax = np.max(np.abs(xm))
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = xmax * np.linalg.norm(xm / xmax)
            zs[i] = xm / norm

    print("Computing all...")
    all_results = []
    for j1i in range(len(valid)):
        for j2i in range(j1i + 1, len(valid)):
            j1, j2 = valid[j1i], valid[j2i]
            r = np.clip(np.dot(zs[j1i], zs[j2i]), -1.0, 1.0)
            if not np.isnan(r):
                all_results.append((all_words[j1], all_words[j2], r, None))

    all_results.sort(key=lambda x: x[2])

    print(f"Total: {len(all_results)}")

    pos = [(w1, w2, r) for w1, w2, r, p in all_results if r > 0.2]
    pos.sort(key=lambda x: x[2], reverse=True)
    neg = [(w1, w2, r) for w1, w2, r, p in all_results if r < 0]

    print(f"\nPositive (>0.2): {len(pos)}")
    print(f"Negative (<0): {len(neg)}")

    print("\n=== Top Positive ===")
    for w1, w2, r in pos[:20]:
        print(f"{w1} - {w2}: {r:.3f}")

    print("\n=== Top Negative ===")
    for w1, w2, r in neg[:20]:
        print(f"{w1} - {w2}: {r:.3f}")

    with open(
        "word_correlation_results.md", "w"
    ) as f:
        f.write("# Word Correlation by User\n\n")
        f.write("用户词频按本人总词频归一化（占比）后计算 Pearson r，"
                "避免发帖量/篇幅差异的混杂。\n\n")
        f.write("## Positive (r > 0.2)\n\n")
        for w1, w2, r in pos[:25]:
            f.write(f"- {w1} - {w2}: {r:.3f}\n")
        f.write("\n## Negative (r < 0)\n\n")
        for w1, w2, r in neg[:25]:
            f.write(f"- {w1} - {w2}: {r:.3f}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
