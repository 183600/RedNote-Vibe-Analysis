import json
import re
from collections import defaultdict, Counter
import numpy as np

from analysis_utils import BASE


RE_TOPIC_TAG = re.compile(r"#\w+\[话题\]")
RE_HASHTAG = re.compile(r"#\w+")
RE_ASTRAL = re.compile(r"[\U00010000-\U0010ffff]")
RE_ALNUM = re.compile(r"[0-9a-zA-Z]+")
RE_NON_CJK = re.compile(r"[^\u4e00-\u9fff]")


def clean_text(text):
    text = str(text)
    text = RE_TOPIC_TAG.sub(" ", text)
    text = RE_HASHTAG.sub(" ", text)
    text = RE_ASTRAL.sub(" ", text)
    text = RE_ALNUM.sub(" ", text)
    text = RE_NON_CJK.sub(" ", text)
    words = text.split()
    return [w for w in words if len(w) >= 2]


def main():
    note_by_user = defaultdict(list)
    word_doc_freq = Counter()

    print("Loading...")
    with open(
        BASE + "exploring_set_fixed.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                user_id = item.get("user_id")
                if not user_id:
                    continue
                text = item.get("note_title", "") + " " + item.get("desc", "")
                words = clean_text(text)
                note_by_user[user_id].append(Counter(words))
                for w in set(words):
                    word_doc_freq[w] += 1
            except:
                continue

    MIN_NOTES = 2
    active_users = {u: n for u, n in note_by_user.items() if len(n) >= MIN_NOTES}

    MIN_USERS = 3
    topWords = [w for w, c in word_doc_freq.most_common(200) if c >= MIN_USERS]

    all_words = topWords
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    user_ids = list(active_users.keys())
    n_users = len(user_ids)
    n_words = len(all_words)

    print(f"{n_users} users, {n_words} words")

    user_totals = {}
    for uid in user_ids:
        total = Counter()
        for c in active_users[uid]:
            total.update(c)
        user_totals[uid] = total

    matrix = np.zeros((n_users, n_words))
    for i, uid in enumerate(user_ids):
        uc = user_totals[uid]
        row = matrix[i]
        for j, word in enumerate(all_words):
            row[j] = uc.get(word, 0)

    valid = [j for j in range(n_words) if np.std(matrix[:, j]) > 0]
    print(f"Valid: {len(valid)}")

    cols = np.ascontiguousarray(matrix[:, valid].T)
    zs = np.empty_like(cols)
    for i in range(cols.shape[0]):
        xm = cols[i] - cols[i].mean()
        xmax = np.max(np.abs(xm))
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = xmax * np.linalg.vector_norm(xm / xmax, axis=-1, keepdims=True)
            zs[i] = xm / norm

    print("Computing all...")
    all_results = []
    for j1i in range(len(valid)):
        for j2i in range(j1i + 1, len(valid)):
            j1, j2 = valid[j1i], valid[j2i]
            r = np.clip(np.vecdot(zs[j1i], zs[j2i]), -1.0, 1.0)
            if not np.isnan(r):
                all_results.append((all_words[j1], all_words[j2], r, None))

    all_results.sort(key=lambda x: x[2])

    print(f"Total: {len(all_results)}")

    pos = [(w1, w2, r) for w1, w2, r, p in all_results if r > 0.2]
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
        f.write("## Positive (r > 0.2)\n\n")
        for w1, w2, r in pos[:25]:
            f.write(f"- {w1} - {w2}: {r:.3f}\n")
        f.write("\n## Negative (r < 0)\n\n")
        for w1, w2, r in neg[:25]:
            f.write(f"- {w1} - {w2}: {r:.3f}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
