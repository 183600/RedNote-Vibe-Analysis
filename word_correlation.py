import json
import re
from collections import defaultdict, Counter
from scipy import stats
import numpy as np


def clean_text(text):
    text = str(text)
    text = re.sub(r"#\w+\[话题\]", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"[0-9a-zA-Z]+", " ", text)
    text = re.sub(r"[^\u4e00-\u9fff]", " ", text)
    words = text.split()
    return [w for w in words if len(w) >= 2]


note_by_user = defaultdict(list)
word_doc_freq = Counter()

print("Loading...")
with open(
    "/home/qwe12345678/RedNote-Vibe-Analysis/exploring_set_fixed.jsonl",
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

matrix = np.zeros((n_users, n_words))
for i, uid in enumerate(user_ids):
    for word in word_to_idx:
        count = sum(
            active_users[uid][j].get(word, 0) for j in range(len(active_users[uid]))
        )
        matrix[i, word_to_idx[word]] = count

valid = [j for j in range(n_words) if np.std(matrix[:, j]) > 0]
print(f"Valid: {len(valid)}")

print("Computing all...")
all_results = []
for j1i in range(len(valid)):
    for j2i in range(j1i + 1, len(valid)):
        j1, j2 = valid[j1i], valid[j2i]
        r, p = stats.pearsonr(matrix[:, j1], matrix[:, j2])
        if not np.isnan(r):
            all_results.append((all_words[j1], all_words[j2], r, p))

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
    "/home/qwe12345678/RedNote-Vibe-Analysis/word_correlation_results.md", "w"
) as f:
    f.write("# Word Correlation by User\n\n")
    f.write("## Positive (r > 0.2)\n\n")
    for w1, w2, r in pos[:25]:
        f.write(f"- {w1} - {w2}: {r:.3f}\n")
    f.write("\n## Negative (r < 0)\n\n")
    for w1, w2, r in neg[:25]:
        f.write(f"- {w1} - {w2}: {r:.3f}\n")

print("\nDone!")
