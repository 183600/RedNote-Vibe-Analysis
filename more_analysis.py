# -*- coding: utf-8 -*-
import heapq
import json
import random
import re
import codecs
from collections import Counter

RE_PUNCT = re.compile(r"[^\w\s]")
RE_DIGITS = re.compile(r"\d+")

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\U00002702-\U000027b0\U0001f190-\U0001f19a"
    "]+"
)

STOPWORDS = set(
    [
        "de",
        "shi",
        "zai",
        "le",
        "he",
        "yu",
        "huo",
        "ba",
        "bei",
        "gei",
        "wo",
        "ni",
        "ta",
        "zhe",
        "na",
        "you",
        "ye",
        "jiu",
        "du",
        "er",
        "ji",
        "yige",
        "shenme",
        "zenme",
        "yi",
        "bu",
        "hen",
        "yao",
        "hui",
        "keyi",
        "neng",
        "dao",
        "shuo",
        "hai",
        "cong",
        "wei",
        "shang",
        "xia",
        "lai",
        "qu",
        "ru",
        "rang",
        "danshi",
        "suoyi",
        "yinywei",
        "ruguo",
        "erqie",
        "bing",
        "qie",
        "zhi",
        "zhishi",
        "yijing",
        "geng",
        "zui",
        "tai",
        "zhen",
        "hao",
        "guo",
        "kan",
        "xiang",
        "zuo",
        "yong",
        "ge",
        "liang",
        "qi",
        "zhong",
        "hou",
        "qian",
        "li",
        "dan",
        "que",
        "jiang",
        "dui",
        "ke",
        "话题",
        "day",
    ]
)


def tokenize(text):
    if not text:
        return []
    text = str(text)
    text = RE_PUNCT.sub(" ", text)
    text = RE_DIGITS.sub("", text)
    words = text.lower().split()
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def load_jsonl(filepath):
    texts = []
    import codecs

    with codecs.open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                texts.append(data)
            except:
                continue
    return texts


def compute_ngrams(texts, n=2, use_title=False, sample_size=5000):
    all_ngrams = Counter()
    sampled = texts[:sample_size] if len(texts) > sample_size else texts
    for item in sampled:
        if use_title:
            text = item.get("note_title", "")
        else:
            text = item.get("desc", "") or item.get("note_content", "")
        words = tokenize(text)
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i : i + n])
                all_ngrams[ngram] += 1
    return all_ngrams


def stream_all(filepath, sample_size=5000, seed=42):
    emoji_desc = Counter()
    emoji_title = Counter()
    tags_desc = Counter()
    tags_title = Counter()
    freq_desc = Counter()
    freq_title = Counter()
    bigrams_desc = Counter()
    bigrams_title = Counter()
    ling_desc = {"exclaim": 0, "question": 0, "ellipsis": 0, "upper": 0, "digit": 0, "link": 0}
    ling_title = {"exclaim": 0, "question": 0, "ellipsis": 0, "upper": 0, "digit": 0, "link": 0}
    total = 0
    rng = random.Random(seed)
    sample = []
    with codecs.open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
            except Exception:
                continue

            text = item.get("desc", "") or item.get("note_content", "")
            title = item.get("note_title", "")

            emoji_desc.update(extract_emojis(text))
            emoji_title.update(extract_emojis(title))
            tags_desc.update(extract_hashtags(text))
            tags_title.update(extract_hashtags(title))

            words = tokenize(text)
            words_title = tokenize(title)

            if len(sample) < sample_size:
                sample.append((text, title, words, words_title))
            else:
                j = rng.randint(0, total)
                if j < sample_size:
                    sample[j] = (text, title, words, words_title)

            freq_desc.update(words)
            freq_title.update(words_title)
            total += 1

    for text, title, words, words_title in sample:
        if len(words) >= 2:
            for bigram in zip(words, words[1:]):
                bigrams_desc[bigram] += 1
        if len(words_title) >= 2:
            for bigram in zip(words_title, words_title[1:]):
                bigrams_title[bigram] += 1
        if text:
            ling_desc["exclaim"] += text.count("!")
            ling_desc["question"] += text.count("?")
            ling_desc["ellipsis"] += text.count("...")
            ling_desc["upper"] += sum(map(str.isupper, text))
            ling_desc["digit"] += sum(map(str.isdigit, text))
            ling_desc["link"] += len(re.findall(r"http[s]?://|www\.", text))
        if title:
            ling_title["exclaim"] += title.count("!")
            ling_title["question"] += title.count("?")
            ling_title["ellipsis"] += title.count("...")
            ling_title["upper"] += sum(map(str.isupper, title))
            ling_title["digit"] += sum(map(str.isdigit, title))
            ling_title["link"] += len(re.findall(r"http[s]?://|www\.", title))

    return {
        "total": total,
        "emoji_desc": emoji_desc.most_common(20),
        "emoji_title": emoji_title.most_common(20),
        "tags_desc": tags_desc.most_common(20),
        "tags_title": tags_title.most_common(20),
        "freq_desc": freq_desc,
        "freq_title": freq_title,
        "bigrams_desc": bigrams_desc.most_common(20),
        "bigrams_title": bigrams_title.most_common(20),
        "ling_desc": ling_desc,
        "ling_title": ling_title,
    }


def extract_emojis(text):
    if not text:
        return []
    return EMOJI_PATTERN.findall(str(text))


def extract_hashtags(text):
    if not text:
        return []
    return re.findall(r"#(\w+)", str(text))


def analyze_unique_words(freqs, files, word_sets=None):
    if word_sets is None:
        word_sets = [set(freq.keys()) for freq in freqs]
    results = {}
    for i in range(len(files)):
        keys_i = word_sets[i]
        for j in range(i + 1, len(files)):
            f1, f2 = files[i], files[j]
            freq1, freq2 = freqs[i], freqs[j]
            keys_j = word_sets[j]
            sorted1 = [
                (w, freq1[w])
                for w in heapq.nlargest(30, keys_i - keys_j, key=freq1.__getitem__)
            ]
            sorted2 = [
                (w, freq2[w])
                for w in heapq.nlargest(30, keys_j - keys_i, key=freq2.__getitem__)
            ]
            key1 = f1 + "_unique_in_" + f2
            key2 = f2 + "_unique_in_" + f1
            results[key1] = sorted1
            results[key2] = sorted2
    common = set.intersection(*word_sets)
    totals = {w: sum(f.get(w, 0) for f in freqs) for w in common}
    results["common_words"] = [
        (w, totals[w]) for w in heapq.nlargest(50, totals, key=totals.get)
    ]
    return results


def analyze_vocabulary_overlap(freqs, files, word_sets=None):
    if word_sets is None:
        word_sets = [set(freq.keys()) for freq in freqs]
    results = {}
    all_sets = word_sets
    sizes = [len(word_set) for word_set in all_sets]
    for f, size in zip(files, sizes):
        results[f + "_total_words"] = size
    pair_common = {}
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            common_count = len(all_sets[i] & all_sets[j])
            union_count = sizes[i] + sizes[j] - common_count
            jaccard = common_count / union_count if union_count else 0
            key = files[i] + "_and_" + files[j]
            pair_common[(i, j)] = common_count
            results[key + "_common"] = common_count
            results[key + "_jaccard"] = round(jaccard, 4)
    common_all = set.intersection(*all_sets)
    union_all = (sum(sizes) - sum(pair_common.values()) + len(common_all))
    results["common_all_three"] = len(common_all)
    results["jaccard_all_three"] = (
        round(len(common_all) / union_all, 4) if union_all else 0
    )
    return results


def main():
    files = [
        "training_set_human.jsonl",
        "training_set_aigc.jsonl",
        "exploring_set.jsonl",
    ]
    file_names = ["training_set_human", "training_set_aigc", "exploring_set"]

    print("=" * 70)
    print("Comprehensive Additional Analysis")
    print("=" * 70)

    stats = {}
    for f in files:
        stats[f] = stream_all(f)
        print("Loaded " + f + ": " + str(stats[f]["total"]) + " records")

    print("\n" + "=" * 70)
    print("1. N-GRAM ANALYSIS (Bigrams)")
    print("=" * 70)

    for use_title in [False, True]:
        label = "Title" if use_title else "Content"
        print("\n--- " + label + " Bigrams ---")
        for f in files:
            top20 = stats[f]["bigrams_title" if use_title else "bigrams_desc"]
            print("\n" + f + " Top 20 bigrams (" + label + "):")
            for ngram, count in top20:
                print("  " + " ".join(ngram) + ": " + str(count))

    print("\n" + "=" * 70)
    print("2. EMOJI ANALYSIS (Content)")
    print("=" * 70)

    for f in files:
        all_emojis = stats[f]["emoji_desc"]
        print("\n" + f + " Top 20 Emojis (Content):")
        for emoji, count in all_emojis:
            print("  " + emoji + ": " + str(count))

    print("\n" + "=" * 70)
    print("2b. EMOJI ANALYSIS (Title)")
    print("=" * 70)

    for f in files:
        all_emojis = stats[f]["emoji_title"]
        print("\n" + f + " Top 20 Emojis (Title):")
        for emoji, count in all_emojis:
            print("  " + emoji + ": " + str(count))

    print("\n" + "=" * 70)
    print("3. HASHTAG ANALYSIS (Content)")
    print("=" * 70)

    for f in files:
        all_tags = stats[f]["tags_desc"]
        print("\n" + f + " Top 20 Hashtags (Content):")
        for tag, count in all_tags:
            print("  #" + tag + ": " + str(count))

    print("\n" + "=" * 70)
    print("3b. HASHTAG ANALYSIS (Title)")
    print("=" * 70)

    for f in files:
        all_tags = stats[f]["tags_title"]
        print("\n" + f + " Top 20 Hashtags (Title):")
        for tag, count in all_tags:
            print("  #" + tag + ": " + str(count))

    print("\n" + "=" * 70)
    print("4. VOCABULARY OVERLAP ANALYSIS (Content)")
    print("=" * 70)

    freqs = [stats[f]["freq_desc"] for f in files]
    desc_word_sets = [set(freq.keys()) for freq in freqs]

    overlap_results = analyze_vocabulary_overlap(freqs, file_names, desc_word_sets)

    print("\n--- Vocabulary Statistics (Content) ---")
    for key, value in overlap_results.items():
        print("  " + key + ": " + str(value))

    print("\n" + "=" * 70)
    print("4b. VOCABULARY OVERLAP ANALYSIS (Title)")
    print("=" * 70)

    freqs_title = [stats[f]["freq_title"] for f in files]
    title_word_sets = [set(freq.keys()) for freq in freqs_title]

    overlap_results_title = analyze_vocabulary_overlap(freqs_title, file_names, title_word_sets)

    print("\n--- Vocabulary Statistics (Title) ---")
    for key, value in overlap_results_title.items():
        print("  " + key + ": " + str(value))

    print("\n" + "=" * 70)
    print("5. UNIQUE WORDS ANALYSIS (Content)")
    print("=" * 70)

    unique_results = analyze_unique_words(freqs, file_names, desc_word_sets)

    print("\n--- Unique Words (Content) ---")
    for key, words in unique_results.items():
        if key != "common_words":
            print("\n" + key + " (Top 10):")
            for w, c in words[:10]:
                print("  " + w + ": " + str(c))

    print("\n--- Common Words (Content - All 3 datasets) ---")
    common_words = unique_results.get("common_words", [])
    for w, c in common_words[:20]:
        print("  " + w + ": " + str(c))

    print("\n" + "=" * 70)
    print("5b. UNIQUE WORDS ANALYSIS (Title)")
    print("=" * 70)

    unique_results_title = analyze_unique_words(freqs_title, file_names, title_word_sets)

    print("\n--- Unique Words (Title) ---")
    for key, words in unique_results_title.items():
        if key != "common_words":
            print("\n" + key + " (Top 10):")
            for w, c in words[:10]:
                print("  " + w + ": " + str(c))

    print("\n--- Common Words (Title - All 3 datasets) ---")
    common_words_title = unique_results_title.get("common_words", [])
    for w, c in common_words_title[:20]:
        print("  " + w + ": " + str(c))

    print("\n" + "=" * 70)
    print("6. ADDITIONAL LINGUISTIC PATTERNS (Content)")
    print("=" * 70)

    for f in files:
        ling = stats[f]["ling_desc"]

        print("\n" + f + " Linguistic Patterns (Content):")
        print("  Exclamation marks: " + str(ling["exclaim"]))
        print("  Question marks: " + str(ling["question"]))
        print("  Ellipses: " + str(ling["ellipsis"]))
        print("  Uppercase chars: " + str(ling["upper"]))
        print("  Digits: " + str(ling["digit"]))
        print("  Links: " + str(ling["link"]))

    print("\n" + "=" * 70)
    print("6b. ADDITIONAL LINGUISTIC PATTERNS (Title)")
    print("=" * 70)

    for f in files:
        ling = stats[f]["ling_title"]

        print("\n" + f + " Linguistic Patterns (Title):")
        print("  Exclamation marks: " + str(ling["exclaim"]))
        print("  Question marks: " + str(ling["question"]))
        print("  Ellipses: " + str(ling["ellipsis"]))
        print("  Uppercase chars: " + str(ling["upper"]))
        print("  Digits: " + str(ling["digit"]))
        print("  Links: " + str(ling["link"]))

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
