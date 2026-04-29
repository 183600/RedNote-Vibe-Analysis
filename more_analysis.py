# -*- coding: utf-8 -*-
import json
import re
import codecs
from collections import Counter

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
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
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


def extract_emojis(text):
    if not text:
        return []
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\U00002702-\U000027b0\U000024c2-\U0001f251\U0001f190-\U0001f19a"
        "]+"
    )
    return emoji_pattern.findall(str(text))


def extract_hashtags(text):
    if not text:
        return []
    return re.findall(r"#(\w+)", str(text))


def analyze_unique_words(freqs, files):
    results = {}
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            f1, f2 = files[i], files[j]
            freq1, freq2 = freqs[i], freqs[j]
            unique1 = set(freq1.keys()) - set(freq2.keys())
            unique2 = set(freq2.keys()) - set(freq1.keys())
            sorted1 = sorted([(w, freq1[w]) for w in unique1], key=lambda x: -x[1])[:30]
            sorted2 = sorted([(w, freq2[w]) for w in unique2], key=lambda x: -x[1])[:30]
            key1 = f1 + "_unique_in_" + f2
            key2 = f2 + "_unique_in_" + f1
            results[key1] = sorted1
            results[key2] = sorted2
    common = set.intersection(*[set(f.keys()) for f in freqs])
    results["common_words"] = sorted(
        [(w, sum(f.get(w, 0) for f in freqs)) for w in common], key=lambda x: -x[1]
    )[:50]
    return results


def analyze_vocabulary_overlap(freqs, files):
    results = {}
    all_sets = []
    for f, freq in zip(files, freqs):
        word_set = set(freq.keys())
        all_sets.append(word_set)
        results[f + "_total_words"] = len(word_set)
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            common = all_sets[i] & all_sets[j]
            union = all_sets[i] | all_sets[j]
            jaccard = len(common) / len(union) if union else 0
            key = files[i] + "_and_" + files[j]
            results[key + "_common"] = len(common)
            results[key + "_jaccard"] = round(jaccard, 4)
    common_all = set.intersection(*all_sets)
    union_all = set.union(*all_sets)
    results["common_all_three"] = len(common_all)
    results["jaccard_all_three"] = (
        round(len(common_all) / len(union_all), 4) if union_all else 0
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

    texts_data = {}
    for f in files:
        texts_data[f] = load_jsonl(f)
        print("Loaded " + f + ": " + str(len(texts_data[f])) + " records")

    print("\n" + "=" * 70)
    print("1. N-GRAM ANALYSIS (Bigrams)")
    print("=" * 70)

    for use_title in [False, True]:
        label = "Title" if use_title else "Content"
        print("\n--- " + label + " Bigrams ---")
        for f in files:
            ngrams = compute_ngrams(texts_data[f], n=2, use_title=use_title)
            top20 = ngrams.most_common(20)
            print("\n" + f + " Top 20 bigrams (" + label + "):")
            for ngram, count in top20:
                print("  " + " ".join(ngram) + ": " + str(count))

    print("\n" + "=" * 70)
    print("2. EMOJI ANALYSIS (Content)")
    print("=" * 70)

    for f in files:
        all_emojis = Counter()
        for item in texts_data[f]:
            text = item.get("desc", "") or item.get("note_content", "")
            all_emojis.update(extract_emojis(text))
        print("\n" + f + " Top 20 Emojis (Content):")
        for emoji, count in all_emojis.most_common(20):
            print("  " + emoji + ": " + str(count))

    print("\n" + "=" * 70)
    print("2b. EMOJI ANALYSIS (Title)")
    print("=" * 70)

    for f in files:
        all_emojis = Counter()
        for item in texts_data[f]:
            title = item.get("note_title", "")
            all_emojis.update(extract_emojis(title))
        print("\n" + f + " Top 20 Emojis (Title):")
        for emoji, count in all_emojis.most_common(20):
            print("  " + emoji + ": " + str(count))

    print("\n" + "=" * 70)
    print("3. HASHTAG ANALYSIS (Content)")
    print("=" * 70)

    for f in files:
        all_tags = Counter()
        for item in texts_data[f]:
            text = item.get("desc", "") or item.get("note_content", "")
            all_tags.update(extract_hashtags(text))
        print("\n" + f + " Top 20 Hashtags (Content):")
        for tag, count in all_tags.most_common(20):
            print("  #" + tag + ": " + str(count))

    print("\n" + "=" * 70)
    print("3b. HASHTAG ANALYSIS (Title)")
    print("=" * 70)

    for f in files:
        all_tags = Counter()
        for item in texts_data[f]:
            title = item.get("note_title", "")
            all_tags.update(extract_hashtags(title))
        print("\n" + f + " Top 20 Hashtags (Title):")
        for tag, count in all_tags.most_common(20):
            print("  #" + tag + ": " + str(count))

    print("\n" + "=" * 70)
    print("4. VOCABULARY OVERLAP ANALYSIS (Content)")
    print("=" * 70)

    freqs = []
    for f in files:
        words = []
        for item in texts_data[f]:
            text = item.get("desc", "") or item.get("note_content", "")
            words.extend(tokenize(text))
        freqs.append(Counter(words))

    overlap_results = analyze_vocabulary_overlap(freqs, file_names)

    print("\n--- Vocabulary Statistics (Content) ---")
    for key, value in overlap_results.items():
        print("  " + key + ": " + str(value))

    print("\n" + "=" * 70)
    print("4b. VOCABULARY OVERLAP ANALYSIS (Title)")
    print("=" * 70)

    freqs_title = []
    for f in files:
        words = []
        for item in texts_data[f]:
            title = item.get("note_title", "")
            words.extend(tokenize(title))
        freqs_title.append(Counter(words))

    overlap_results_title = analyze_vocabulary_overlap(freqs_title, file_names)

    print("\n--- Vocabulary Statistics (Title) ---")
    for key, value in overlap_results_title.items():
        print("  " + key + ": " + str(value))

    print("\n" + "=" * 70)
    print("5. UNIQUE WORDS ANALYSIS (Content)")
    print("=" * 70)

    unique_results = analyze_unique_words(freqs, file_names)

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

    unique_results_title = analyze_unique_words(freqs_title, file_names)

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
        punct_counts = {"exclaim": 0, "question": 0, "ellipsis": 0}
        upper_counts = 0
        digit_counts = 0
        link_counts = 0

        sample = texts_data[f][:5000]
        for item in sample:
            text = item.get("desc", "") or item.get("note_content", "")
            if text:
                punct_counts["exclaim"] += text.count("!")
                punct_counts["question"] += text.count("?")
                punct_counts["ellipsis"] += text.count("...")
                upper_counts += sum(1 for c in text if c.isupper())
                digit_counts += sum(1 for c in text if c.isdigit())
                link_counts += len(re.findall(r"http[s]?://|www\.", text))

        print("\n" + f + " Linguistic Patterns (Content):")
        print("  Exclamation marks: " + str(punct_counts["exclaim"]))
        print("  Question marks: " + str(punct_counts["question"]))
        print("  Ellipses: " + str(punct_counts["ellipsis"]))
        print("  Uppercase chars: " + str(upper_counts))
        print("  Digits: " + str(digit_counts))
        print("  Links: " + str(link_counts))

    print("\n" + "=" * 70)
    print("6b. ADDITIONAL LINGUISTIC PATTERNS (Title)")
    print("=" * 70)

    for f in files:
        punct_counts = {"exclaim": 0, "question": 0, "ellipsis": 0}
        upper_counts = 0
        digit_counts = 0
        link_counts = 0

        sample = texts_data[f][:5000]
        for item in sample:
            title = item.get("note_title", "")
            if title:
                punct_counts["exclaim"] += title.count("!")
                punct_counts["question"] += title.count("?")
                punct_counts["ellipsis"] += title.count("...")
                upper_counts += sum(1 for c in title if c.isupper())
                digit_counts += sum(1 for c in title if c.isdigit())
                link_counts += len(re.findall(r"http[s]?://|www\.", title))

        print("\n" + f + " Linguistic Patterns (Title):")
        print("  Exclamation marks: " + str(punct_counts["exclaim"]))
        print("  Question marks: " + str(punct_counts["question"]))
        print("  Ellipses: " + str(punct_counts["ellipsis"]))
        print("  Uppercase chars: " + str(upper_counts))
        print("  Digits: " + str(digit_counts))
        print("  Links: " + str(link_counts))

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
