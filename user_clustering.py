# -*- coding: utf-8 -*-
import json
import re
import math
from collections import defaultdict, Counter

# 22个真正的跨话题通用文风词
cross_topic_words = [
    u'哭惹', u'飞吻', u'姐妹们', u'笑哭', u'害羞', u'浪漫生活的记录者',
    u'派对', u'偷笑', u'萌萌哒', u'宝子们', u'失望', u'种草',
    u'记录吧就现在', u'我的日常', u'日常', u'捂脸', u'所以', u'大学生',
    u'汗颜', u'最后', u'总结', u'但是'
]

print("Loading...")
note_by_user = defaultdict(lambda: defaultdict(int))

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
            for w in words:
                if w in cross_topic_words:
                    note_by_user[user_id][w] += 1
        except:
            pass

print("Total users: " + str(len(note_by_user)))

MIN_COUNT = 3
active_users = {u: c for u, c in note_by_user.items() if sum(c.values()) >= MIN_COUNT}
print("Active users (>= " + str(MIN_COUNT) + " words): " + str(len(active_users)))

user_ids = list(active_users.keys())[:3000]
n_users = len(user_ids)
n_words = len(cross_topic_words)

print("Matrix: " + str(n_users) + " x " + str(n_words))

# Build feature matrix (normalized)
matrix = []
for uid in user_ids:
    total = float(sum(active_users[uid].values()))
    if total == 0:
        total = 1
    row = [active_users[uid].get(w, 0) / total for w in cross_topic_words]
    matrix.append(row)

def simple_kmeans(data, k, max_iter=20):
    n = len(data)
    centroids = [data[int(n * i / k)] for i in range(k)]
    
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for i in range(n):
            min_dist = float('inf')
            min_idx = 0
            for j, c in enumerate(centroids):
                d = sum((data[i][x] - c[x]) ** 2 for x in range(len(data[i])))
                if d < min_dist:
                    min_dist = d
                    min_idx = j
            clusters[min_idx].append(i)
        
        new_centroids = []
        for cl in clusters:
            if cl:
                new_c = [0.0] * len(data[0])
                for i in cl:
                    for x in range(len(data[0])):
                        new_c[x] += data[i][x]
                for x in range(len(data[0])):
                    new_c[x] /= len(cl)
                new_centroids.append(new_c)
            else:
                new_centroids.append(centroids[len(new_centroids)])
        centroids = new_centroids
    
    labels = [0] * n
    for j, cl in enumerate(clusters):
        for i in cl:
            labels[i] = j
    return labels, centroids

print("Clustering...")
k = 5
labels, centroids = simple_kmeans(matrix, k)

# Analyze clusters
results = []
for cluster_id in range(k):
    indices = [i for i, l in enumerate(labels) if l == cluster_id]
    
    word_avg = defaultdict(float)
    for w in cross_topic_words:
        for idx in indices:
            word_avg[w] += matrix[idx][cross_topic_words.index(w)]
        if len(indices) > 0:
            word_avg[w] /= len(indices)
    
    top_words_cluster = sorted(word_avg.items(), key=lambda x: x[1], reverse=True)[:10]
    results.append((cluster_id, len(indices), top_words_cluster))

print("\n=== Results ===")
for cluster_id, size, top_words_cluster in results:
    print("\n--- Cluster " + str(cluster_id) + " (" + str(size) + " users) ---")
    for w, v in top_words_cluster:
        if v > 0.001:
            print("  " + w.encode('utf-8') + ": " + str(round(v, 4)))

# Save to file
with open("word_correlation_extended_results.md", "wb") as f:
    f.write((u"# 基于22个跨话题文风词的用户聚类分析\n\n").encode('utf-8'))
    f.write((u"使用22个在所有话题词频都超过0.05%的跨话题词进行用户分类\n\n").encode('utf-8'))
    
    for cluster_id, size, top_words_cluster in results:
        f.write((u"## Cluster " + str(cluster_id) + u" (" + str(size) + u" users)\n\n").encode('utf-8'))
        f.write((u"特征词:\n\n").encode('utf-8'))
        for w, v in top_words_cluster:
            if v > 0.001:
                f.write((u"- " + w + u": " + str(round(v, 4)) + u"\n").encode('utf-8'))
        f.write(u"\n".encode('utf-8'))

print("\nDone!")