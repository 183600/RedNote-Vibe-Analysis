# -*- coding: utf-8 -*-
"""基于 22 个跨话题文风词的探索集用户风格聚类（简易 k-means）。

输出：`user_clustering_results.md`（此前误写到 word_correlation_extended_results.md，
与词频相关性脚本的输出语义不符，已改为独立文件名，避免互相覆盖）。
"""
import json
import random
from collections import defaultdict, Counter

from analysis_utils import BASE, post_text

# 22个真正的跨话题通用文风词
cross_topic_words = [
    u'哭惹', u'飞吻', u'姐妹们', u'笑哭', u'害羞', u'浪漫生活的记录者',
    u'派对', u'偷笑', u'萌萌哒', u'宝子们', u'失望', u'种草',
    u'记录吧就现在', u'我的日常', u'日常', u'捂脸', u'所以', u'大学生',
    u'汗颜', u'最后', u'总结', u'但是'
]

def main():
    print("Loading...")
    note_by_user = defaultdict(lambda: defaultdict(int))

    # 探索集里约 64% 记录的顶层 user_id 为空串；先做廉价的子串判定跳过
    # json.loads，其余记录照常解析（该判定已在本数据集全量记录上验证
    # 与 bool(item.get("user_id")) 完全一致）。
    with open(BASE + "exploring_set_fixed.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line or line.isspace():
                continue
            if '"user_id": ""' in line:
                continue
            item = json.loads(line)
            user_id = item.get("user_id")
            if not user_id:
                continue
            text = post_text(item)
            for w in cross_topic_words:
                c = text.count(w)
                if c:
                    note_by_user[user_id][w] += c

    print("Total users: " + str(len(note_by_user)))

    MIN_COUNT = 3
    active_users = {u: c for u, c in note_by_user.items() if sum(c.values()) >= MIN_COUNT}
    print("Active users (>= " + str(MIN_COUNT) + " words): " + str(len(active_users)))

    # 固定种子随机抽样，避免按首现顺序截断造成话题层面的选择偏差
    user_ids = random.Random(42).sample(
        sorted(active_users), min(3000, len(active_users)))
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
        dim = len(data[0]) if n else 0
        centroids = [data[int(n * i / k)] for i in range(k)]
    
        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            for i, row in enumerate(data):
                min_dist = float('inf')
                min_idx = 0
                for j, c in enumerate(centroids):
                    d = 0.0
                    for rv, cx in zip(row, c):
                        diff = rv - cx
                        d += diff * diff
                    if d < min_dist:
                        min_dist = d
                        min_idx = j
                clusters[min_idx].append(i)
        
            new_centroids = []
            for cl in clusters:
                if cl:
                    new_c = [0.0] * dim
                    for i in cl:
                        row = data[i]
                        for x in range(dim):
                            new_c[x] += row[x]
                    for x in range(dim):
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
    word_idx = {w: i for i, w in enumerate(cross_topic_words)}
    results = []
    for cluster_id in range(k):
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
    
        word_avg = defaultdict(float)
        for w in cross_topic_words:
            col = word_idx[w]
            for idx in indices:
                word_avg[w] += matrix[idx][col]
            if len(indices) > 0:
                word_avg[w] /= len(indices)
    
        top_words_cluster = sorted(word_avg.items(), key=lambda x: x[1], reverse=True)[:10]
        results.append((cluster_id, len(indices), top_words_cluster))

    print("\n=== Results ===")
    for cluster_id, size, top_words_cluster in results:
        print("\n--- Cluster " + str(cluster_id) + " (" + str(size) + " users) ---")
        for w, v in top_words_cluster:
            if v > 0.001:
                print("  " + w + ": " + str(round(v, 4)))

    # Save to file
    with open(BASE + "user_clustering_results.md", "wb") as f:
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


if __name__ == "__main__":
    main()
