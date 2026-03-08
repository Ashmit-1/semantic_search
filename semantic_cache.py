import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, similarity_threshold=0.90):

        self.cache = {}
        self.threshold = similarity_threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding, cluster_id):

        if cluster_id not in self.cache:
            self.miss_count += 1
            return None

        entries = self.cache[cluster_id]

        for entry in entries:

            sim = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            print("Similarity: ", sim)

            if sim >= self.threshold:

                self.hit_count += 1

                return {
                    "matched_query": entry["query"],
                    "similarity": sim,
                    "result": entry["result"]
                }

        self.miss_count += 1
        return None
    
    def add(self, query, query_embedding, result, cluster_id):

        entry = {
            "query": query,
            "embedding": query_embedding,
            "result": result
        }

        if cluster_id not in self.cache:
            self.cache[cluster_id] = []

        self.cache[cluster_id].append(entry)

    def stats(self):

        total_entries = sum(len(v) for v in self.cache.values())

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }
    
    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    