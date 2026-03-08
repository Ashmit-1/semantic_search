import numpy as np
import pandas as pd
import ast
import faiss
import joblib

from sentence_transformers import SentenceTransformer
from semantic_cache import SemanticCache

# Load dataset
df = pd.read_csv("clustered_dataset.csv")

df["cluster_probs"] = df["cluster_probs"].apply(ast.literal_eval)

texts = df["clean_text"].tolist()
dominant_clusters = df["dominant_cluster"].tolist()

# Load embeddings
embeddings = np.load("embeddings.npy").astype("float32")

# Load FAISS index
index = faiss.read_index("newsgroups.index")

# Load models
model = SentenceTransformer("all-MiniLM-L6-v2")

gmm = joblib.load("gmm_model.pkl")

reducer = joblib.load("umap_reducer.pkl")

# Initialize semantic cache
# I have tested for various threshold values but 0.75 seemed to be the sweet spot
cache = SemanticCache(similarity_threshold=0.75)

# Vector search function
def vector_search(query_embedding, k=3):

    query_embedding = np.array([query_embedding]).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        results.append(texts[i])

    return results


# Query loop

print("\nSemantic Search System Ready")
print("Type 'exit' to quit\n")

while True:

    query = input("Enter query: ")

    if query.lower() == "exit":
        print("Exiting system...")
        break


    # Embed query
    query_embedding = model.encode(query)


    # Reduce embedding using UMAP
    reduced_query = reducer.transform([query_embedding])


    # Predict cluster
    cluster_probs = gmm.predict_proba(reduced_query)[0]

    cluster_id = np.argmax(cluster_probs)

    print(f"\nPredicted cluster: {cluster_id}")


    # Check semantic cache
    cache_result = cache.lookup(query_embedding, cluster_id)

    if cache_result:

        print("\nCACHE HIT")
        print("Matched query:", cache_result["matched_query"])
        print("Similarity:", round(cache_result["similarity"], 3))
        print("Result:\n", cache_result["result"])

    else:

        print("\nCACHE MISS — Searching Vector DB")

        results = vector_search(query_embedding)

        result_text = results[0]

        print("\nTop Result:\n")
        print(result_text[:500], "...")

        cache.add(query, query_embedding, result_text, cluster_id)

    print("\nCache stats:", cache.stats())
    print("\n---------------------------------\n")