import os
import numpy as np
import pandas as pd
import ast
import faiss
import joblib
import umap

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from semantic_cache import SemanticCache


# FastAPI initialization

app = FastAPI(title="Semantic Search API")


# Request Schema

class QueryRequest(BaseModel):
    query: str


# Application State

class AppState:
    model = None
    reducer = None
    gmm = None
    index = None
    cache = None
    texts = None


state = AppState()


# Startup Event

@app.on_event("startup")
async def startup_event():

    print("Loading models...")

    # Load dataset
    df = pd.read_csv("cluster_probs.csv")

    df["cluster_probs"] = df["cluster_probs"].apply(ast.literal_eval)

    df = pd.read_csv("clean_text.csv")
    state.texts = df["clean_text"].tolist()

    # Load embeddings
    embeddings = np.load("models/embeddings.npy").astype("float32")

    # Load FAISS index
    state.index = faiss.read_index("models/newsgroups.index")

    # Load ML models
    state.model = SentenceTransformer("all-MiniLM-L6-v2")


    # if the reducer exits then load it otherwise create it and then load it
    state.reducer = None
    if os.path.exists("models/umap_reducer.pkl"):
        state.reducer = joblib.load("models/umap_reducer.pkl")
    else:
        embeddings = np.load("models/embeddings.npy")
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=10,
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        os.makedirs("models", exist_ok=True)  
        joblib.dump(reducer, "models/umap_reducer.pkl")
        state.reducer = joblib.load("models/umap_reducer.pkl")

    state.gmm = joblib.load("models/gmm_model.pkl")

    # Initialize cache
    state.cache = SemanticCache(similarity_threshold=0.75)

    print("System ready.")


# Vector search

def vector_search(query_embedding, k=3):

    query_embedding = np.array([query_embedding]).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = state.index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        results.append(state.texts[i])

    return results


# Query Endpoint

@app.post("/query")
async def query_endpoint(request: QueryRequest):

    query = request.query

    # Embed query
    query_embedding = state.model.encode(query)

    # Dimensionality reduction
    reduced_query = state.reducer.transform([query_embedding])

    # Predict cluster
    cluster_probs = state.gmm.predict_proba(reduced_query)[0]

    cluster_id = int(np.argmax(cluster_probs))

    # Cache lookup
    cache_result = state.cache.lookup(query_embedding, cluster_id)

    if cache_result:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": float(cache_result["similarity"]),
            "result": cache_result["result"],
            "dominant_cluster": cluster_id
        }

    # Cache miss → vector search
    results = vector_search(query_embedding)

    result_text = results[0]

    # Store in cache
    state.cache.add(query, query_embedding, result_text, cluster_id)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_text,
        "dominant_cluster": cluster_id
    }


# Cache statistics endpoint

@app.get("/cache/stats")
async def cache_stats():

    return state.cache.stats()


# Clear cache endpoint

@app.delete("/cache")
async def clear_cache():

    state.cache.clear()

    return {"message": "Cache cleared successfully"}