### Run the project using Docker

Clone/Open the project directory:


 ```
cd semantic_search
```
Start the service:

```
docker compose up --build
```
#### Access the API:

Base URL:

```

 http://localhost:8000
```

Interactive Docs (Swagger): 
```
http://localhost:8000/docs
```


---

# Decisions that I have taken at different parts of the assignment and the justifications:

### Loading the Dataset
The news dataset is a text dataset with folders as labels and the files inside as the actual data. I loaded the data from those files into two Python lists: `document`, containing the text content, and `label`, containing the label for each document.

---

### Storing Dataset in Structured Format
Using data directly from files is inconvenient, so I stored it in a pandas DataFrame and exported it in CSV format for easier future use.

---

### Cleaning the Dataset
The dataset contained unnecessary information such as email-like headers, footers, and spaces. To improve the quality of vector embeddings, I removed:

- **Headers**: Newsgroups, Path, From, Message-ID, Sender, Organization, References, Date, Lines, etc.
- **Footers**: Sections separated by "--" (two hyphens).
- **Empty spaces**: Unnecessary spaces scattered throughout the documents.

I retained:
- **Subject**: Most articles have a subject, which conveys relevant information.
- **Actual text content**: The main body of the document between the header and footer.

---

### Converting to Vector Embeddings
The cleaned data was converted to vector embeddings using the `all-MiniLM-L6-v2` model. This model is fast and memory-efficient, making it suitable for this application. Other models were either too heavy or did not perform well. LLM-based embedding models were not considered due to cost and unnecessary complexity.

The embeddings were exported as a NumPy array and also stored in CSV format for further use.

---

### Storing in a Vector Store
For this application, I chose **FAISS** as the vector database because:
- It is lightweight, aligning with the requirement to build a lightweight semantic search engine.
- Alternatives like ChromaDB or Pinecone have heavier dependencies and are overkill for this use case.

For retrieval, I used **cosine similarity**. Since FAISS internally uses inner product, I normalized the embeddings to make them equivalent.

---

### Clustering
For clustering, I used **Gaussian Mixture Model (GMM)** for the following reasons:
- Hard clustering (e.g., k-means) was not acceptable for this assignment.
- GMM models data as a mixture of probability distributions, allowing us to see the probability of an article belonging to a cluster rather than assigning a hard label.
- GMM is probabilistic, stable, and a standard ML approach.
- It is not computationally expensive, keeping the overall system lightweight.

#### Challenges with GMM
GMM does not perform well in high-dimensional spaces. The embeddings generated have a size of 384, which is too high for GMM.

#### Dimensionality Reduction
To address this, I used **UMAP (Uniform Manifold Approximation and Projection)** to reduce the dimensions to 10. UMAP preserves information effectively and captures local topology due to its non-linear nature.

With reduced dimensions, the GMM criterion was set to `full` instead of `diag`, allowing the model to capture non-linear relationships.

#### Determining the Number of Clusters
The number of clusters was determined using the **Bayesian Information Criterion (BIC)**, which computes the log-likelihood among clusters. Lower BIC values indicate more distinct clusters.

---

### Finalizing the Number of Clusters
From the BIC values and the plotted graph, the global minimum occurred at around 245 clusters. However, 245 clusters for a dataset of 20,000 samples would make the search too specific and lose generalization.

Observations:
- The steepest dip in BIC occurred between 0 and 100 clusters.
- Numerically, the highest dip was between 5 and 25 clusters.

To maintain generalization, I chose **25 clusters**.

#### Pros:
- The search remains general and not overly specific.
- Overfitting is avoided.
- A high cluster count would result in each article having a near-1 probability for one cluster and negligible probabilities for others, effectively labeling rather than clustering.
- Fewer outliers are expected.

#### Cons:
- A larger number of clusters could have improved search and cache hit speed.

---

### Cluster Analysis
To evaluate clustering performance, I:
- Reduced dimensions and selected 2D vectors for plotting.
- Plotted the distribution of clusters across two features.
- Plotted the distribution of outliers (documents with a cluster membership probability < 0.1).
- Plotted entropies to measure uncertainty in cluster assignments. Entropy near 0 indicates strong cluster membership.
- Computed **TF-IDF** for each cluster and displayed the top 5 words to identify probable cluster topics.

---

### Cache 

- While building the cache I make it using python dictionary and python lists.
- I have kept the similarity threshold to 0.75 which means any query which has a similarity of 0.75 or above with previous query will be a cache hit.

#### Role of clustering in cache:
- Without the clustering the cache search would have been O(n) but with the clustering the search is reduced to O (n / cluster count)
- This is because, when the api gets a user query it first computes the most similar cluster and then it looks in the cache and if it is a miss it retrieves documents from only the identified cluster.