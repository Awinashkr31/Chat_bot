import os
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

def main():
    log_file = "logs/unmatched_queries.log"
    if not os.path.exists(log_file):
        print("No unmatched queries log found.")
        return
        
    queries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                query = parts[1]
                queries.append(query)
                
    if not queries:
        print("No queries to analyze.")
        return
        
    unique_queries = list(set(queries))
    if len(unique_queries) < 2:
        print("Not enough unique queries to cluster.")
        for q in unique_queries: print(f" - {q}")
        return
        
    print(f"Analyzing {len(unique_queries)} unique unmatched queries...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(unique_queries)
    
    # We use cosine distance. Distance of 0.4 implies cosine similarity > 0.6
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.4,
        metric='cosine',
        linkage='average'
    )
    labels = clusterer.fit_predict(embeddings)
    
    clusters = defaultdict(list)
    for q, label in zip(unique_queries, labels):
        clusters[label].append(q)
        
    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("\n--- Intent Gap Clusters ---")
    for label, cluster_queries in sorted_clusters:
        if len(cluster_queries) > 1:
            print(f"\nCluster {label} ({len(cluster_queries)} queries):")
            for q in cluster_queries[:10]:
                print(f"  - {q}")
            if len(cluster_queries) > 10:
                print(f"  ... and {len(cluster_queries) - 10} more")

if __name__ == "__main__":
    main()
