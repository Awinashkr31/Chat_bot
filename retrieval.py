import os
import faiss
import numpy as np

class RetrievalSystem:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.index = None
        self.chunks = []
        
    def load_documents(self, docs_dir="docs"):
        if not os.path.exists(docs_dir):
            return
            
        all_text = ""
        for file in os.listdir(docs_dir):
            if file.endswith(".txt") or file.endswith(".md"):
                with open(os.path.join(docs_dir, file), "r", encoding="utf-8") as f:
                    all_text += f.read() + "\n\n"
                    
        # Simple chunking logic (approx 200 words per chunk, 50 words overlap)
        words = all_text.split()
        chunk_size = 200
        overlap = 50
        
        self.chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 20:
                self.chunks.append(chunk)
                
        if self.chunks:
            embeddings = self.embed_model.encode(self.chunks, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            dimension = embeddings.shape[1]
            # Use Inner Product (Cosine Similarity on normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            print(f"Indexed {len(self.chunks)} chunks in FAISS")

    def search(self, query, top_k=2, threshold=0.45):
        if not self.index or not self.chunks:
            return None
            
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        
        similarities, indices = self.index.search(q_emb, top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and sim >= threshold:
                results.append(self.chunks[idx])
        
        if results:
            return "Here's what I found in the documents:\n\n" + "\n\n---\n\n".join(results)
        return None
