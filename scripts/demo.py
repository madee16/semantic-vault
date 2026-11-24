"""
SemanticVault - Interactive Local Testing & Demo Script
Demonstrates end-to-end functionality against a running SemanticVault server:
1. Health check verification
2. API Key creation with scoped permissions
3. Document ingestion with text chunking & embedding generation
4. Semantic vector search & RRF hybrid search
"""
import sys
import time
import httpx

BASE_URL = "http://localhost:8000/api/v1"

def print_step(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def main():
    print_step("Step 1: Checking SemanticVault API Health")
    try:
        with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
            resp = client.get("/health")
            if resp.status_code != 200:
                print(f"[!] Health check failed with status {resp.status_code}: {resp.text}")
                sys.exit(1)
            data = resp.json()
            print(f"[+] Status:             {data['status']}")
            print(f"[+] Database:           {data['database']}")
            print(f"[+] Ollama Engine:      {data['ollama']}")
            print(f"[+] Embedding Model:    {data['embedding_model']} ({data['embedding_dimension']} dimensions)")
    except Exception as e:
        print(f"[!] Could not connect to {BASE_URL}: {e}")
        print("[*] Make sure the server is running: .venv\\Scripts\\uvicorn.exe src.main:app --reload")
        sys.exit(1)

    print_step("Step 2: Generating Developer API Key")
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        key_payload = {
            "email": f"local_tester_{int(time.time())}@example.com",
            "name": "Local Tester",
            "key_name": "Local Interactive Key",
            "scopes": ["documents:read", "documents:write", "search:read", "admin:keys"],
            "monthly_quota": 5000,
        }
        resp = client.post("/keys", json=key_payload)
        resp.raise_for_status()
        key_data = resp.json()
        raw_key = key_data["raw_key"]
        print(f"[+] Key Name:       {key_data['name']}")
        print(f"[+] API Key Prefix: {key_data['key_prefix']}")
        print(f"[+] Full Raw Key:   {raw_key}")
        print(f"[+] Scopes:         {', '.join(key_data['scopes'])}")
        print(f"[+] Monthly Quota:  {key_data['monthly_quota']} requests (Period: {key_data['current_period']})")

    auth_headers = {"X-API-Key": raw_key}

    print_step("Step 3: Ingesting Sample Engineering Documents")
    sample_docs = [
        (
            "postgresql_pgvector.txt",
            (
                "PostgreSQL pgvector is an open-source vector similarity search extension. "
                "It enables storing high-dimensional embeddings directly alongside relational tables. "
                "pgvector supports HNSW (Hierarchical Navigable Small World) and IVFFlat indexes, "
                "offering sub-10ms nearest-neighbor lookups using cosine distance (<=>) and L2 distance (<->)."
            ),
            {"topic": "database", "technology": "postgresql"},
        ),
        (
            "hybrid_retrieval_rrf.md",
            (
                "# Hybrid Search with Reciprocal Rank Fusion (RRF)\n\n"
                "Dense vector search excels at conceptual semantic retrieval, but can struggle with "
                "exact keyword terms such as version strings or part numbers. "
                "Reciprocal Rank Fusion (RRF) combines dense vector rankings with sparse keyword rankings "
                "without requiring fragile linear score normalization."
            ),
            {"topic": "information_retrieval", "technology": "algorithms"},
        ),
    ]

    with httpx.Client(base_url=BASE_URL, headers=auth_headers, timeout=60.0) as client:
        for filename, content, meta in sample_docs:
            files = {"file": (filename, content.encode("utf-8"), "text/plain")}
            data = {"chunk_size": 250, "chunk_overlap": 30, "metadata": str(meta).replace("'", '"')}
            
            print(f"[*] Ingesting '{filename}'...")
            start_t = time.time()
            resp = client.post("/documents", files=files, data=data)
            elapsed = (time.time() - start_t) * 1000
            
            if resp.status_code == 409:
                print(f"    [-] Already indexed (duplicate content hash detected).")
            else:
                resp.raise_for_status()
                doc_info = resp.json()
                print(f"    [+] Ingested ID: {doc_info['id']} ({doc_info['total_chunks']} chunks, {elapsed:.1f}ms)")

    print_step("Step 4: Executing Semantic Vector Search")
    test_queries = [
        "How fast are nearest neighbor lookups in pgvector?",
        "Why is Reciprocal Rank Fusion used for combining search results?",
    ]

    with httpx.Client(base_url=BASE_URL, headers=auth_headers, timeout=30.0) as client:
        for q in test_queries:
            print(f"\n[*] Query: \"{q}\"")
            search_payload = {
                "query": q,
                "top_k": 2,
                "hybrid": True,
                "similarity_threshold": 0.4,
            }
            start_t = time.time()
            resp = client.post("/search", json=search_payload)
            elapsed = (time.time() - start_t) * 1000
            resp.raise_for_status()
            search_res = resp.json()
            
            print(f"    [+] Mode: {search_res['search_mode']} | Found: {search_res['total_results']} chunks | Latency: {elapsed:.1f}ms")
            for idx, r in enumerate(search_res["results"], 1):
                print(f"    [{idx}] Score: {r['similarity_score']:.4f} | File: {r['filename']}")
                preview = r['content'].replace('\n', ' ')[:120]
                print(f"        \"{preview}...\"")

    print_step("Local Testing Completed Successfully!")
    print("[+] All endpoints responded with expected enterprise behavior.")
    print("[+] Interactive Swagger Docs: http://localhost:8000/docs\n")

if __name__ == "__main__":
    main()
