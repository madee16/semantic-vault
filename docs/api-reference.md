# SemanticVault: API Reference & Developer Guide

SemanticVault exposes a RESTful JSON API served over FastAPI. All endpoints require standard HTTP headers and authentication credentials where indicated.

- **Base URL:** `http://localhost:8000/api/v1`
- **Interactive OpenAPI Documentation:** `http://localhost:8000/docs`
- **ReDoc Specification:** `http://localhost:8000/redoc`

---

## 1. Authentication & Security Headers

All protected endpoints require an active developer API key passed in the custom HTTP header:

```http
X-API-Key: sv_live_...
```

### Permission Scopes
API keys are bounded by granular scopes:
| Scope | Description |
| :--- | :--- |
| `documents:read` | Read ingested documents, metadata, and chunk listings |
| `documents:write` | Ingest new documents and trigger chunk deletion |
| `search:read` | Execute semantic, hybrid, and filtered vector queries |
| `admin:keys` | Issue, list, and revoke API keys |
| `admin:all` | Full administrative access across all system resources |

---

## 2. API Endpoints

### 2.1 Health Check
Check operational status of the API, database connectivity, and local Ollama model engine.

- **Method:** `GET`
- **Path:** `/api/v1/health`
- **Authentication:** Public (No API key required)

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "project": "SemanticVault",
  "version": "1.0.0",
  "database": "connected",
  "ollama": "connected",
  "embedding_model": "embeddinggemma",
  "embedding_dimension": 768
}
```

#### Curl Example
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

---

### 2.2 Create API Key
Issue a cryptographically secure, scoped API key for a developer.

- **Method:** `POST`
- **Path:** `/api/v1/keys`
- **Authentication:** Public / Admin Key

#### Request Body
```json
{
  "developer_email": "developer@example.com",
  "developer_name": "Dev Organization",
  "name": "Production Backend Key",
  "scopes": ["documents:read", "documents:write", "search:read"],
  "monthly_quota": 10000,
  "expires_in_days": 90
}
```

#### Response (`201 Created`)
```json
{
  "id": "6a3e8785-555e-4bb0-8012-70b1cb7a61d1",
  "developer_id": "7b55f1f2-1a4c-473d-9d47-66a90dcbb302",
  "name": "Production Backend Key",
  "raw_key": "sv_live_4a8df9e81b6c5432098d... (displayed ONCE only)",
  "key_prefix": "sv_live_4a8df9e8",
  "scopes": ["documents:read", "documents:write", "search:read"],
  "monthly_quota": 10000,
  "requests_used": 0,
  "expires_at": "2027-02-22T00:00:00Z",
  "created_at": "2026-11-24T02:00:00Z"
}
```

#### Curl Example
```bash
curl -X POST "http://localhost:8000/api/v1/keys" \
  -H "Content-Type: application/json" \
  -d '{
    "developer_email": "developer@example.com",
    "developer_name": "Dev Organization",
    "name": "Search Key",
    "scopes": ["search:read"],
    "monthly_quota": 5000
  }'
```

---

### 2.3 List & Revoke API Keys
Inspect active credentials or revoke a compromised key immediately.

- **List Keys:** `GET /api/v1/keys` (Scope: `admin:keys` or `admin:all`)
- **Revoke Key:** `DELETE /api/v1/keys/{key_id}` (Scope: `admin:keys` or `admin:all`)

#### Curl Revocation Example
```bash
curl -X DELETE "http://localhost:8000/api/v1/keys/6a3e8785-555e-4bb0-8012-70b1cb7a61d1" \
  -H "X-API-Key: sv_live_admin_secret_key"
```

---

### 2.4 Document Ingestion
Upload and index a text, Markdown, or PDF document. The platform automatically normalizes text, parses structure, splits into sliding-window chunks, and computes dense embeddings.

- **Method:** `POST`
- **Path:** `/api/v1/documents`
- **Scope Required:** `documents:write`
- **Content-Type:** `multipart/form-data`

#### Form Parameters
- `file`: The document binary (`.txt`, `.md`, or `.pdf`).
- `chunk_size` *(optional, int, default: 500)*: Target character length per chunk.
- `chunk_overlap` *(optional, int, default: 50)*: Overlap between consecutive chunks.
- `metadata` *(optional, JSON string)*: Custom key-value pairs (e.g. `{"author": "Engineering", "category": "architecture"}`).

#### Response (`201 Created`)
```json
{
  "id": "1d2e3f4a-5b6c-7d8e-9f0a-1b2c3d4e5f6a",
  "filename": "neural_architecture.pdf",
  "file_type": "pdf",
  "status": "indexed",
  "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "total_chunks": 14,
  "metadata_json": {
    "author": "Engineering",
    "category": "architecture"
  },
  "created_at": "2026-11-24T02:10:00Z",
  "updated_at": "2026-11-24T02:10:01Z"
}
```

#### Curl Ingestion Example
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "X-API-Key: sv_live_your_api_key_here" \
  -F "file=@./system_spec.md" \
  -F "chunk_size=600" \
  -F "chunk_overlap=60" \
  -F 'metadata={"team": "AI", "project": "SemanticVault"}'
```

---

### 2.5 List & Delete Documents
- **List Documents:** `GET /api/v1/documents?limit=20&offset=0` (Scope: `documents:read`)
- **Get Document Details:** `GET /api/v1/documents/{doc_id}` (Scope: `documents:read`)
- **Delete Document:** `DELETE /api/v1/documents/{doc_id}` (Scope: `documents:write`)
  - Deleting a document triggers cascade deletion in PostgreSQL, removing all associated embeddings from the HNSW vector index.

---

### 2.6 Semantic & Hybrid Search
Execute semantic similarity queries against indexed knowledge with top-k ranking, threshold filtering, and hybrid RRF support.

- **Method:** `POST`
- **Path:** `/api/v1/search`
- **Scope Required:** `search:read`
- **Content-Type:** `application/json`

#### Request Body Schema
```json
{
  "query": "How does vector cosine distance calculate similarity?",
  "top_k": 5,
  "similarity_threshold": 0.65,
  "hybrid": true,
  "hybrid_alpha": 0.5,
  "rewrite_query": false,
  "metadata_filter": {
    "project": "SemanticVault"
  }
}
```

#### Parameter Breakdown
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | *(Required)* | Natural language search query |
| `top_k` | `integer` | `5` | Maximum number of results to retrieve (1 to 100) |
| `similarity_threshold`| `float` | `null` | Minimum similarity score (0.0 to 1.0) |
| `hybrid` | `boolean` | `false` | Enable Reciprocal Rank Fusion of vector and lexical results |
| `hybrid_alpha` | `float` | `0.5` | Hybrid weight: 1.0 = pure vector, 0.0 = pure keyword |
| `rewrite_query` | `boolean` | `false` | Expand and rewrite query using local LLM before search |
| `metadata_filter` | `object` | `null` | Exact match key-value filter on document metadata |

#### Response (`200 OK`)
```json
{
  "query": "How does vector cosine distance calculate similarity?",
  "rewritten_query": null,
  "total_results": 2,
  "search_mode": "hybrid",
  "execution_time_ms": 14.8,
  "results": [
    {
      "chunk_id": "c71a39f0-32df-4796-bf70-9831ce81bf21",
      "document_id": "1d2e3f4a-5b6c-7d8e-9f0a-1b2c3d4e5f6a",
      "filename": "concepts.md",
      "chunk_index": 0,
      "content": "Cosine similarity measures the angle between two non-zero vectors in an inner product space...",
      "similarity_score": 0.9421,
      "metadata": {
        "project": "SemanticVault"
      }
    }
  ]
}
```

#### Curl Search Example
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sv_live_your_api_key_here" \
  -d '{
    "query": "What is the HNSW algorithm?",
    "top_k": 3,
    "hybrid": true
  }'
```

---

## 3. Python SDK / HTTPX Integration Example

The following standalone Python script demonstrates complete programmatic integration with SemanticVault:

```python
import httpx

BASE_URL = "http://localhost:8000/api/v1"
DEV_EMAIL = "engineer@example.com"

# 1. Generate an API Key
with httpx.Client(base_url=BASE_URL) as client:
    key_resp = client.post("/keys", json={
        "developer_email": DEV_EMAIL,
        "developer_name": "AI Engineer",
        "name": "Integration Key",
        "scopes": ["documents:read", "documents:write", "search:read"],
        "monthly_quota": 5000,
    })
    key_resp.raise_for_status()
    api_key = key_resp.json()["raw_key"]
    print(f"[+] Obtained API Key: {api_key}")

# 2. Ingest a Document
auth_headers = {"X-API-Key": api_key}
with httpx.Client(base_url=BASE_URL, headers=auth_headers) as client:
    sample_content = (
        "PostgreSQL pgvector enables vector similarity search directly inside SQL. "
        "It supports HNSW and IVFFlat index types with cosine, L2, and inner product distance metrics."
    )
    files = {"file": ("pgvector_overview.txt", sample_content.encode("utf-8"), "text/plain")}
    data = {"chunk_size": 300, "chunk_overlap": 30, "metadata": '{"category": "database"}'}

    doc_resp = client.post("/documents", files=files, data=data)
    doc_resp.raise_for_status()
    doc_id = doc_resp.json()["id"]
    print(f"[+] Ingested Document ID: {doc_id} with {doc_resp.json()['total_chunks']} chunks")

    # 3. Execute Hybrid Semantic Search
    search_resp = client.post("/search", json={
        "query": "What index types does pgvector support?",
        "top_k": 3,
        "hybrid": True,
        "similarity_threshold": 0.5,
    })
    search_resp.raise_for_status()
    results = search_resp.json()["results"]
    
    print(f"[+] Found {len(results)} matching chunks:")
    for res in results:
        print(f"    - Score: {res['similarity_score']:.4f} | Chunk: {res['content']}")
```
