# SemanticVault: System Architecture & Technical Design

## 1. Executive Overview

**SemanticVault** is a developer-facing Semantic Search as a Service platform engineered in Python and FastAPI. It enables organizations and client applications to upload unstructured text, PDF, and Markdown documents, extract and chunk their content, generate high-dimensional vector representations using local embedding models, and retrieve semantically relevant information with sub-second latency.

Unlike superficial vector search wrappers, SemanticVault incorporates enterprise backend guarantees:
- **Strict Vector Index Compatibility:** Enforces model name, version, and dimension boundaries at the database level. Vectors from distinct models or differing dimensions are mathematically prohibited from sharing an index.
- **Cryptographic API Key Infrastructure:** Issues developer API keys hashed via salted SHA-256, verified in constant time, bounded by granular permission scopes (`documents:read`, `documents:write`, `search:read`, `admin:keys`, `admin:all`), rolling rate limits, and atomic SQL monthly quotas.
- **Hybrid Retrieval:** Blends dense vector cosine similarity with sparse lexical search using Reciprocal Rank Fusion (RRF).
- **Local AI Sovereignty:** Powered by Ollama (`embeddinggemma:latest` for 768-dimensional embeddings and `llama3.2:1b` for query expansion and document enrichment), eliminating external cloud dependencies and third-party data egress.

---

## 2. High-Level Architecture Diagram

```mermaid
graph TD
    Client["Client / External Developer Application"]
    
    subgraph "API Gateway Layer (FastAPI)"
        Auth["Authentication & Scope Dependency (require_api_key)"]
        RateLimiter["Sliding-Window Rate Limiter (Token Bucket)"]
        QuotaEnforcer["Atomic SQL Quota Guard"]
        Router["API Router (/keys, /documents, /search)"]
    end

    subgraph "Local AI Inference Layer (Ollama / PyTorch)"
        EmbedService["OllamaProvider (/api/embed - embeddinggemma)"]
        LLMService["OllamaProvider (/api/chat - llama3.2:1b)"]
        STFallback["SentenceTransformers Fallback (all-MiniLM-L6-v2)"]
    end

    subgraph "Core Business Logic Services"
        Parser["DocumentParser (PDF, Markdown, TXT)"]
        Chunker["ChunkingService (Sliding Window & Overlap)"]
        DocService["DocumentService (Ingestion & Deduplication)"]
        SearchService["SearchService (Dense Vector, Cosine & RRF Hybrid)"]
        KeyService["KeyService (Crypto Generation, Scopes, Revocation)"]
    end

    subgraph "Persistence Layer (PostgreSQL 16 + pgvector)"
        DB_Keys["api_keys / developers"]
        DB_Index["embedding_indexes (Model, Version, Dim Constraint)"]
        DB_Docs["documents (Content Hash, Metadata JSON)"]
        DB_Chunks["document_chunks (HNSW Vector Index, vector_cosine_ops)"]
    end

    Client -->|X-API-Key + HTTP Payload| Auth
    Auth --> KeyService
    KeyService --> DB_Keys
    Auth --> RateLimiter
    Auth --> QuotaEnforcer
    QuotaEnforcer --> DB_Keys
    Auth --> Router

    Router -->|POST /documents| DocService
    DocService --> Parser
    DocService --> Chunker
    DocService --> EmbedService
    DocService --> DB_Docs
    DocService --> DB_Chunks

    Router -->|POST /search| SearchService
    SearchService -->|Optional Rewrite| LLMService
    SearchService --> EmbedService
    SearchService --> DB_Chunks
```

---

## 3. Document Ingestion Lifecycle

The ingestion pipeline handles raw document upload, format extraction, text normalization, sliding-window chunking, vector embedding generation, and atomic persistence.

```mermaid
sequenceDiagram
    autonumber
    actor Developer
    participant API as FastAPI Ingestion Endpoint
    participant Guard as Security & Quota Guard
    participant Parser as DocumentParser
    participant Chunker as ChunkingService
    participant AI as Local AI Engine (Ollama)
    participant DB as PostgreSQL + pgvector

    Developer->>API: POST /api/v1/documents (file, index_id, chunk_size)
    API->>Guard: Validate X-API-Key & scope (documents:write)
    Guard->>DB: Atomic increment requests_used WHERE requests_used < monthly_quota
    Guard-->>API: Authorized
    API->>Parser: Parse file bytes (PyMuPDF / Regex Normalizer)
    Parser-->>API: Clean raw text + SHA-256 content hash
    API->>DB: Check for duplicate content_hash
    alt Duplicate Document
        API-->>Developer: HTTP 409 Conflict (Document already indexed)
    end
    API->>Chunker: Split text (chunk_size, chunk_overlap)
    Chunker-->>API: List of text chunks with token counts
    API->>AI: Generate dense embeddings for chunks (embeddinggemma)
    AI-->>API: List of 768-dimensional float vectors
    API->>API: Validate vector dimensions == Index Dimension (768)
    API->>DB: Atomic transaction: Insert Document & DocumentChunks
    DB-->>API: Commit successful
    API-->>Developer: HTTP 201 Created (Document ID, Chunk Count)
```

---

## 4. Search & Hybrid Retrieval Lifecycle

SemanticVault provides dense semantic vector search, metadata filtering, similarity score thresholds, optional query expansion/rewriting, and Reciprocal Rank Fusion (RRF) hybrid retrieval.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI Search Endpoint
    participant AI as Ollama Service
    participant Engine as SearchService
    participant PG as PostgreSQL (pgvector)

    Client->>API: POST /api/v1/search (query, top_k, hybrid, rewrite_query)
    API->>API: Verify API Key & scope (search:read)
    
    opt Query Expansion Enabled (rewrite_query=true)
        API->>AI: llama3.2:1b prompt (Generate search variations)
        AI-->>API: Rewritten query with synonyms and intent expansion
    end

    API->>AI: embeddinggemma embed(query)
    AI-->>API: 768-dimensional query vector

    alt Standard Semantic Search
        API->>PG: SELECT chunks ORDER BY embedding <=> query_vec LIMIT top_k
        PG-->>Engine: Nearest neighbor chunks + Cosine Distance
    else Hybrid Search (Dense Vector + Sparse Keyword RRF)
        par Dense Vector Retrieval
            API->>PG: Retrieve top 20 by cosine distance (<=>)
        and Sparse Lexical Retrieval
            API->>PG: Retrieve top 20 by ILIKE / Full-Text keyword matching
        end
        Engine->>Engine: Compute Reciprocal Rank Fusion (RRF) scores
    end

    Engine->>Engine: Filter results below similarity_threshold
    Engine-->>API: Formatted SearchResponse (chunks, scores, metadata)
    API-->>Client: HTTP 200 OK
```

---

## 5. Database Schema & Vector Indexing Design

### 5.1 Relational & Vector Entities

1. **`developers`**: Account entity managing organization identity and billing contact.
2. **`api_keys`**: Cryptographic credentials containing key prefix (for $O(1)$ indexed lookup), salted SHA-256 hash, assigned scopes, expiration timestamp, active status, and monthly usage metrics.
3. **`embedding_indexes`**: Enforces vector space invariance. Tracks model name (`embeddinggemma`), version (`v1`), dimension (`768`), and distance metric (`cosine`).
4. **`documents`**: Parent document entity with SHA-256 `content_hash` for deduplication, extraction metadata, and cascade deletion relationships.
5. **`document_chunks`**: Individual text fragments storing chunk index, token count, text content, and the high-dimensional vector `embedding` column.

### 5.2 HNSW Vector Indexing Architecture

SemanticVault utilizes Hierarchical Navigable Small World (HNSW) graphs in PostgreSQL:

```sql
CREATE INDEX idx_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

- **Distance Metric (`vector_cosine_ops`):** Measures normalized directional similarity:
  $$\text{Cosine Distance}(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- **$M = 16$:** Maximum bidirectional links per node in each graph layer. Balances recall accuracy against indexing memory footprint.
- **$ef\_construction = 64$:** Size of the dynamic candidate list during graph construction, guaranteeing high retrieval accuracy on complex semantic manifolds.

---

## 6. Security & Quota Architecture

### 6.1 Cryptographic Key Pipeline
1. **Generation:** 32 bytes of cryptographically secure random entropy (`secrets.token_hex(24)`), prefixed with `sv_live_`.
2. **Prefix Indexing:** The first 16 characters (`sv_live_` + 8 hex chars) are stored unhashed as `key_prefix` to enable $O(1)$ candidate retrieval in the database without scanning table rows.
3. **Salted Hashing:** Stored as $\text{HMAC-SHA256}(\text{key}, \text{salt})$ with a minimum 32-character system salt. Plaintext keys are never logged, cached, or persisted.
4. **Constant-Time Verification:** Key hashes are validated using `hmac.compare_digest` to prevent side-channel timing attacks.

### 6.2 Atomic Concurrency Protection
To prevent race condition attacks where multiple concurrent requests exhaust an API key quota simultaneously, SemanticVault executes atomic SQL updates:

```sql
UPDATE api_keys 
SET requests_used = requests_used + 1 
WHERE id = :key_id AND requests_used < monthly_quota;
```

This guarantees that database engine row-level locks enforce exact monthly quotas without application-level race vulnerabilities.
