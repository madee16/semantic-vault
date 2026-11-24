# SemanticVault: Architecture Decision Records (ADRs)

This document captures the key architectural decisions made during the design and engineering of SemanticVault, including the context, evaluated alternatives, selected decisions, and technical trade-offs.

---

## ADR-001: PostgreSQL + pgvector vs. Standalone Vector Databases

### Status
Accepted

### Context
Semantic search requires storing high-dimensional vectors and executing approximate nearest neighbor (ANN) distance calculations. Options considered:
1. **Dedicated Vector DBs:** Pinecone, Qdrant, Milvus, Weaviate.
2. **Relational Vector Extensions:** PostgreSQL with `pgvector`.

### Decision
Adopt **PostgreSQL 16 with pgvector**.

### Consequences & Rationale
- **Relational Integrity & ACID Transactions:** Documents, chunks, API keys, developer quotas, and embedding indexes share transactional boundaries. A document deletion cascades atomically to all its chunk vectors. Standalone vector DBs require a "dual-write" architecture with an external SQL DB, introducing eventual consistency anomalies and distributed transaction failures.
- **Cost & Operational Simplicity:** Eliminates the operational overhead and cloud subscription costs of third-party SaaS vector databases.
- **Performance:** pgvector’s HNSW indexing achieves sub-10ms retrieval on millions of vectors with $>98\%$ recall.
- **Trade-off:** Standalone vector databases can scale past 100M+ vectors with horizontal sharding more naturally. For SemanticVault’s target use case (up to several million documents per developer organization), pgvector provides the optimal balance of consistency, simplicity, and throughput.

---

## ADR-002: Ollama Local Inference vs. Cloud Embedding APIs

### Status
Accepted

### Context
The platform requires generating 768-dimensional dense vector embeddings for documents and search queries, as well as optional generative query expansion.
Options:
1. **Cloud APIs:** OpenAI (`text-embedding-3-small`), Cohere Embed, Google Vertex AI.
2. **Local Inference:** Ollama hosting quantized open weights (`embeddinggemma`, `llama3.2:1b`), with `SentenceTransformers` as an in-process PyTorch fallback.

### Decision
Adopt **Ollama for local sovereign AI inference**.

### Consequences & Rationale
- **Zero Third-Party Data Egress:** Confidential enterprise documents and proprietary customer data never leave the deployment boundary, fulfilling strict GDPR/HIPAA compliance requirements.
- **Cost Predictability:** Eliminates token-based billing. High-volume document indexing incurs zero marginal financial cost.
- **Offline / Air-Gapped Capabilities:** Can be deployed in air-gapped data centers.
- **Hardware Optimization:** `embeddinggemma:latest` runs in ~621 MB of VRAM on laptop GPUs (e.g. RTX 4050 6GB) or pure CPU, allowing rapid development on consumer workstations.
- **Trade-off:** Inference throughput is bounded by local GPU/CPU compute rather than elastic cloud concurrency.

---

## ADR-003: Salted SHA-256 API Key Hashing vs. Signed JWTs

### Status
Accepted

### Context
Client applications require programmatic API authentication.
Options:
1. **Signed JSON Web Tokens (JWTs):** Stateless tokens signed by a private key with an expiration claim.
2. **Hashed API Key Tokens (`sv_live_...`):** High-entropy random tokens hashed with a salt in the database.

### Decision
Adopt **Hashed API Keys with Salted SHA-256 and Key Prefix Indexing**.

### Consequences & Rationale
- **Instant Revocation:** JWTs cannot be revoked instantly without maintaining a distributed revocation blocklist (which defeats their stateless advantage). With hashed API keys, setting `is_active = False` or `revoked_at = NOW()` immediately invalidates the key on the next HTTP request.
- **Quota Tracking:** API products require tracking monthly usage per key. Because database access is already required to decrement quotas, stateless tokens offer zero performance advantage.
- **Developer Experience:** Long-lived static tokens (`sv_live_...`) are standard for developer APIs (e.g., Stripe, OpenAI, Anthropic), unlike user session JWTs that require complex refresh-token rotation loops.
- **Security:** Keys are generated via `secrets.token_hex(24)` (192 bits of cryptographic entropy), hashed with SHA-256 and a secret salt, and verified via constant-time comparison (`hmac.compare_digest`).

---

## ADR-004: In-Memory Sliding-Window Token Bucket vs. External Redis Rate Limiting

### Status
Accepted

### Context
To protect the server from denial-of-service (DDoS) and aggressive client flooding, requests must be rate-limited per API key.
Options:
1. **External Redis Cluster:** Centralized Redis instance tracking sliding-window timestamps with `ZADD` / `ZREMRANGEBYSCORE`.
2. **In-Memory Sliding-Window Rate Limiter:** Process-local sliding window tracking request timestamps using deque per API key.

### Decision
Implement an **In-Memory Sliding-Window Rate Limiter** with modular dependency injection.

### Consequences & Rationale
- **Zero Extra Infrastructure:** Developers can run SemanticVault with zero external dependencies beyond PostgreSQL.
- **Sub-Microsecond Latency:** Rate-limit checks execute purely in memory without network hop latency to a cache cluster.
- **Trade-off:** In a multi-replica deployment behind a load balancer, process-local memory is not synchronized across worker processes. For multi-replica production setups, the interface can be swapped for a Redis-backed backend.

---

## ADR-005: Atomic SQL Quota Updates vs. Read-Modify-Write Application Locks

### Status
Accepted

### Context
Each API key possesses a monthly quota (e.g. 10,000 requests). Under high concurrency, 20 parallel requests could read `requests_used = 9,999`, all see that quota is not exceeded, and all proceed, causing a race-condition quota bypass.

Options:
1. **Read-Modify-Write with In-Memory Lock:** Python `asyncio.Lock` guarding database read followed by commit.
2. **Atomic SQL Conditional Update:** Single SQL query decrementing quota atomically at the database engine level.

### Decision
Adopt **Atomic SQL Conditional Updates**:
```sql
UPDATE api_keys 
SET requests_used = requests_used + 1 
WHERE id = :key_id AND requests_used < monthly_quota;
```

### Consequences & Rationale
- **Eliminates Race Conditions:** The database engine's row-level lock evaluates the `WHERE` condition atomically. If the quota has been reached, `UPDATE` matches 0 rows and returns `rowcount = 0`.
- **Process & Connection Agnostic:** Works across multiple worker processes, Celery workers, and async event loops.
- **Eliminates SQLAlchemy Identity Map Caching:** Bypasses in-memory session caching issues that cause stale reads under high concurrency.

---

## ADR-006: Reciprocal Rank Fusion (RRF) for Hybrid Retrieval

### Status
Accepted

### Context
Pure dense vector search frequently fails on exact lexical keywords, version strings, and domain-specific acronyms. Combining dense and sparse results requires a score normalization strategy.

Options:
1. **Linear Weighted Sum Score Normalization:** Normalizing vector cosine score and BM25 score to $[0, 1]$ and adding them: $\text{score} = \alpha S_{\text{vector}} + (1 - \alpha) S_{\text{sparse}}$.
2. **Reciprocal Rank Fusion (RRF):** Combining based on rank position rather than raw score magnitudes:
   $$\text{RRF}(d) = \frac{\alpha}{60 + r_{\text{vector}}(d)} + \frac{1 - \alpha}{60 + r_{\text{keyword}}(d)}$$

### Decision
Adopt **Reciprocal Rank Fusion (RRF)**.

### Consequences & Rationale
- **Scale Invariance:** Vector cosine similarity and lexical frequency algorithms (TF-IDF, BM25, ILIKE) have completely different score distributions and decay curves. Normalizing them linearly often leads to one retrieval method skewing the combined result. RRF only depends on relative rank order, making it completely robust to incompatible score distributions.
- **Empirical Industry Standard:** Used in production search engines (Elasticsearch, Vespa, Azure AI Search) due to consistently higher Mean Reciprocal Rank (MRR) and NDCG@10.
