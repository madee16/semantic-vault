# SemanticVault: Technical Interview Preparation & Defense Guide

This guide is designed to help you confidently present, explain, and defend the architecture and engineering decisions of **SemanticVault** in senior AI Engineering, Backend Architecture, and Systems Design interviews.

---

## 1. The Elevator Pitch (30-Second Summary)

> *"SemanticVault is a production-grade Semantic Search as a Service platform built in Python and FastAPI on PostgreSQL with pgvector. Rather than just wrapping an external AI API, it implements full multi-tenant developer infrastructure: cryptographic API keys with salted SHA-256 hashing, constant-time verification, rolling rate limits, and atomic SQL quota enforcement.*
>
> *On the retrieval side, it strictly enforces vector index compatibility to prevent model/dimension drift, implements sliding-window chunking with word boundary preservation, and features a hybrid retrieval pipeline combining dense vector embeddings with sparse keyword search using Reciprocal Rank Fusion (RRF). The entire inference stack runs locally on quantized open models via Ollama, ensuring zero cloud data egress and sub-15ms retrieval latency."*

---

## 2. Core Technical Questions & Battle-Tested Answers

### Question 1: "Why did you implement explicit vector index compatibility tracking instead of just storing vectors in a table?"
**Answer:**
> *"In production vector search, mixing embeddings from different models—or even different versions of the same model—is mathematically invalid. A 768-dimensional vector produced by Gemma represents a completely different latent manifold than a 768-dimensional vector produced by BERT or a 384-dimensional vector from MiniLM. Cosine distance between mismatched embeddings produces meaningless noise.*
>
> *In SemanticVault, I created an `EmbeddingIndex` entity that explicitly tracks `model_name`, `model_version`, `dimension`, and `distance_metric`. Every document and chunk holds a foreign key to an active index. If an API client attempts to ingest or search with an incompatible vector dimension, the service rejects the request immediately at the Pydantic schema validation layer before querying the database. Furthermore, switching models requires executing a controlled re-indexing migration."*

---

### Question 2: "How did you prevent race conditions when enforcing developer monthly quotas under heavy concurrent traffic?"
**Answer:**
> *"In high-throughput APIs, standard Read-Modify-Write patterns (`SELECT requests_used ... if used < quota: used += 1 ... COMMIT`) are vulnerable to race conditions. If 20 concurrent requests hit the server when quota is 9,999 of 10,000, all 20 will read 9,999 and all 20 will proceed.*
>
> *To solve this, SemanticVault executes an atomic conditional SQL update:*
> ```sql
> UPDATE api_keys 
> SET requests_used = requests_used + 1 
> WHERE id = :key_id AND requests_used < monthly_quota;
> ```
> *The database engine evaluates the condition under row-level lock semantics. If the key has reached its quota, the `WHERE` clause fails to match, zero rows are updated, and the endpoint returns an HTTP 429 Too Many Requests. In our adversarial attack test suite, we fired 20 simultaneous async requests against a key with quota = 5; exactly 5 succeeded with 200 OK and exactly 15 were rejected with 429."*

---

### Question 3: "Why choose Reciprocal Rank Fusion (RRF) over linear score interpolation for your hybrid search?"
**Answer:**
> *"Dense vector similarity and sparse keyword scores operate on completely different mathematical scales and distributions. Cosine similarity produces scores bounded between $[0, 1]$ or $[-1, 1]$, whereas keyword matching algorithms like BM25 produce unbounded scores dependent on inverse document frequency and document length.*
>
> *Attempting linear interpolation ($\alpha \cdot S_{\text{vector}} + (1 - \alpha) \cdot S_{\text{bm25}}$) requires arbitrary min-max normalization that breaks whenever document lengths vary. RRF sidesteps score normalization entirely by relying strictly on rank position:*
> $$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
> *This provides scale invariance, robust noise dampening via smoothing constant $k=60$, and consistent retrieval quality regardless of vocabulary size."*

---

### Question 4: "Why did you build your own API key system rather than using JWTs or third-party auth like Auth0?"
**Answer:**
> *"JWTs are designed for stateless user sessions where short lifetimes (e.g. 15 minutes) and client-side storage are acceptable. For a developer-facing API product, clients need long-lived credentials (`sv_live_...`) embedded in environment variables or backend servers.*
>
> *JWTs have a major fatal flaw for developer APIs: you cannot revoke a JWT instantly without consulting a centralized revocation blacklist, which defeats the stateless premise of JWTs. Furthermore, since our API must check rate limits and monthly request quotas on every request anyway, database access is already required.*
>
> *By using salted SHA-256 hashed API keys with prefix indexing (`sv_live_` + 8 hex chars unhashed for $O(1)$ lookups) and constant-time verification (`hmac.compare_digest`), we achieve instant revocation, zero timing attack vulnerability, and full quota management with sub-millisecond lookup latency."*

---

### Question 5: "What are the trade-offs of using HNSW vs. IVFFlat in pgvector?"
**Answer:**
> *"IVFFlat partitions the vector space using k-means Voronoi cells. It has very low memory usage and builds quickly, but has two significant downsides: its recall drops if queries lie on Voronoi boundaries, and it requires periodic retraining as new vectors drift the cluster centroids.*
>
> *HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where upper layers have long-range skip connections and lower layers have dense local links. It provides:
> 1. Exceptional recall ($>98\%$) even under high dimensionalities ($d=768$).
> 2. Sub-10ms query latency.
> 3. Incremental updates: vectors can be inserted continuously without rebuilding or retraining the graph.*
>
> *The trade-off is higher memory consumption and slower index build times. In SemanticVault, we configure $M=16$ and $ef\_construction=64$, which provides the optimal balance of recall accuracy and VRAM/RAM efficiency."*

---

## 3. System Design Walkthrough: Scaling SemanticVault to 10M Documents

When asked: *"How would you scale SemanticVault from thousands of documents to 10,000,000 documents with 1,000 queries per second?"*

### Architecture Evolution:
1. **Asynchronous Ingestion with Task Queues:**
   - Move document parsing, chunking, and embedding generation out of the HTTP request lifecycle into background workers using Celery / ARQ backed by Redis or RabbitMQ.
   - The `/documents` endpoint immediately returns `HTTP 202 Accepted` with a job ID, allowing the client to poll or receive webhooks when indexing finishes.

2. **Distributed Vector Indexing & Read Replicas:**
   - Scale PostgreSQL with streaming read replicas behind a connection pooler like **PgBouncer**. Search queries (`POST /search`) are directed exclusively to read replicas with warm HNSW index buffers in RAM.
   - For 10M vectors of dimension 768 ($768 \times 4\text{ bytes} \approx 3\text{ KB}$ per vector), the raw embeddings occupy ~30 GB of RAM. Adding HNSW graph overhead (~1.5x), an enterprise database instance with 64 GB of RAM keeps the entire vector index in memory for sub-10ms lookups.

3. **Distributed Rate Limiting & Token Buckets:**
   - Replace the in-memory sliding-window rate limiter with a Redis cluster executing atomic Lua scripts to track request timestamps across multiple stateless FastAPI container replicas.

4. **Dedicated GPU Embedding Workers:**
   - Decouple Ollama / PyTorch embedding generation into a dedicated auto-scaling worker pool running on NVIDIA A10G / L4 GPUs with batching (e.g. batch size = 64 chunks per forward pass), maximizing GPU tensor core utilization.

---

## 4. Live Coding & Debugging War Stories

### Story: Solving the In-Memory SQLAlchemy Session Cache Concurrency Bug
> *"During implementation of the concurrent quota attack suite, we fired 20 simultaneous requests against an API key with a quota limit of 5. Initially, all 20 requests succeeded because in an asynchronous framework, multiple concurrent coroutines reusing an active SQLAlchemy session were reading cached entities from the identity map.*
>
> *Rather than adding Python-level mutex locks which only work inside a single process, I redesigned the quota decrement to execute an atomic conditional SQL UPDATE statement:*
> ```sql
> UPDATE api_keys SET requests_used = requests_used + 1 WHERE id = :id AND requests_used < monthly_quota
> ```
> *By checking `result.rowcount > 0`, the enforcement became purely atomic at the PostgreSQL engine level. It is completely resilient across multiple async tasks, worker processes, and container replicas, demonstrating why database-level constraints should always supersede application-level assertions for concurrency safety."*

---

### Question 6: "How did you prevent tenant takeover and privilege escalation in API key creation?"
**Answer:**
> *"In multi-tenant APIs, registration endpoints are a primary attack vector. If `POST /keys` blindly upserts or fetches a developer by email, any unauthenticated attacker could pass a victim's email address and generate a backdoor API key granting full access to that victim's documents.
>
> In SemanticVault, we enforce strict authorization boundaries:
> 1. If an email does not exist, public self-registration is allowed, but high-privilege scopes like `admin:all` are mathematically blocked from self-assignment.
> 2. If the email already exists, the endpoint requires authentication (`X-API-Key`) with `admin:keys` scope belonging to that exact developer account (`developer_id == current_key.developer_id`). An unauthenticated request triggers HTTP 409 Conflict.
> 3. To prevent child key privilege escalation, an issued key's scopes are strictly validated as a subset of the authorizing key's scopes. A key with `documents:read` cannot issue a child key with `admin:keys` or `documents:write`."*

---

### Question 7: "Why is executing vector similarity inside PostgreSQL with pgvector better than pulling embeddings into Python?"
**Answer:**
> *"Naive vector implementations load all document chunks into application memory and compute cosine similarity using NumPy or Python loops. While this works for 20 documents in a unit test, it collapses in production:
> 1. **Network I/O:** Transferring thousands of 768-dimensional float vectors (3 KB each) over database connection pools wastes bandwidth and saturates memory buffers.
> 2. **Compute Bottlenecks:** Python's Global Interpreter Lock (GIL) and serialization overhead degrade retrieval latency from milliseconds to multiple seconds.
> 3. **Index Bypassing:** Python-side calculations cannot leverage HNSW or IVFFlat vector indexes.
>
> In SemanticVault, we execute vector search directly inside PostgreSQL using the `<=>` cosine distance operator and HNSW indexing:
> ```sql
> SELECT dc.id, dc.content, (1.0 - (dc.embedding <=> :query_vec)) AS similarity_score
> FROM document_chunks dc
> JOIN documents d ON dc.document_id = d.id
> WHERE d.developer_id = :developer_id
> ORDER BY dc.embedding <=> :query_vec ASC
> LIMIT :top_k;
> ```
> PostgreSQL's C-level HNSW implementation traverses graph layers in sub-10ms latency and returns only the top $K$ matching rows to the application server."*

---

### Question 8: "How did you eliminate quadratic $O(N^2)$ bottlenecks in your hybrid search?"
**Answer:**
> *"In early versions of hybrid search with Reciprocal Rank Fusion, merging dense vector candidates and sparse lexical candidates used a linear search inside an outer loop:
> `v_sim = next(s for s, c, _ in vector_scores if c.id == chunk.id)`
> For $N$ candidates, this performed an $O(N^2)$ quadratic scan.
>
> We refactored RRF to use a hash-map join:
> 1. Dense and lexical candidates are ranked in $O(K \log K)$ and indexed into hash maps (`chunk.id -> rank_info`).
> 2. The union of chunk IDs is evaluated via $O(1)$ dictionary lookups to compute:
>    $$\text{RRF}(d) = \frac{\alpha}{60 + r_{\text{vector}}(d)} + \frac{1 - \alpha}{60 + r_{\text{keyword}}(d)}$$
> 3. The merged candidate list (at most $2K$ items) is sorted in $O(K \log K)$.
> This reduces overall complexity from $O(N^2)$ to $O(K \log K)$, where $K$ is bounded by the query limit (e.g. 50)."*

---

### Question 9: "How does your monthly quota system handle calendar month rollovers without background workers or Redis?"
**Answer:**
> *"Instead of a static counter or requiring an external cron worker to reset quotas at midnight on the first of every month, SemanticVault tracks a `current_period` column in `YYYY-MM` format on each API key.
>
> When `record_atomic_usage` executes, it evaluates the active calendar month. If the key's `current_period` is older than the current month, an atomic SQL conditional update executes:
> ```sql
> UPDATE api_keys 
> SET requests_used = 1, current_period = :current_period 
> WHERE id = :key_id AND current_period != :current_period AND is_active = true;
> ```
> If that matches, the first request of the month atomically resets the counter. Subsequent requests in the same month atomically increment `requests_used` where `requests_used < monthly_quota`.
> This guarantees self-healing, zero-maintenance calendar billing alignment under full database ACID guarantees without needing Celery Beat, Cron, or Redis."*

