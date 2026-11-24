# SemanticVault: Installation, Environment & Setup Guide

This guide covers everything required to deploy, run, test, and troubleshoot SemanticVault in local development or production containerized environments.

---

## 1. Prerequisites

Before installing SemanticVault, ensure your host development environment has:
- **Operating System:** Windows 10/11, macOS, or Ubuntu 22.04+ (WSL2 recommended on Windows)
- **Python:** Version 3.12+ (Verify with `python --version`)
- **Docker & Docker Compose:** Docker Desktop or Docker Engine 24+
- **Ollama:** Installed and running locally (`http://localhost:11434`)
- **Hardware Recommendations:**
  - 8 GB+ RAM (16 GB recommended)
  - Dedicated GPU optional (RTX 3050/4050 Laptop GPU with 6 GB VRAM accelerates Ollama embeddings, but CPU inference runs seamlessly as well)

---

## 2. Local Environment Configuration

### 2.1 Clone Repository & Prepare `.env`
Navigate to the project root and create your local environment file:

```bash
cd AI-Engineering-Portfolio/01-SemanticVault
cp .env.example .env
```

### 2.2 Configure Environment Variables
Inspect `.env` and adjust the variables to match your deployment:

```dotenv
# Environment Mode
ENVIRONMENT=development
DEBUG=false

# Database Configuration (PostgreSQL 16 with pgvector)
DATABASE_URL=postgresql+asyncpg://postgres:postgrespassword@localhost:5432/semanticvault
DB_ECHO=false
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# API Security (Salt must be >= 32 characters)
API_KEY_SALT=semanticvault-production-secure-salt-32-chars-long
DEFAULT_MONTHLY_QUOTA=10000
RATE_LIMIT_PER_MINUTE=60

# Local AI / Ollama Inference Engine
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=60.0

# Embedding Constraints (Strict Dimension Validation)
EMBEDDING_MODEL=embeddinggemma
EMBEDDING_MODEL_VERSION=v1
EMBEDDING_DIM=768
GENERATIVE_MODEL=llama3.2:1b
```

---

## 3. Ollama Local Model Setup

SemanticVault runs completely sovereign AI inference locally via Ollama.

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```
2. Pull the embedding and generative models:
   ```bash
   ollama pull embeddinggemma
   ollama pull llama3.2:1b
   ```
3. Verify models are loaded:
   ```bash
   ollama list
   ```
   Output should list:
   - `embeddinggemma:latest` (~621 MB)
   - `llama3.2:1b:latest` (~1.3 GB)

---

## 4. Launching the Vector Database (Docker)

SemanticVault leverages the official `pgvector/pgvector:pg16` Docker image.

### 4.1 Start the PostgreSQL Container
```bash
docker compose up -d db
```

### 4.2 Verify Container Health
```bash
docker ps --filter "name=semanticvault-db"
```
The status should indicate `Up (healthy)` and port `0.0.0.0:5432->5432/tcp` open.

---

## 5. Local Python Application Setup

### 5.1 Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install production and development dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2 Start the FastAPI Application
Run the backend with hot-reload enabled:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 6. Running Tests & Security Verification

SemanticVault includes an extensive suite of 51 unit, integration, and security tests.

### 6.1 Run the Full Test Suite
```bash
pytest -v
```

### 6.2 Run the Adversarial Security Attack Suite
Verifies resistance against tampered tokens, expired keys, privilege escalation, and race-condition quota exhaustion:
```bash
pytest tests/integration/test_security_attacks.py -v
```

### 6.3 Run Live pgvector Integration Tests
Verifies real HNSW indexing, cosine distance (`<=>`), and nearest-neighbor search against PostgreSQL:
```bash
pytest tests/integration/test_pgvector_live.py -v
```

---

## 7. Troubleshooting Common Issues

### Issue 1: `pg_isready` Connection Refused (Port 5432)
- **Cause:** Another PostgreSQL service is already running on the host machine.
- **Solution:** Stop host PostgreSQL (`net stop postgresql-x64-16` on Windows or `systemctl stop postgresql` on Linux), or change host port mapping in `docker-compose.yml` to `"5433:5432"` and update `DATABASE_URL` in `.env`.

### Issue 2: Ollama Connection Error (`HTTP 502 / ConnectionRefused`)
- **Cause:** The Ollama service is not running or blocked by a local firewall.
- **Solution:** Verify `curl http://localhost:11434/api/tags` returns a 200 JSON payload. Ensure `OLLAMA_BASE_URL` in `.env` matches your Ollama instance address.

### Issue 3: Incompatible Embedding Dimensions (`HTTP 400 Bad Request`)
- **Cause:** Attempting to query an index created with `embeddinggemma` (768-dim) using a different model like `all-MiniLM-L6-v2` (384-dim).
- **Solution:** Check the `embedding_indexes` table. SemanticVault strictly forbids mixed-dimension vector spaces. Either re-index the documents or initialize a new index for the new model.
