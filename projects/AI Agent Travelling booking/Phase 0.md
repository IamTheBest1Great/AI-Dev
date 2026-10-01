# Phase 0 — Project Setup

The goal of Phase 0 is simple:

```text
┌──────────────────────────────────────────────┐
│              PHASE 0 COMPLETE               │
│                                              │
│  FastAPI Running                             │
│  PostgreSQL Running in Docker                │
│  FastAPI ↔ PostgreSQL Connection Working     │
│  Environment Variables Working               │
│  Health Check Working                        │
└──────────────────────────────────────────────┘
```

## 0.1 What we build in this phase

Start with a **small structure**, not the full agent architecture yet.

```text
travel-agent/
│
├── app/
│   │
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── health.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   └── database/
│       ├── __init__.py
│       ├── session.py
│       └── base.py
│
├── tests/
│   └── __init__.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── docker-compose.yml
```

We are **not creating these yet**:

```text
❌ agents/
❌ graph/
❌ tools/
❌ memory/
❌ services/
❌ models/
❌ schemas/
❌ migrations/
```

They will be added when Phase 1 and later phases actually need them.

---

# 0.2 Build sequence

Follow this exact order:

```text
STEP 1
Create project folder
        ↓
STEP 2
Create Python virtual environment
        ↓
STEP 3
Install FastAPI dependencies
        ↓
STEP 4
Create FastAPI application
        ↓
STEP 5
Run health endpoint
        ↓
STEP 6
Create PostgreSQL Docker container
        ↓
STEP 7
Add environment configuration
        ↓
STEP 8
Configure SQLAlchemy
        ↓
STEP 9
Test database connection
        ↓
STEP 10
Phase 0 complete
```

---

# 0.3 Step 1 — Create the project

```text
travel-agent/
```

Inside:

```text
app/
tests/
```

At this point:

```text
travel-agent/
├── app/
└── tests/
```

---

# 0.4 Step 2 — Create a virtual environment

From the project root:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Your terminal should look roughly like:

```text
(venv) PS C:\...\travel-agent>
```

---

# 0.5 Step 3 — Install the initial dependencies

For Phase 0, keep dependencies minimal.

```bash
pip install fastapi "uvicorn[standard]" sqlalchemy psycopg pydantic-settings
```

Then:

```bash
pip freeze > requirements.txt
```

### Why each dependency?

| Package | Purpose |
|---|---|
| `fastapi` | Build the backend API |
| `uvicorn` | Run the FastAPI server |
| `sqlalchemy` | Database ORM |
| `psycopg` | PostgreSQL driver |
| `pydantic-settings` | Read environment variables |

Do **not** install LangGraph, LangChain, LLM SDKs, memory libraries, or agent frameworks yet.

---

# 0.6 Step 4 — Create `main.py`

File:

```text
app/main.py
```

Its responsibility:

```text
Start FastAPI
     ↓
Register routers
     ↓
Application entry point
```

The first architecture looks like:

```text
Browser / Postman
        │
        ▼
   FastAPI App
   app/main.py
        │
        ▼
      Router
```

---

# 0.7 Step 5 — Create the health feature

Create:

```text
app/api/routes/health.py
```

This should expose:

```text
GET /health
```

Flow:

```text
User
  │
  ▼
GET /health
  │
  ▼
health.py
  │
  ▼
{
  "status": "healthy"
}
```

This proves:

```text
✓ FastAPI is running
✓ Routing works
✓ Application structure works
```

Test it before doing anything else.

---

# 0.8 Step 6 — Add PostgreSQL with Docker

Now add:

```text
docker-compose.yml
```

Architecture:

```text
┌──────────────────┐
│   FastAPI        │
│                  │
│ localhost:8000   │
└────────┬─────────┘
         │
         │ PostgreSQL Connection
         │
         ▼
┌──────────────────┐
│ Docker           │
│                  │
│ PostgreSQL       │
│ localhost:5432   │
└──────────────────┘
```

Start the database:

```bash
docker compose up -d
```

Verify:

```bash
docker ps
```

At this point:

```text
✓ PostgreSQL container running
✓ Port 5432 exposed
```

---

# 0.9 Step 7 — Add `.env`

Create:

```text
.env
```

This contains configuration such as:

```text
DATABASE_URL
SECRET_KEY
DEBUG
```

Flow:

```text
.env
  │
  ▼
core/config.py
  │
  ▼
Settings Object
  │
  ├── Database URL
  ├── Secret Key
  └── Debug
```

### Important rule

Your application should **not directly read**:

```python
os.getenv("DATABASE_URL")
```

all over the project.

Instead:

```text
.env
   ↓
config.py
   ↓
settings
   ↓
Entire application
```

This gives you one centralized configuration source.

---

# 0.10 Step 8 — Configure the database

Create:

```text
app/database/
├── session.py
└── base.py
```

## `session.py`

Responsibility:

```text
DATABASE_URL
      │
      ▼
SQLAlchemy Engine
      │
      ▼
Session Factory
      │
      ▼
Database Session
```

Later:

```text
API Request
     │
     ▼
Route
     │
     ▼
get_db()
     │
     ▼
Database Session
     │
     ▼
PostgreSQL
```

## `base.py`

This will become the foundation for all future models.

Later:

```text
Base
 │
 ├── User
 ├── Trip
 ├── Booking
 └── UserPreference
```

Every database model will inherit from this base.

---

# 0.11 Step 9 — Test the database connection

Before moving forward, test:

```text
FastAPI
   │
   ▼
SQLAlchemy
   │
   ▼
psycopg
   │
   ▼
PostgreSQL Docker Container
```

Your health system can eventually become:

```text
GET /health
```

Response:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

The database must actually be queried—not just assumed to be running.

---

# Phase 0 complete architecture

```text
                        USER
                          │
                          ▼
                 ┌────────────────┐
                 │    FastAPI     │
                 │    main.py     │
                 └────────┬───────┘
                          │
                          ▼
                 ┌────────────────┐
                 │     Router     │
                 │   health.py    │
                 └────────┬───────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ Database Layer │
                 │   session.py   │
                 └────────┬───────┘
                          │
                          ▼
                 ┌────────────────┐
                 │   PostgreSQL   │
                 │    Docker      │
                 └────────────────┘
```

# Phase 0 definition of done

Do **not** move to Phase 1 until all of these work:

```text
□ Project folder created
□ Virtual environment created
□ FastAPI installed
□ Uvicorn starts successfully
□ GET /health works
□ PostgreSQL Docker container runs
□ PostgreSQL data persists via a Docker volume
□ .env is working
□ SQLAlchemy engine is created
□ FastAPI successfully connects to PostgreSQL
□ Database health check works
□ requirements.txt exists
□ .gitignore excludes .env and venv
```

## After Phase 0

Your next vertical slice will be:

```text
PHASE 1

POST /travel/plan
        │
        ▼
Validate Request
        │
        ▼
Trip Schema
        │
        ▼
Trip Service
        │
        ▼
Save to PostgreSQL
        │
        ▼
Return Created Trip
```

That will be the first real feature of the Travel Agent.
