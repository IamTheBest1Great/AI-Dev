# Phase 0 — Complete Code Step by Step

We will build Phase 0 in the exact order below. **Run and test each step before moving to the next.**

---

# Step 1 — Create the project

Create this manually:

```text
travel-agent/
```

Open it in VS Code.

---

# Step 2 — Create and activate virtual environment

From the project root:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

You should see:

```text
(venv) PS C:\...\travel-agent>
```

---

# Step 3 — Install dependencies

```powershell
pip install fastapi "uvicorn[standard]" sqlalchemy psycopg pydantic-settings
```

For now, these are enough.

---

# Step 4 — Create the first structure

Create:

```text
travel-agent/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
└── requirements.txt
```

The `__init__.py` file can be empty.

## `app/__init__.py`

```python
```

## `app/main.py`

```python
from fastapi import FastAPI

app = FastAPI(
    title="Travel Agent API",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "Travel Agent API is running"
    }
```

Run:

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Expected:

```json
{
  "message": "Travel Agent API is running"
}
```

Also check FastAPI's automatic docs:

```text
http://127.0.0.1:8000/docs
```

---

# Step 5 — Create the Health Route

Now create:

```text
travel-agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   └── api/
│       ├── __init__.py
│       │
│       └── routes/
│           ├── __init__.py
│           └── health.py
```

## `app/api/__init__.py`

```python
```

## `app/api/routes/__init__.py`

```python
```

## `app/api/routes/health.py`

```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("")
def health_check():
    return {
        "status": "healthy"
    }
```

Now update `app/main.py`:

```python
from fastapi import FastAPI

from app.api.routes.health import router as health_router


app = FastAPI(
    title="Travel Agent API",
    version="0.1.0"
)


app.include_router(health_router)


@app.get("/")
def root():
    return {
        "message": "Travel Agent API is running"
    }
```

Your flow is now:

```text
User
 │
 │ GET /health
 ▼
main.py
 │
 │ include_router()
 ▼
health.py
 │
 ▼
health_check()
 │
 ▼
{
  "status": "healthy"
}
```

Test:

```text
http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

# Step 6 — Create `.gitignore`

Create:

```text
.gitignore
```

Code:

```gitignore
# Virtual environment
venv/

# Environment variables
.env

# Python cache
__pycache__/
*.py[cod]

# Test and coverage
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
```

---

# Step 7 — Create `.env`

Create:

```text
.env
```

For now:

```env
APP_NAME=Travel Agent API
DEBUG=true

DATABASE_URL=postgresql+psycopg://travel_user:travel_password@localhost:5432/travel_db

SECRET_KEY=change-this-to-a-random-secret-key
```

Later, we will generate a proper secret key.

---

# Step 8 — Create `.env.example`

This file is safe to commit to Git.

```text
.env.example
```

```env
APP_NAME=Travel Agent API
DEBUG=true

DATABASE_URL=postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/YOUR_DATABASE

SECRET_KEY=your-secret-key
```

---

# Step 9 — Create centralized configuration

Add:

```text
app/
│
└── core/
    ├── __init__.py
    └── config.py
```

## `app/core/__init__.py`

```python
```

## `app/core/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Travel Agent API"
    DEBUG: bool = True

    DATABASE_URL: str

    SECRET_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
```

### Flow

```text
.env
  │
  ▼
config.py
  │
  ▼
Settings()
  │
  ▼
settings
  │
  ├── settings.DATABASE_URL
  ├── settings.SECRET_KEY
  └── settings.DEBUG
```

---

# Step 10 — Test configuration

Temporarily update:

## `app/main.py`

```python
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0"
)


app.include_router(health_router)


@app.get("/")
def root():
    return {
        "message": f"{settings.APP_NAME} is running",
        "debug": settings.DEBUG
    }
```

Restart:

```powershell
uvicorn app.main:app --reload
```

Expected:

```json
{
  "message": "Travel Agent API is running",
  "debug": true
}
```

**Do not return `SECRET_KEY` or `DATABASE_URL` from an API.**

---

# Step 11 — Create PostgreSQL Docker setup

Create:

```text
docker-compose.yml
```

Code:

```yaml
services:
  postgres:
    image: postgres:16

    container_name: travel-agent-postgres

    environment:
      POSTGRES_USER: travel_user
      POSTGRES_PASSWORD: travel_password
      POSTGRES_DB: travel_db

    ports:
      - "5432:5432"

    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Your database architecture:

```text
FastAPI Application
       │
       │ localhost:5432
       ▼
┌──────────────────────┐
│ Docker Container     │
│                      │
│ PostgreSQL 16        │
│                      │
│ travel_db            │
└──────────────────────┘
```

Start it:

```powershell
docker compose up -d
```

Check:

```powershell
docker ps
```

You should see:

```text
travel-agent-postgres
```

Check logs if necessary:

```powershell
docker logs travel-agent-postgres
```

---

# Step 12 — Create the database package

Create:

```text
app/
│
└── database/
    ├── __init__.py
    ├── base.py
    └── session.py
```

## `app/database/__init__.py`

```python
```

---

# Step 13 — Create `base.py`

## `app/database/base.py`

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

This is the base class for future models.

Later:

```text
Base
 │
 ├── User
 │
 ├── Trip
 │
 ├── Booking
 │
 └── UserPreference
```

Example later:

```python
class Trip(Base):
    ...
```

---

# Step 14 — Create the database engine and session

## `app/database/session.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings


engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
```

The flow:

```text
.env
 │
 ▼
DATABASE_URL
 │
 ▼
config.py
 │
 ▼
settings.DATABASE_URL
 │
 ▼
session.py
 │
 ▼
create_engine()
 │
 ▼
PostgreSQL
```

---

# Step 15 — Add database health checking

Now update:

## `app/api/routes/health.py`

```python
from fastapi import APIRouter
from sqlalchemy import text

from app.database.session import engine


router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("")
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database": "connected"
        }

    except Exception:
        return {
            "status": "unhealthy",
            "database": "disconnected"
        }
```

Now the actual flow is:

```text
User
 │
 │ GET /health
 ▼
health.py
 │
 ▼
engine.connect()
 │
 ▼
SQLAlchemy
 │
 ▼
psycopg
 │
 ▼
PostgreSQL
 │
 ▼
SELECT 1
 │
 ▼
Database Connected?
 │
 ├── YES → healthy
 │
 └── NO  → unhealthy
```

Test:

```text
http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

---

# Step 16 — Add basic tests

Create:

```text
tests/
├── __init__.py
└── test_health.py
```

Install:

```powershell
pip install pytest httpx
```

## `tests/test_health.py`

```python
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["database"] == "connected"
```

Run:

```powershell
pytest
```

Expected:

```text
1 passed
```

---

# Step 17 — Generate `requirements.txt`

Now that Phase 0 dependencies are installed:

```powershell
pip freeze > requirements.txt
```

Your `requirements.txt` will contain more than just the packages you manually installed because it includes their dependencies too.

That is normal.

---

# Final Phase 0 folder structure

At the end, your project should look like this:

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
│   │   │
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
│       ├── base.py
│       └── session.py
│
├── tests/
│   ├── __init__.py
│   └── test_health.py
│
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
└── requirements.txt
```

# Complete Phase 0 flowchart

```text
                         USER
                           │
                           │ HTTP Request
                           ▼
                ┌─────────────────────┐
                │       FastAPI       │
                │      main.py        │
                └──────────┬──────────┘
                           │
                           │ include_router()
                           ▼
                ┌─────────────────────┐
                │    Health Router    │
                │     health.py       │
                └──────────┬──────────┘
                           │
                           │ GET /health
                           ▼
                ┌─────────────────────┐
                │  SQLAlchemy Engine  │
                │     session.py      │
                └──────────┬──────────┘
                           │
                           │ DATABASE_URL
                           ▼
                ┌─────────────────────┐
                │      config.py      │
                │      Settings       │
                └──────────┬──────────┘
                           │
                           │ reads
                           ▼
                         .env
                           │
                           │
                           ▼
                ┌─────────────────────┐
                │    PostgreSQL 16    │
                │       Docker        │
                └──────────┬──────────┘
                           │
                           │ SELECT 1
                           ▼
                     DB CONNECTED
                           │
                           ▼
                {
                  "status": "healthy",
                  "database": "connected"
                }
```

# Phase 0 checklist

```text
□ Virtual environment activated
□ FastAPI server runs
□ GET / works
□ GET /health works
□ Swagger docs work at /docs
□ .env configuration works
□ PostgreSQL Docker container runs
□ PostgreSQL volume persists data
□ SQLAlchemy connects successfully
□ GET /health confirms database connection
□ pytest passes
□ .env is ignored by Git
□ requirements.txt generated
```

Once every item passes, **Phase 0 is complete** and the next step is **Phase 1: Trip API**, where we will build the first complete vertical slice:

```text
Request
   ↓
Pydantic Schema
   ↓
Route
   ↓
Service
   ↓
SQLAlchemy Model
   ↓
PostgreSQL
   ↓
Response Schema
```
