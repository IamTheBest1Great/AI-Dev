Yes. Let's do **Phase 2 — Database Layer** properly and in order.

We already have:

```text
Phase 0 → Development Setup       ✅
Phase 1 → FastAPI Foundation      ✅
Phase 2 → Database Layer          ← STARTING NOW
```

For this phase, we'll build **PostgreSQL + SQLAlchemy + Alembic + our initial models**.

---

# Phase 2 — Database Layer

## What we are building

```text
                    FastAPI
                       │
                       ▼
                 SQLAlchemy ORM
                       │
          ┌────────────┴────────────┐
          │                         │
       Models                   Session
          │                         │
          └────────────┬────────────┘
                       ▼
                   PostgreSQL
                       │
                       ▼
                  Alembic
                (migrations)
```

For the RAG project, our first database will contain:

```text
documents
    │
    └──────< document_chunks
```

Later we'll add:

```text
queries
answers
evaluation_results
```

---

# Step 1 — Database design

Before writing code, define what PostgreSQL is responsible for.

### PostgreSQL stores

```text
Document metadata
        +
Document chunks
        +
Queries
        +
Answers
        +
Evaluation data
```

### Qdrant stores

```text
Embeddings
        +
Vector metadata
```

So don't put everything into Qdrant.

The relationship will eventually look like:

```text
                 PostgreSQL
              ┌──────────────┐
              │  documents   │
              └──────┬───────┘
                     │
                     │ 1:N
                     ▼
              ┌──────────────┐
              │    chunks    │
              └──────┬───────┘
                     │
                  chunk_id
                     │
                     ▼
              ┌──────────────┐
              │    Qdrant    │
              │              │
              │   vector     │
              │   payload    │
              └──────────────┘
```

---

# Step 2 — Create the Document model

Create:

```text
app/models/document.py
```

```python
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    file_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="uploaded",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
```

---

# Step 3 — Create DocumentChunk model

Create:

```text
app/models/document_chunk.py
```

```python
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    document = relationship(
        "Document",
        back_populates="chunks",
    )
```

---

# Step 4 — Update models `__init__.py`

Change:

```text
app/models/__init__.py
```

to:

```python
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

__all__ = [
    "Document",
    "DocumentChunk",
]
```

This gives us:

```text
Document
   │
   │ 1:N
   ▼
DocumentChunk
```

---

# Step 5 — Make Alembic aware of the models

Open:

```text
alembic/env.py
```

Find:

```python
target_metadata = None
```

Replace it with:

```python
from app.db.database import Base
from app.models import Document, DocumentChunk

target_metadata = Base.metadata
```

The imports are important.

They ensure SQLAlchemy knows about:

```text
Document
DocumentChunk
```

when Alembic generates migrations.

---

# Step 6 — Configure Alembic database URL

Because we're already using `.env`, don't duplicate the database URL manually in `alembic.ini`.

Open:

```text
alembic/env.py
```

Add:

```python
from app.core.config import settings
```

Then find the configuration section and set:

```python
config.set_main_option(
    "sqlalchemy.url",
    settings.database_url,
)
```

A clean top section should look approximately like:

```python
from logging.config import fileConfig

from alembic import context

from app.core.config import settings
from app.db.database import Base
from app.models import Document, DocumentChunk

config = context.config

config.set_main_option(
    "sqlalchemy.url",
    settings.database_url,
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
```

---

# Step 7 — Generate the first migration

Now run from:

```text
backend/
```

with your venv active:

```powershell
alembic revision --autogenerate -m "create documents and document chunks"
```

Alembic should detect:

```text
documents
document_chunks
```

and create something like:

```text
alembic/
└── versions/
    └── xxxxx_create_documents_and_document_chunks.py
```

---

# Step 8 — Apply the migration

Run:

```powershell
alembic upgrade head
```

Now PostgreSQL should contain:

```text
rag_db
│
├── documents
├── document_chunks
└── alembic_version
```

---

# Step 9 — Verify PostgreSQL

Run:

```powershell
docker exec -it rag-postgres psql -U rag_user -d rag_db
```

Then:

```sql
\dt
```

You should see:

```text
public | alembic_version
public | document_chunks
public | documents
```

Check the documents table:

```sql
\d documents
```

And:

```sql
\d document_chunks
```

Then exit:

```sql
\q
```

---

# Step 10 — Test SQLAlchemy connection

We should also verify that FastAPI can actually communicate with PostgreSQL.

Create:

```text
app/api/database.py
```

```python
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import get_db

router = APIRouter(
    prefix="/database",
    tags=["Database"],
)


@router.get("/health")
def database_health(
    db: Session = Depends(get_db),
) -> dict:
    db.execute(text("SELECT 1"))

    return {
        "status": "ok",
        "database": "connected",
    }
```

Then update:

```text
app/main.py
```

Add:

```python
from app.api.database import router as database_router
```

and:

```python
app.include_router(
    database_router,
    prefix=settings.api_v1_prefix,
)
```

Now:

```text
GET /api/v1/database/health
```

should return:

```json
{
  "status": "ok",
  "database": "connected"
}
```

---

# Step 11 — Our database layer now looks like this

```text
app/
│
├── db/
│   ├── database.py
│   └── base.py
│
├── models/
│   ├── __init__.py
│   ├── document.py
│   └── document_chunk.py
│
└── api/
    └── database.py

alembic/
│
├── versions/
│   └── create_documents_and_document_chunks.py
│
└── env.py
```

And the database:

```text
                 PostgreSQL
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
     documents          document_chunks
          │                     │
          └───────── 1:N ──────┘
```

---

# Why these two tables first?

Because they directly support the upcoming ingestion pipeline.

Eventually:

```text
Upload PDF
    ↓
Create Document
    ↓
Extract text
    ↓
Create DocumentChunk rows
    ↓
Generate embeddings
    ↓
Store vectors in Qdrant
```

So:

```text
documents.id
      │
      ▼
document_chunks.document_id
      │
      ▼
Qdrant payload
      │
      └── document_id
      └── chunk_id
      └── page_number
```

This gives us the link between the **relational database and vector database**.

---

## One improvement to make now

For timestamps, I recommend using PostgreSQL's timezone-aware server-side timestamps rather than `datetime.utcnow` as we move toward production.

We can refine that before finalizing the models.

---

# Phase 2 completion checklist

```text
DATABASE LAYER
│
├── PostgreSQL                    ⬜
│
├── SQLAlchemy                   ⬜
│
├── Database session              ⬜
│
├── Document model                ⬜
│
├── DocumentChunk model           ⬜
│
├── Relationship                  ⬜
│
├── Alembic configuration         ⬜
│
├── Initial migration             ⬜
│
├── Migration applied             ⬜
│
└── Database health endpoint      ⬜
```

**Do these steps in order.** Once you run `alembic revision --autogenerate` and `alembic upgrade head`, we can verify the output before moving to **Phase 3 — Document Ingestion**.
