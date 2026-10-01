# Phase 1 — Trip API

Now that Phase 0 is done, Phase 1 is our **first complete vertical slice**:

```text
User
 │
 ▼
POST /travel/plan
 │
 ▼
Pydantic Schema
 │
 ▼
Route
 │
 ▼
Trip Service
 │
 ▼
SQLAlchemy Model
 │
 ▼
PostgreSQL
 │
 ▼
Response Schema
 │
 ▼
User
```

This follows the Phase 1 structure from your project plan: schema → model → migration → service → route → test. Pasted markdown

---

# 1. What Phase 1 will achieve

At the end, you will be able to send:

```http
POST /travel/plan
```

with:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "duration": 4,
  "budget": 30000
}
```

And the backend will:

```text
Validate request
      ↓
Create Trip
      ↓
Save Trip in PostgreSQL
      ↓
Return trip_id
```

Example response:

```json
{
  "id": "8b8c...",
  "origin": "Mumbai",
  "destination": "Goa",
  "duration": 4,
  "budget": 30000,
  "status": "PLANNING"
}
```

---

# 2. Phase 1 folder structure

We're going to add:

```text
BACKEND/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       └── travel.py          ← NEW
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── database/
│   │   ├── base.py
│   │   └── session.py
│   │
│   ├── models/
│   │   └── trip.py                ← NEW
│   │
│   ├── schemas/
│   │   └── trip.py                ← NEW
│   │
│   └── services/
│       └── trip_service.py        ← NEW
│
├── migrations/
│   └── ...
│
├── tests/
│   └── test_trip.py               ← NEW
│
├── alembic.ini                    ← NEW
├── .env
├── docker-compose.yml
└── requirements.txt
```

---

# 3. Phase 1 — Step 1: Install Alembic

We need migrations because we're adding our first database table.

Run:

```powershell
pip install alembic
```

Then update:

```powershell
pip freeze > requirements.txt
```

---

# 4. Step 2: Create the Trip schema

Create:

```text
app/schemas/trip.py
```

This defines what the API accepts and returns.

```python
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field


class TripCreate(BaseModel):
    origin: str = Field(min_length=2, max_length=100)
    destination: str = Field(min_length=2, max_length=100)
    duration: int = Field(gt=0, le=365)
    budget: Decimal = Field(gt=0)


class TripResponse(BaseModel):
    id: UUID
    origin: str
    destination: str
    duration: int
    budget: Decimal
    status: str

    model_config = {
        "from_attributes": True
    }
```

---

# 5. Understand the schema

When the user sends:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "duration": 4,
  "budget": 30000
}
```

`TripCreate` validates it.

For example:

```json
{
  "origin": "",
  "destination": "Goa",
  "duration": -4,
  "budget": -100
}
```

will fail validation.

So:

```text
HTTP Request
     ↓
TripCreate
     ↓
Valid?
 ┌───┴────┐
NO       YES
 │         │
 ▼         ▼
422      Service
```

---

# 6. Step 3: Create Trip model

Create:

```text
app/models/trip.py
```

```python
import uuid
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import DateTime, Numeric, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    origin: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    destination: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    duration: Mapped[int] = mapped_column(
        nullable=False
    )

    budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="PLANNING"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
```

---

# 7. Why do we need both Schema and Model?

This is important.

### Schema

```text
app/schemas/trip.py
```

represents:

> **API data**

### Model

```text
app/models/trip.py
```

represents:

> **Database data**

So:

```text
                    REQUEST
                       │
                       ▼
                TripCreate
                  SCHEMA
                       │
                       ▼
                 Trip Service
                       │
                       ▼
                    Trip
                   MODEL
                       │
                       ▼
                 PostgreSQL
```

They are not the same thing.

---

# 8. Step 4: Update `database/base.py`

Your existing file is:

```text
app/database/base.py
```

Currently:

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

Keep that.

But Alembic needs to know about your model.

We'll handle that in the migration configuration.

---

# 9. Step 5: Create `models/__init__.py`

Create:

```text
app/models/__init__.py
```

```python
from app.models.trip import Trip

__all__ = ["Trip"]
```

This isn't strictly required for SQLAlchemy, but it keeps the model package clean.

---

# 10. Step 6: Create Alembic

From the `BACKEND` directory:

```powershell
alembic init migrations
```

This creates:

```text
migrations/
├── versions/
├── env.py
├── README
└── script.py.mako

alembic.ini
```

Your project now becomes:

```text
BACKEND/
│
├── app/
├── migrations/
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
│
└── alembic.ini
```

---

# 11. Step 7: Configure Alembic

Open:

```text
migrations/env.py
```

Find:

```python
target_metadata = None
```

Replace/configure the relevant imports and metadata:

```python
from app.core.config import settings
from app.database.base import Base
from app.models.trip import Trip


target_metadata = Base.metadata
```

And make sure Alembic uses your database URL.

A straightforward version of `migrations/env.py` is:

```python
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from app.core.config import settings
from app.database.base import Base
from app.models.trip import Trip


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


config.set_main_option(
    "sqlalchemy.url",
    settings.DATABASE_URL
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = settings.DATABASE_URL

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

---

# 12. Step 8: Create the migration

Run:

```powershell
alembic revision --autogenerate -m "create trips table"
```

Alembic should detect:

```text
Trip model
    ↓
New table detected
    ↓
Create migration
```

You'll get something like:

```text
migrations/
└── versions/
    └── abc123_create_trips_table.py
```

**Don't manually write the migration yet.**

Let Alembic generate it.

---

# 13. Step 9: Apply migration

Run:

```powershell
alembic upgrade head
```

Now:

```text
SQLAlchemy Model
       ↓
Alembic Migration
       ↓
PostgreSQL
       ↓
trips table
```

You can verify with DBeaver or PostgreSQL.

The table should contain approximately:

```text
trips
────────────────────────
id
origin
destination
duration
budget
status
created_at
```

---

# 14. Step 10: Create Trip Service

Now we implement the business logic.

Create:

```text
app/services/trip_service.py
```

```python
from sqlalchemy.orm import Session

from app.models.trip import Trip
from app.schemas.trip import TripCreate


def create_trip(
    db: Session,
    trip_data: TripCreate
) -> Trip:

    trip = Trip(
        origin=trip_data.origin,
        destination=trip_data.destination,
        duration=trip_data.duration,
        budget=trip_data.budget,
        status="PLANNING"
    )

    db.add(trip)
    db.commit()
    db.refresh(trip)

    return trip
```

---

# 15. What does the service do?

The service sits between the route and database.

```text
Route
  │
  ▼
Trip Service
  │
  ├── Create Trip object
  ├── Add to session
  ├── Commit
  └── Refresh
  │
  ▼
PostgreSQL
```

The route should **not** contain:

```python
db.add(...)
db.commit(...)
```

The service owns the business operation.

---

# 16. Step 11: Create Travel Route

Create:

```text
app/api/routes/travel.py
```

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.trip import TripCreate, TripResponse
from app.services.trip_service import create_trip


router = APIRouter(
    prefix="/travel",
    tags=["Travel"]
)


@router.post(
    "/plan",
    response_model=TripResponse,
    status_code=201
)
def plan_trip(
    trip_data: TripCreate,
    db: Session = Depends(get_db)
):
    trip = create_trip(
        db=db,
        trip_data=trip_data
    )

    return trip
```

---

# 17. Step 12: Connect route to `main.py`

Open:

```text
app/main.py
```

Update it:

```python
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.travel import router as travel_router
from app.core.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0"
)


app.include_router(health_router)
app.include_router(travel_router)


@app.get("/")
def root():
    return {
        "message": f"{settings.APP_NAME} is running",
        "debug": settings.DEBUG
    }
```

Now FastAPI knows:

```text
/health

/travel/plan
```

---

# 18. Complete Phase 1 request flow

Now the whole thing works:

```text
                         USER
                           │
                           │
                           ▼
                POST /travel/plan
                           │
                           ▼
                 ┌─────────────────┐
                 │  travel.py      │
                 │     ROUTE       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  TripCreate     │
                 │    SCHEMA       │
                 └────────┬────────┘
                          │
                    Valid data
                          │
                          ▼
                 ┌─────────────────┐
                 │ trip_service.py │
                 │    SERVICE      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Trip Model   │
                 │     MODEL       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   SQLAlchemy    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   PostgreSQL    │
                 │      trips      │
                 └────────┬────────┘
                          │
                          ▼
                 TripResponse Schema
                          │
                          ▼
                         USER
```

This is the same route → service → model → PostgreSQL flow specified in the project plan. Pasted markdown

---

# 19. Step 13: Run the backend

Start PostgreSQL:

```powershell
docker compose up -d
```

Activate venv if necessary:

```powershell
.\venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

You should now see:

```text
Travel

POST /travel/plan
```

---

# 20. Step 14: Test with Swagger

Click:

```text
POST /travel/plan
```

Then **Try it out**.

Send:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "duration": 4,
  "budget": 30000
}
```

Click **Execute**.

Expected response:

```json
{
  "id": "some-uuid",
  "origin": "Mumbai",
  "destination": "Goa",
  "duration": 4,
  "budget": "30000.00",
  "status": "PLANNING"
}
```

---

# 21. Step 15: Verify PostgreSQL

Open DBeaver or your PostgreSQL client.

Run:

```sql
SELECT * FROM trips;
```

You should see:

```text
id | origin | destination | duration | budget | status
-------------------------------------------------------
...| Mumbai | Goa         | 4        | 30000  | PLANNING
```

This is the important milestone:

```text
                    PHASE 1

Frontend/Postman
       │
       ▼
POST /travel/plan
       │
       ▼
Schema Validation
       │
       ▼
Route
       │
       ▼
Service
       │
       ▼
SQLAlchemy Model
       │
       ▼
PostgreSQL
       │
       ▼
Response
```

---

# 22. Step 16: Add the test

Create:

```text
tests/test_trip.py
```

For the first test, we can test the API validation separately:

```python
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_create_trip_validation():
    response = client.post(
        "/travel/plan",
        json={
            "origin": "Mumbai",
            "destination": "Goa",
            "duration": 4,
            "budget": 30000
        }
    )

    assert response.status_code == 201

    data = response.json()

    assert data["origin"] == "Mumbai"
    assert data["destination"] == "Goa"
    assert data["duration"] == 4
    assert data["status"] == "PLANNING"
```

### One caveat

This test currently uses your configured PostgreSQL database. For a proper production-quality test suite, we'll later create a **separate test database/transaction fixture** rather than writing test data into your development database.

For Phase 1, the important thing is to understand the complete request path.

---

# 23. Phase 1 — Definition of Done

Don't move to Phase 2 until these work:

```text
□ TripCreate schema works
□ Invalid trip data is rejected
□ Trip SQLAlchemy model works
□ Alembic is configured
□ trips migration generated
□ trips migration applied
□ trips table exists
□ Trip service creates records
□ POST /travel/plan works
□ Trip is saved in PostgreSQL
□ Response contains trip information
□ Swagger shows POST /travel/plan
□ Basic test passes
```

## What you have learned in Phase 1

```text
                    FASTAPI
                       │
                       ▼
                   ROUTING
                       │
                       ▼
                  PYDANTIC
                   SCHEMAS
                       │
                       ▼
                  SERVICES
                BUSINESS LOGIC
                       │
                       ▼
                  SQLALCHEMY
                    MODELS
                       │
                       ▼
                   ALEMBIC
                  MIGRATIONS
                       │
                       ▼
                 POSTGRESQL
```

**Phase 1 deliberately has no AI, no LangGraph, no tools, and no agents.** We're establishing the reliable application/data foundation first. The next phase will add the first external capability: **weather search**.
