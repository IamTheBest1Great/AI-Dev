# Phase 4 — Flight Search with `randomapi.dev`

Yes. Let's rebuild **Phase 4 from scratch, completely and consistently**, using the real mock API at `randomapi.dev` rather than hard-coded flight data.

I checked the current documentation: `GET https://randomapi.dev/api/flights` requires 3-letter IATA hub codes for `origin` and `destination`, supports `status`, `from`, `to`, `cabin`, `count`, and `seed`, requires no API key, and returns synthetic flight fixtures in a `data` envelope. The flight records contain airline, airports, status, departure/arrival, duration, aircraft, cabin, gate, and terminal. They do **not** represent real schedules, fares, availability, or live status. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

We'll therefore build a clean adapter around that API.

---

# 1. Phase 4 goal

We want this:

```text
                    Flight Request
                         │
                         ▼
                  FastAPI Route
                         │
                         ▼
              FlightSearchRequest
                         │
                         ▼
                 Flight Service
                         │
                         ▼
              randomapi.dev API
                         │
                         ▼
              Provider JSON response
                         │
                         ▼
                Normalize / Map
                         │
                         ▼
             FlightSearchResponse
                         │
                         ▼
                       User
```

Later:

```text
Agent
  ↓
flight_search_tool
  ↓
flight_service
  ↓
real flight provider
```

So we're building the correct abstraction now.

---

# 2. Files for Phase 4

## New files

```text
app/
├── api/
│   └── routes/
│       └── flights.py
│
├── schemas/
│   └── flight.py
│
├── services/
│   └── flight_service.py
│
└── tools/
    └── flight_tools.py

tests/
└── test_flight.py
```

## Modified file

```text
app/main.py
```

We do **not** need a database migration for this phase.

---

# 3. Final project structure

After Phase 4:

```text
BACKEND/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── travel.py
│   │       ├── weather.py
│   │       ├── travel_dates.py
│   │       └── flights.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── session.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── trip.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── trip.py
│   │   ├── weather.py
│   │   ├── weather_decision.py
│   │   └── flight.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── trip_service.py
│   │   ├── geocoding_service.py
│   │   ├── weather_service.py
│   │   ├── weather_decision_service.py
│   │   └── flight_service.py
│   │
│   └── tools/
│       ├── __init__.py
│       ├── weather_tools.py
│       └── flight_tools.py
│
├── migrations/
│
├── tests/
│   ├── test_health.py
│   ├── test_trip.py
│   ├── test_weather.py
│   ├── test_weather_decision.py
│   └── test_flight.py
│
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── venv/
```

---

# 4. Dependency

You already installed `httpx` for the weather phase.

Verify:

```powershell
pip install httpx
```

Then:

```powershell
pip freeze > requirements.txt
```

---

# 5. `app/schemas/flight.py`

Create:

```text
app/schemas/flight.py
```

Full code:

```python
from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator


class FlightSearchRequest(BaseModel):
    origin: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Za-z]{3}$",
        description="Origin IATA airport code",
    )

    destination: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Za-z]{3}$",
        description="Destination IATA airport code",
    )

    departure_date: date

    return_date: date | None = None

    passengers: int = Field(
        default=1,
        ge=1,
        le=9,
    )

    cabin: str = Field(
        default="economy",
    )

    count: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    seed: int = Field(
        default=42,
    )

    @model_validator(mode="after")
    def validate_request(self):

        if self.origin.upper() == self.destination.upper():
            raise ValueError(
                "Origin and destination must be different"
            )

        if (
            self.return_date is not None
            and self.return_date < self.departure_date
        ):
            raise ValueError(
                "return_date must be on or after departure_date"
            )

        allowed_cabins = {
            "economy",
            "premiumEconomy",
            "business",
            "first",
        }

        if self.cabin not in allowed_cabins:
            raise ValueError(
                "Invalid cabin. Choose one of: "
                "economy, premiumEconomy, business, first"
            )

        return self


class AirportInfo(BaseModel):
    iata: str
    name: str
    city: str
    country: str
    country_code: str
    timezone: str | None = None


class AirlineInfo(BaseModel):
    name: str
    code: str


class FlightOption(BaseModel):
    id: str
    flight_number: str

    airline: AirlineInfo

    origin: AirportInfo
    destination: AirportInfo

    status: str
    status_as_of: datetime

    departure: datetime
    arrival: datetime

    duration_minutes: int

    aircraft: str
    cabin: str

    gate: str | None = None
    terminal: str | None = None


class FlightSearchResponse(BaseModel):
    origin: str
    destination: str

    departure_date: date
    return_date: date | None

    passengers: int

    provider: str

    outbound_flights: list[FlightOption]
    return_flights: list[FlightOption]
```

---

# 6. What this schema does

The input:

```json
{
  "origin": "CPH",
  "destination": "JFK",
  "departure_date": "2026-10-05"
}
```

gets converted into:

```text
FlightSearchRequest
```

The provider response gets converted into:

```text
FlightOption
```

and finally:

```text
FlightSearchResponse
```

---

# 7. Why there is no price

This is important.

`randomapi.dev` currently documents its flight endpoint as **fictional fixture data**, not a fare/availability API. Its documented response schema does not provide a flight fare. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

Therefore we should **not** create:

```python
price: float
```

and pretend the value came from the provider.

Later, when we integrate a provider that actually supplies fares, we'll add price and availability.

---

# 8. `app/services/flight_service.py`

This is the most important file in Phase 4.

Create:

```text
app/services/flight_service.py
```

Use this complete code:

```python
import httpx

from app.schemas.flight import (
    AirlineInfo,
    AirportInfo,
    FlightOption,
    FlightSearchRequest,
    FlightSearchResponse,
)


FLIGHT_API_URL = (
    "https://randomapi.dev/api/flights"
)

PROVIDER_NAME = "randomapi.dev"


def _map_airline(
    data: dict,
) -> AirlineInfo:

    return AirlineInfo(
        name=data["name"],
        code=data["code"],
    )


def _map_airport(
    data: dict,
) -> AirportInfo:

    return AirportInfo(
        iata=data["iata"],
        name=data["name"],
        city=data["city"],
        country=data["country"],
        country_code=data["countryCode"],
        timezone=data.get("timezone"),
    )


def _map_flight(
    data: dict,
) -> FlightOption:

    return FlightOption(
        id=data["id"],
        flight_number=data["flightNumber"],
        airline=_map_airline(
            data["airline"]
        ),
        origin=_map_airport(
            data["origin"]
        ),
        destination=_map_airport(
            data["destination"]
        ),
        status=data["status"],
        status_as_of=data["statusAsOf"],
        departure=data["departure"],
        arrival=data["arrival"],
        duration_minutes=data[
            "durationMinutes"
        ],
        aircraft=data["aircraft"],
        cabin=data["cabin"],
        gate=data.get("gate"),
        terminal=data.get("terminal"),
    )


async def _request_flights(
    origin: str,
    destination: str,
    departure_date: str,
    count: int,
    seed: int,
    cabin: str,
) -> list[FlightOption]:

    params = {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "status": "scheduled",
        "from": departure_date,
        "to": departure_date,
        "count": count,
        "seed": seed,
        "cabin": cabin,
    }

    async with httpx.AsyncClient(
        timeout=10.0
    ) as client:

        response = await client.get(
            FLIGHT_API_URL,
            params=params,
        )

        response.raise_for_status()

        payload = response.json()

    records = payload.get(
        "data",
        [],
    )

    return [
        _map_flight(record)
        for record in records
    ]


async def search_flights(
    request: FlightSearchRequest,
) -> FlightSearchResponse:

    origin = request.origin.upper()
    destination = request.destination.upper()

    # ----------------------------------------
    # Outbound flights
    # ----------------------------------------

    outbound_flights = await _request_flights(
        origin=origin,
        destination=destination,
        departure_date=(
            request.departure_date.isoformat()
        ),
        count=request.count,
        seed=request.seed,
        cabin=request.cabin,
    )

    # ----------------------------------------
    # Return flights
    # ----------------------------------------

    return_flights = []

    if request.return_date is not None:

        return_flights = await _request_flights(
            origin=destination,
            destination=origin,
            departure_date=(
                request.return_date.isoformat()
            ),
            count=request.count,
            seed=request.seed + 1,
            cabin=request.cabin,
        )

    # ----------------------------------------
    # Final response
    # ----------------------------------------

    return FlightSearchResponse(
        origin=origin,
        destination=destination,
        departure_date=request.departure_date,
        return_date=request.return_date,
        passengers=request.passengers,
        provider=PROVIDER_NAME,
        outbound_flights=outbound_flights,
        return_flights=return_flights,
    )
```

---

# 9. What this service does

There are three main jobs.

## Job 1 — Call provider

It constructs:

```text
GET https://randomapi.dev/api/flights
```

with:

```text
origin
destination
status
from
to
count
seed
cabin
```

These parameters are documented by the provider. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

---

## Job 2 — Receive provider JSON

The provider returns:

```json
{
  "data": [
    {
      "id": "...",
      "flightNumber": "...",
      "airline": {},
      "origin": {},
      "destination": {},
      "status": "scheduled",
      "statusAsOf": "...",
      "departure": "...",
      "arrival": "...",
      "durationMinutes": 474,
      "aircraft": "...",
      "cabin": "economy",
      "gate": "...",
      "terminal": "..."
    }
  ]
}
```

The current API documentation describes this envelope and those fields. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

---

## Job 3 — Normalize

Provider:

```text
flightNumber
durationMinutes
statusAsOf
countryCode
```

Our application:

```text
flight_number
duration_minutes
status_as_of
country_code
```

This is called **normalization**.

That means the rest of our application does not have to know randomapi.dev's field naming conventions.

---

# 10. Why `status=scheduled`

We want future flights for travel planning.

So we explicitly send:

```python
"status": "scheduled"
```

The provider supports `scheduled`, `boarding`, `departed`, `arrived`, and `cancelled`. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

For this phase, `scheduled` is the relevant fixture state.

---

# 11. Why `from` and `to` are the same date

If the requested date is:

```text
2026-10-05
```

we send:

```text
from=2026-10-05
to=2026-10-05
```

The API defines `from` as the earliest synthetic scheduled departure and `to` as the latest synthetic scheduled departure. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

---

# 12. Why use `seed`

We use:

```python
seed=request.seed
```

The provider documents the seed as deterministic: the same seed produces the same generated records. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

For example:

```text
Outbound
seed = 42

Return
seed = 43
```

This lets us generate two independent fixture sets.

---

# 13. `app/tools/flight_tools.py`

Create:

```text
app/tools/flight_tools.py
```

```python
from app.schemas.flight import (
    FlightSearchRequest,
    FlightSearchResponse,
)

from app.services.flight_service import (
    search_flights,
)


async def flight_search_tool(
    request: FlightSearchRequest,
) -> FlightSearchResponse:

    return await search_flights(
        request=request
    )
```

---

# 14. Why do we have a tool?

Later the architecture will become:

```text
LLM / Agent
      │
      ▼
flight_search_tool
      │
      ▼
flight_service
      │
      ▼
flight provider
```

The agent does **not** need to know:

```text
HTTP
randomapi.dev
query parameters
JSON mapping
```

That's the job of our application layer.

---

# 15. `app/api/routes/flights.py`

Create:

```text
app/api/routes/flights.py
```

```python
import httpx

from fastapi import APIRouter, HTTPException

from app.schemas.flight import (
    FlightSearchRequest,
    FlightSearchResponse,
)

from app.services.flight_service import (
    search_flights,
)


router = APIRouter(
    prefix="/flights",
    tags=["Flights"],
)


@router.post(
    "/search",
    response_model=FlightSearchResponse,
)
async def search(
    request: FlightSearchRequest,
):

    try:

        return await search_flights(
            request=request
        )

    except httpx.HTTPStatusError as error:

        status_code = (
            error.response.status_code
        )

        if status_code == 400:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Flight provider rejected "
                    "the request. Check that the "
                    "airport codes are supported."
                ),
            )

        raise HTTPException(
            status_code=502,
            detail=(
                "Flight provider returned "
                f"HTTP {status_code}"
            ),
        )

    except httpx.HTTPError:

        raise HTTPException(
            status_code=502,
            detail=(
                "Flight provider is "
                "currently unavailable"
            ),
        )
```

---

# 16. Why handle provider errors?

The provider may reject unsupported airport codes.

For example:

```text
XYZ
```

may not belong to its supported hub network.

The provider documents that codes outside its supported hubs return `400`. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

Our API turns that into a useful application-level error.

---

# 17. Update `app/main.py`

Your complete `main.py` should now be:

```python
from fastapi import FastAPI

from app.api.routes.health import (
    router as health_router,
)

from app.api.routes.travel import (
    router as travel_router,
)

from app.api.routes.weather import (
    router as weather_router,
)

from app.api.routes.travel_dates import (
    router as travel_dates_router,
)

from app.api.routes.flights import (
    router as flights_router,
)


app = FastAPI(
    title="Travel Agent API",
    version="0.1.0",
)


app.include_router(
    health_router
)

app.include_router(
    travel_router
)

app.include_router(
    weather_router
)

app.include_router(
    travel_dates_router
)

app.include_router(
    flights_router
)


@app.get("/")
def root():

    return {
        "message": "Travel Agent API is running"
    }
```

---

# 18. Test the provider directly first

Before testing our backend, verify that randomapi.dev itself works.

Use the documented route style:

```text
https://randomapi.dev/api/flights?origin=CPH&destination=JFK&count=5&seed=42
```

The provider's documentation gives CPH → JFK as a working example. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

You can open this in the browser or use PowerShell:

```powershell
Invoke-RestMethod `
  "https://randomapi.dev/api/flights?origin=CPH&destination=JFK&status=scheduled&from=2026-10-05&to=2026-10-05&count=5&seed=42&cabin=economy"
```

You should receive JSON with:

```text
data
meta
```

and flight records inside `data`.

---

# 19. Start your backend

```powershell
.\venv\Scripts\Activate.ps1
```

Then:

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

You should see:

```text
GET  /health
POST /travel/plan
GET  /weather/{destination}
POST /travel/best-dates
POST /flights/search
```

---

# 20. Test `/flights/search`

Use:

```http
POST /flights/search
```

Body:

```json
{
  "origin": "CPH",
  "destination": "JFK",
  "departure_date": "2026-10-05",
  "return_date": "2026-10-08",
  "passengers": 2,
  "cabin": "economy",
  "count": 5,
  "seed": 42
}
```

Use a route supported by the mock provider; CPH → JFK is explicitly documented. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

---

# 21. Expected response structure

You should receive something like:

```json
{
  "origin": "CPH",
  "destination": "JFK",
  "departure_date": "2026-10-05",
  "return_date": "2026-10-08",
  "passengers": 2,
  "provider": "randomapi.dev",
  "outbound_flights": [
    {
      "id": "flt_...",
      "flight_number": "NX1-1000",
      "airline": {
        "name": "Example Airline",
        "code": "NX1"
      },
      "origin": {
        "iata": "CPH",
        "name": "Copenhagen Kastrup Airport",
        "city": "Copenhagen",
        "country": "Denmark",
        "country_code": "DK",
        "timezone": "Europe/Copenhagen"
      },
      "destination": {
        "iata": "JFK",
        "name": "John F. Kennedy International Airport",
        "city": "New York",
        "country": "United States",
        "country_code": "US",
        "timezone": "America/New_York"
      },
      "status": "scheduled",
      "status_as_of": "...",
      "departure": "...",
      "arrival": "...",
      "duration_minutes": 474,
      "aircraft": "wide-body twinjet",
      "cabin": "economy",
      "gate": "B17",
      "terminal": "T2"
    }
  ],
  "return_flights": []
}
```

The exact synthetic records depend on parameters and seed. The provider documents these response fields and explicitly marks them as fictional fixture values. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

---

# 22. Why `return_flights` might be empty

Don't immediately assume this is a bug.

We are making two separate provider requests:

```text
CPH → JFK
Oct 5
seed 42
```

and:

```text
JFK → CPH
Oct 8
seed 43
```

The synthetic generator can produce different results for the two requests.

---

# 23. One-way search

This should also work:

```json
{
  "origin": "CPH",
  "destination": "JFK",
  "departure_date": "2026-10-05",
  "passengers": 1,
  "cabin": "economy",
  "count": 5,
  "seed": 42
}
```

Result:

```text
outbound_flights → populated
return_flights   → []
```

---

# 24. `passengers` clarification

You may notice that:

```python
passengers
```

isn't sent to randomapi.dev.

That's deliberate.

The current mock flight API does not document passenger count as one of its flight-generation parameters. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

We still keep it in our application schema because our **travel domain** needs it.

Later, a real provider may use it for:

```text
availability
fares
passenger pricing
fare rules
```

---

# 25. Validation tests

Create:

```text
tests/test_flight.py
```

Use this complete version:

```python
from datetime import date, datetime

import pytest

from fastapi.testclient import TestClient

from app.main import app

from app.schemas.flight import (
    AirlineInfo,
    AirportInfo,
    FlightOption,
    FlightSearchResponse,
)


client = TestClient(app)


def create_mock_flight() -> FlightOption:

    return FlightOption(
        id="flt_test_001",
        flight_number="NX1-1000",
        airline=AirlineInfo(
            name="Test Airline",
            code="NX1",
        ),
        origin=AirportInfo(
            iata="CPH",
            name="Copenhagen Airport",
            city="Copenhagen",
            country="Denmark",
            country_code="DK",
            timezone="Europe/Copenhagen",
        ),
        destination=AirportInfo(
            iata="JFK",
            name="John F. Kennedy International Airport",
            city="New York",
            country="United States",
            country_code="US",
            timezone="America/New_York",
        ),
        status="scheduled",
        status_as_of=datetime(
            2026,
            10,
            2,
            12,
            0,
        ),
        departure=datetime(
            2026,
            10,
            5,
            10,
            0,
        ),
        arrival=datetime(
            2026,
            10,
            5,
            18,
            0,
        ),
        duration_minutes=480,
        aircraft="wide-body twinjet",
        cabin="economy",
        gate="B17",
        terminal="T2",
    )


def test_flight_search_endpoint(
    monkeypatch,
):

    mock_flight = create_mock_flight()

    async def mock_search_flights(request):

        return FlightSearchResponse(
            origin=request.origin.upper(),
            destination=request.destination.upper(),
            departure_date=request.departure_date,
            return_date=request.return_date,
            passengers=request.passengers,
            provider="randomapi.dev",
            outbound_flights=[
                mock_flight
            ],
            return_flights=[
                mock_flight
            ],
        )

    monkeypatch.setattr(
        "app.api.routes.flights.search_flights",
        mock_search_flights,
    )

    response = client.post(
        "/flights/search",
        json={
            "origin": "CPH",
            "destination": "JFK",
            "departure_date": "2026-10-05",
            "return_date": "2026-10-08",
            "passengers": 2,
            "cabin": "economy",
            "count": 5,
            "seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["origin"] == "CPH"
    assert data["destination"] == "JFK"

    assert (
        data["departure_date"]
        == "2026-10-05"
    )

    assert (
        data["return_date"]
        == "2026-10-08"
    )

    assert data["passengers"] == 2

    assert (
        data["provider"]
        == "randomapi.dev"
    )

    assert len(
        data["outbound_flights"]
    ) == 1

    assert len(
        data["return_flights"]
    ) == 1


def test_same_airport_rejected():

    response = client.post(
        "/flights/search",
        json={
            "origin": "CPH",
            "destination": "CPH",
            "departure_date": "2026-10-05",
        },
    )

    assert response.status_code == 422


def test_invalid_return_date():

    response = client.post(
        "/flights/search",
        json={
            "origin": "CPH",
            "destination": "JFK",
            "departure_date": "2026-10-08",
            "return_date": "2026-10-05",
        },
    )

    assert response.status_code == 422


def test_invalid_cabin():

    response = client.post(
        "/flights/search",
        json={
            "origin": "CPH",
            "destination": "JFK",
            "departure_date": "2026-10-05",
            "cabin": "invalid",
        },
    )

    assert response.status_code == 422


def test_invalid_iata_code():

    response = client.post(
        "/flights/search",
        json={
            "origin": "MUMBAI",
            "destination": "JFK",
            "departure_date": "2026-10-05",
        },
    )

    assert response.status_code == 422
```

---

# 26. Why the tests don't call randomapi.dev

This is important.

Your application calls:

```text
randomapi.dev
```

during real execution.

But automated tests should not depend on:

```text
Internet
External API uptime
External API behavior
Rate limits
```

So we mock:

```python
search_flights()
```

inside the route test.

That makes our test deterministic.

---

# 27. Run tests

Run:

```powershell
pytest
```

You should now have:

```text
tests/test_trip.py
tests/test_weather.py
tests/test_weather_decision.py
tests/test_flight.py
```

and the flight tests should all pass.

Because your previous suite had 4 tests, your total count will depend on how many tests you currently have in those existing files.

The important result is:

```text
FAILED 0
```

---

# 28. Test schema validation manually

### Invalid origin

```json
{
  "origin": "MUMBAI",
  "destination": "JFK",
  "departure_date": "2026-10-05"
}
```

Your API should return:

```text
422
```

because the schema expects a three-letter IATA code.

---

### Same origin/destination

```json
{
  "origin": "CPH",
  "destination": "CPH",
  "departure_date": "2026-10-05"
}
```

Returns:

```text
422
```

---

### Invalid return date

```json
{
  "origin": "CPH",
  "destination": "JFK",
  "departure_date": "2026-10-08",
  "return_date": "2026-10-05"
}
```

Returns:

```text
422
```

---

# 29. Test unsupported airport

Try a code that isn't in randomapi.dev's supported hub network:

```json
{
  "origin": "XYZ",
  "destination": "JFK",
  "departure_date": "2026-10-05"
}
```

Our Pydantic validation accepts `XYZ` because it's syntactically a valid three-letter code.

Then randomapi.dev may reject it with `400`, because the provider accepts only its supported hub network. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

Our route converts that provider rejection into:

```text
400
Flight provider rejected the request.
Check that the airport codes are supported.
```

That's exactly the distinction we want:

```text
Our validation
     ↓
Is this syntactically an IATA-style code?

Provider validation
     ↓
Does the mock provider actually support this hub?
```

---

# 30. Current Phase 4 architecture

We now have:

```text
                         USER
                           │
                           ▼
                  POST /flights/search
                           │
                           ▼
                  FlightSearchRequest
                           │
                           ▼
                   Flight API Route
                           │
                           ▼
                   Flight Service
                           │
                           ▼
                  HTTP GET request
                           │
                           ▼
                    randomapi.dev
                           │
                           ▼
                    JSON response
                           │
                           ▼
                  Provider → App mapping
                           │
                           ▼
                  FlightSearchResponse
                           │
                           ▼
                          USER
```

And the future agent path is:

```text
                         AGENT
                           │
                           ▼
                 flight_search_tool
                           │
                           ▼
                   Flight Service
                           │
                           ▼
                   Flight Provider
```

---

# 31. How Phase 4 connects to Phase 3

This is where our travel agent is starting to become interesting.

Phase 3:

```text
Destination
    ↓
Weather Forecast
    ↓
Weather Decision Engine
    ↓
Best Travel Window

Example:
Oct 5 → Oct 8
```

Phase 4:

```text
Origin
Destination
Travel Window
    ↓
Flight Search
    ↓
Available fixture flights
```

Together:

```text
                 USER
                  │
       "Find a 4-day trip"
                  │
                  ▼
          Weather Engine
                  │
                  ▼
           Oct 5 → Oct 8
                  │
                  ▼
           Flight Search
                  │
          ┌───────┴────────┐
          ▼                ▼
      Outbound           Return
       Oct 5              Oct 8
```

This is the exact foundation we need for the eventual agent.

---

# 32. What we are NOT doing in Phase 4

Do **not** add these yet:

```text
❌ Cheapest flight selection
❌ Best flight recommendation
❌ Flight ranking
❌ Booking
❌ Payment
❌ User approval
❌ LLM
❌ LangGraph
❌ Memory
```

Right now the flight capability has one responsibility:

> Search and return flight fixtures for a requested route/date.

---

# 33. Definition of Done

Phase 4 is complete when:

```text
✅ FlightSearchRequest schema
✅ IATA validation
✅ Date validation
✅ Cabin validation
✅ Passenger validation

✅ randomapi.dev integration
✅ HTTP client
✅ Scheduled-flight filtering
✅ Date filtering
✅ Cabin filtering
✅ Count
✅ Seed
✅ Outbound search
✅ Return search

✅ Provider response normalization
✅ FlightSearchResponse

✅ Flight service
✅ Flight tool
✅ Flight API route
✅ Error handling
✅ Tests
```

And your API exposes:

```text
GET  /health
POST /travel/plan
GET  /weather/{destination}
POST /travel/best-dates
POST /flights/search
```

---

# 34. One limitation to remember

`randomapi.dev` is a **mock-data provider**, not a real flight inventory system. It explicitly says its generated airlines, flight numbers, departure/arrival times, gates, terminals, and statuses are fictional, and that it does not provide real schedules, fares, availability, or live status. Its route generator also accepts only its curated hub network. [randomapi.dev](https://randomapi.dev/apis/flights?utm_source=chatgpt.com)

That's exactly why it is appropriate **at this stage**.

Our architecture will later allow:

```text
randomapi.dev
      ↓
      ↓ replace provider
      ↓
Real Flight API
```

without changing the agent/tool/API architecture.

### Phase sequence now

```text
Phase 0 ✅ Infrastructure
      ↓
Phase 1 ✅ Trip + PostgreSQL
      ↓
Phase 2 ✅ Weather Integration
      ↓
Phase 3 ✅ Weather Decision Engine
      ↓
Phase 4 🔵 Flight Search
      ↓
Phase 5 → Hotel Search
      ↓
Phase 6 → Cab Search
      ↓
Phase 7 → Combine capabilities
```

For now, implement the files above and run `pytest`. **Don't start Phase 5 until the flight endpoint works with `CPH → JFK` and the full test suite is passing.**
