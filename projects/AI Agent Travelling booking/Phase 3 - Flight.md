# Phase 4 — Flight Search

Phase 3 is complete, so now we add the **flight-search capability**.

For this phase, we will deliberately use a **mock flight provider** rather than connecting to a real airline/GDS API. That lets us finish the architecture and agent-facing tool without introducing API keys, provider-specific authentication, quotas, or booking contracts yet.

The Phase 4 flow is:

```text
Best Travel Dates
       │
       ▼
Flight Search
       │
       ├── Origin
       ├── Destination
       ├── Departure Date
       ├── Return Date
       └── Passengers
       │
       ▼
Flight Service
       │
       ▼
Mock Flight Provider
       │
       ▼
Normalize Flight Data
       │
       ▼
Flight Response
```

---

# 1. Final files for Phase 4

Add these:

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

And modify:

```text
app/main.py
```

Your relevant structure becomes:

```text
BACKEND/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       ├── travel.py
│   │       ├── weather.py
│   │       ├── travel_dates.py
│   │       └── flights.py          ← NEW
│   │
│   ├── schemas/
│   │   ├── trip.py
│   │   ├── weather.py
│   │   ├── weather_decision.py
│   │   └── flight.py               ← NEW
│   │
│   ├── services/
│   │   ├── trip_service.py
│   │   ├── geocoding_service.py
│   │   ├── weather_service.py
│   │   ├── weather_decision_service.py
│   │   └── flight_service.py       ← NEW
│   │
│   └── tools/
│       ├── weather_tools.py
│       └── flight_tools.py          ← NEW
│
└── tests/
    ├── test_health.py
    ├── test_trip.py
    ├── test_weather.py
    ├── test_weather_decision.py
    └── test_flight.py               ← NEW
```

No new Python package is required for the mock implementation.

---

# 2. What exactly are we building?

Suppose the weather engine gives us:

```text
Goa
October 5 → October 8
4 days
```

Now the flight layer receives:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "departure_date": "2026-10-05",
  "return_date": "2026-10-08",
  "passengers": 2
}
```

And returns:

```text
Outbound flights
    ↓
Mumbai → Goa
    ↓
October 5

Return flights
    ↓
Goa → Mumbai
    ↓
October 8
```

---

# 3. Create `app/schemas/flight.py`

This defines the application's internal flight structure.

```python
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator


class FlightSearchRequest(BaseModel):
    origin: str = Field(
        min_length=2,
        max_length=100
    )

    destination: str = Field(
        min_length=2,
        max_length=100
    )

    departure_date: date

    return_date: date | None = None

    passengers: int = Field(
        default=1,
        ge=1,
        le=9
    )

    max_stops: int = Field(
        default=2,
        ge=0,
        le=2
    )

    max_price: Decimal | None = Field(
        default=None,
        gt=0
    )

    @model_validator(mode="after")
    def validate_dates(self):
        if (
            self.return_date is not None
            and self.return_date < self.departure_date
        ):
            raise ValueError(
                "return_date must be on or after departure_date"
            )

        return self


class FlightOption(BaseModel):
    flight_id: str

    airline: str
    flight_number: str

    origin: str
    destination: str

    departure_time: datetime
    arrival_time: datetime

    duration_minutes: int
    stops: int

    price: Decimal
    currency: str


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

# 4. Why do we have these schemas?

## `FlightSearchRequest`

This is the input:

```text
Mumbai
Goa
5 Oct
8 Oct
2 passengers
```

---

## `FlightOption`

This represents **one flight**.

Example:

```text
MockJet
MJ201

Mumbai → Goa

Departure: 06:30
Arrival: 07:40

Duration: 70 min
Stops: 0

Price: ₹11,000
```

The `price` in this implementation is the **total price for the requested number of passengers**.

---

## `FlightSearchResponse`

This separates:

```text
outbound_flights
```

from:

```text
return_flights
```

which will be useful when we eventually book the trip.

---

# 5. Create `app/services/flight_service.py`

This is the actual flight-search service.

For now, the provider is mocked.

```python
from datetime import date, datetime, time, timedelta
from decimal import Decimal

from app.schemas.flight import (
    FlightOption,
    FlightSearchRequest,
    FlightSearchResponse,
)


MOCK_PROVIDER_NAME = "mock-flight-provider"


def _generate_mock_flights(
    origin: str,
    destination: str,
    flight_date: date,
    passengers: int,
) -> list[FlightOption]:

    templates = [
        {
            "airline": "DemoAir",
            "flight_number": "DA101",
            "departure_hour": 6,
            "departure_minute": 30,
            "duration_minutes": 75,
            "stops": 0,
            "price": Decimal("4500"),
        },
        {
            "airline": "MockJet",
            "flight_number": "MJ202",
            "departure_hour": 9,
            "departure_minute": 15,
            "duration_minutes": 90,
            "stops": 0,
            "price": Decimal("5200"),
        },
        {
            "airline": "Sample Airways",
            "flight_number": "SA303",
            "departure_hour": 12,
            "departure_minute": 45,
            "duration_minutes": 115,
            "stops": 1,
            "price": Decimal("3900"),
        },
        {
            "airline": "TestWings",
            "flight_number": "TW404",
            "departure_hour": 16,
            "departure_minute": 20,
            "duration_minutes": 95,
            "stops": 0,
            "price": Decimal("6100"),
        },
        {
            "airline": "DemoJet",
            "flight_number": "DJ505",
            "departure_hour": 20,
            "departure_minute": 10,
            "duration_minutes": 110,
            "stops": 1,
            "price": Decimal("4200"),
        },
    ]

    flights = []

    for index, template in enumerate(templates, start=1):

        departure = datetime.combine(
            flight_date,
            time(
                template["departure_hour"],
                template["departure_minute"],
            ),
        )

        arrival = (
            departure
            + timedelta(
                minutes=template["duration_minutes"]
            )
        )

        total_price = (
            template["price"] * passengers
        )

        flight = FlightOption(
            flight_id=(
                f"MOCK-{flight_date}-"
                f"{index}"
            ),
            airline=template["airline"],
            flight_number=template["flight_number"],
            origin=origin,
            destination=destination,
            departure_time=departure,
            arrival_time=arrival,
            duration_minutes=template[
                "duration_minutes"
            ],
            stops=template["stops"],
            price=total_price,
            currency="INR",
        )

        flights.append(flight)

    return flights


async def search_flights(
    request: FlightSearchRequest,
) -> FlightSearchResponse:

    # ----------------------------------------
    # Generate outbound flights
    # ----------------------------------------

    outbound_flights = _generate_mock_flights(
        origin=request.origin,
        destination=request.destination,
        flight_date=request.departure_date,
        passengers=request.passengers,
    )

    # ----------------------------------------
    # Apply max stops filter
    # ----------------------------------------

    outbound_flights = [
        flight
        for flight in outbound_flights
        if flight.stops <= request.max_stops
    ]

    # ----------------------------------------
    # Apply max price filter
    # ----------------------------------------

    if request.max_price is not None:
        outbound_flights = [
            flight
            for flight in outbound_flights
            if flight.price <= request.max_price
        ]

    # ----------------------------------------
    # Generate return flights
    # ----------------------------------------

    return_flights = []

    if request.return_date is not None:

        return_flights = _generate_mock_flights(
            origin=request.destination,
            destination=request.origin,
            flight_date=request.return_date,
            passengers=request.passengers,
        )

        return_flights = [
            flight
            for flight in return_flights
            if flight.stops <= request.max_stops
        ]

        if request.max_price is not None:
            return_flights = [
                flight
                for flight in return_flights
                if flight.price <= request.max_price
            ]

    return FlightSearchResponse(
        origin=request.origin,
        destination=request.destination,
        departure_date=request.departure_date,
        return_date=request.return_date,
        passengers=request.passengers,
        provider=MOCK_PROVIDER_NAME,
        outbound_flights=outbound_flights,
        return_flights=return_flights,
    )
```

---

# 6. Understand the flight service

The service currently does:

```text
FlightSearchRequest
        │
        ▼
Generate mock outbound flights
        │
        ▼
Apply filters
        │
        ▼
Generate mock return flights
        │
        ▼
Apply filters
        │
        ▼
FlightSearchResponse
```

There is **no LLM here**.

There is **no recommendation logic here**.

There is **no booking here**.

That's intentional.

---

# 7. Why mock flights?

We want the architecture to be:

```text
Agent
  ↓
Flight Tool
  ↓
Flight Service
  ↓
Flight Provider
```

Today:

```text
Flight Service
      ↓
Mock Provider
```

Later:

```text
Flight Service
      ↓
Real Flight API
```

So when we introduce a real provider, the rest of the application does not need to fundamentally change.

---

# 8. Create `app/tools/flight_tools.py`

This is the agent-facing interface.

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

Later an agent can effectively do:

```text
I need flights from Mumbai to Goa
for the dates I selected.
```

and invoke:

```text
flight_search_tool()
```

---

# 9. Create `app/api/routes/flights.py`

This exposes flight search through FastAPI.

```python
from fastapi import APIRouter

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

    return await search_flights(
        request=request
    )
```

---

# 10. Update `app/main.py`

Add:

```python
from app.api.routes.flights import (
    router as flights_router,
)
```

Then register it:

```python
app.include_router(
    flights_router
)
```

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

# 11. Test the API manually

Start the backend:

```powershell
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Go to:

```text
http://127.0.0.1:8000/docs
```

You should now see:

```text
GET  /health

POST /travel/plan

GET  /weather/{destination}

POST /travel/best-dates

POST /flights/search
```

---

# 12. Test `/flights/search`

Use:

```http
POST /flights/search
```

Body:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "departure_date": "2026-10-05",
  "return_date": "2026-10-08",
  "passengers": 2
}
```

---

# 13. Expected response

You should get something structurally like:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "departure_date": "2026-10-05",
  "return_date": "2026-10-08",
  "passengers": 2,
  "provider": "mock-flight-provider",
  "outbound_flights": [
    {
      "flight_id": "MOCK-2026-10-05-1",
      "airline": "DemoAir",
      "flight_number": "DA101",
      "origin": "Mumbai",
      "destination": "Goa",
      "departure_time": "2026-10-05T06:30:00",
      "arrival_time": "2026-10-05T07:45:00",
      "duration_minutes": 75,
      "stops": 0,
      "price": 9000,
      "currency": "INR"
    }
  ],
  "return_flights": [
    {
      "flight_id": "MOCK-2026-10-08-1",
      "airline": "DemoAir",
      "flight_number": "DA101",
      "origin": "Goa",
      "destination": "Mumbai",
      "departure_time": "2026-10-08T06:30:00",
      "arrival_time": "2026-10-08T07:45:00",
      "duration_minutes": 75,
      "stops": 0,
      "price": 9000,
      "currency": "INR"
    }
  ]
}
```

The exact list will contain multiple mock flights.

---

# 14. Test filters

## Maximum stops

Request:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "departure_date": "2026-10-05",
  "return_date": "2026-10-08",
  "passengers": 2,
  "max_stops": 0
}
```

Only nonstop flights should remain.

---

## Maximum price

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "departure_date": "2026-10-05",
  "return_date": "2026-10-08",
  "passengers": 2,
  "max_price": 10000
}
```

Flights whose total price exceeds ₹10,000 are removed.

---

# 15. Test one-way flight search

Because `return_date` is optional, this also works:

```json
{
  "origin": "Mumbai",
  "destination": "Goa",
  "departure_date": "2026-10-05",
  "passengers": 1
}
```

Then:

```text
outbound_flights → flights returned
return_flights   → []
```

---

# 16. Create `tests/test_flight.py`

```python
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_flight_search():

    response = client.post(
        "/flights/search",
        json={
            "origin": "Mumbai",
            "destination": "Goa",
            "departure_date": "2026-10-05",
            "return_date": "2026-10-08",
            "passengers": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["origin"] == "Mumbai"
    assert data["destination"] == "Goa"

    assert data["departure_date"] == "2026-10-05"
    assert data["return_date"] == "2026-10-08"

    assert data["passengers"] == 2

    assert data["provider"] == "mock-flight-provider"

    assert len(data["outbound_flights"]) > 0
    assert len(data["return_flights"]) > 0


def test_flight_search_max_stops():

    response = client.post(
        "/flights/search",
        json={
            "origin": "Mumbai",
            "destination": "Goa",
            "departure_date": "2026-10-05",
            "return_date": "2026-10-08",
            "passengers": 1,
            "max_stops": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    for flight in data["outbound_flights"]:
        assert flight["stops"] == 0

    for flight in data["return_flights"]:
        assert flight["stops"] == 0


def test_flight_search_invalid_dates():

    response = client.post(
        "/flights/search",
        json={
            "origin": "Mumbai",
            "destination": "Goa",
            "departure_date": "2026-10-08",
            "return_date": "2026-10-05",
            "passengers": 1,
        },
    )

    assert response.status_code == 422
```

---

# 17. Run the tests

```powershell
pytest
```

You should now have:

```text
tests/test_trip.py              ✅
tests/test_weather.py           ✅
tests/test_weather_decision.py  ✅
tests/test_flight.py            ✅
```

So the expected result becomes approximately:

```text
7 passed
```

because the existing four tests are joined by the three flight tests.

---

# 18. Important architecture distinction

At this point, you have:

```text
WEATHER

weather_service
      ↓
Weather Provider
      ↓
Forecast


WEATHER DECISION

weather_decision_service
      ↓
Weather Forecast
      ↓
Sliding Window
      ↓
Score
      ↓
Best Dates


FLIGHTS

flight_service
      ↓
Mock Flight Provider
      ↓
Available Flights
```

They are **three separate capabilities**.

---

# 19. We are NOT doing this yet

Do not add:

```text
❌ Flight recommendation
❌ Cheapest flight selection
❌ Best flight selection
❌ Booking
❌ Payment
❌ LLM
❌ LangGraph
❌ Memory
```

Those belong to later phases.

Right now the flight layer should simply answer:

> "Given these dates and route, what flights are available?"

---

# 20. Full Phase 4 flow

The complete feature now looks like:

```text
                    USER
                      │
                      │
               Travel dates
                      │
                      ▼
              Flight API Route
                      │
                      ▼
            FlightSearchRequest
                      │
                      ▼
              Flight Service
                      │
                      ▼
            Mock Flight Provider
                      │
              ┌───────┴────────┐
              ▼                ▼
        Outbound flights   Return flights
              │                │
              └───────┬────────┘
                      ▼
             FlightSearchResponse
                      │
                      ▼
                    USER
```

---

# 21. How this connects to Phase 3

Eventually, these two phases connect:

```text
User
 │
 ├── Destination: Goa
 ├── Duration: 4 days
 └── Origin: Mumbai
 │
 ▼
Weather Decision Engine
 │
 ▼
Oct 5 → Oct 8
 │
 ▼
Flight Search
 │
 ├── Mumbai → Goa
 │     Oct 5
 │
 └── Goa → Mumbai
       Oct 8
```

That connection is exactly what we'll use later when we build the **overall travel-planning workflow**.

For now, we're keeping each capability independently testable.

---

# 22. Phase 4 Definition of Done

Phase 4 is complete when all of these work:

```text
✅ Flight schema created
✅ Flight search request validation
✅ Return-date validation
✅ Mock flight provider
✅ Outbound flight search
✅ Return flight search
✅ Passenger handling
✅ Maximum stops filter
✅ Maximum price filter
✅ Flight service
✅ Flight tool
✅ Flight API route
✅ Router registered
✅ One-way search works
✅ Round-trip search works
✅ Flight tests pass
```

Your architecture is now:

```text
PHASE 0 ✅
Infrastructure
      ↓
PHASE 1 ✅
Trip + PostgreSQL
      ↓
PHASE 2 ✅
Weather Integration
      ↓
PHASE 3 ✅
Weather Decision Engine
      ↓
PHASE 4
Flight Search  ← YOU ARE HERE
      ↓
PHASE 5
Hotel Search
      ↓
PHASE 6
Cab Search
      ↓
PHASE 7+
Combine capabilities
```

### One important rule for this phase

**Keep the mock provider exactly as a mock.** Don't start adding real flight API credentials or provider-specific code yet. We want the complete travel-agent architecture established first; then replacing the mock with a real provider becomes a controlled change instead of mixing API integration with the rest of the system.
