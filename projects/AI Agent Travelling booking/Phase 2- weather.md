Yes. Let's make this clean and complete.

Below is the **full weather implementation for Phase 2 + Phase 3**, with every file you need, including the files that were added, modified, and tested.

The weather provider used here is Open-Meteo. Its current forecast API supports coordinate-based forecasts, daily variables including maximum/minimum temperature, precipitation probability, precipitation sum, and weather code, with forecasts available up to 16 days; its geocoding API accepts a location name and returns coordinates/timezone. [Open-Meteo](https://open-meteo.com/en/docs?past_days=1\&utm_source=chatgpt.com)

---

# 1. Final folder structure

Your project should look like this after completing these phases:

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
│   │       └── travel_dates.py
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
│   │   └── weather_decision.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── trip_service.py
│   │   ├── geocoding_service.py
│   │   ├── weather_service.py
│   │   └── weather_decision_service.py
│   │
│   └── tools/
│       ├── __init__.py
│       └── weather_tools.py
│
├── migrations/
│
├── tests/
│   ├── test_health.py
│   ├── test_trip.py
│   ├── test_weather.py
│   └── test_weather_decision.py
│
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── venv/
```

---

# 2. Install dependency

You need `httpx` for calling the weather APIs.

```powershell
pip install httpx
```

Then update:

```powershell
pip freeze > requirements.txt
```

---

# 3. `app/schemas/weather.py`

This defines the **weather data contract** used by the rest of our application.

```python
from datetime import date

from pydantic import BaseModel


class WeatherDay(BaseModel):
    date: date
    temperature_max: float
    temperature_min: float
    precipitation_probability: float
    precipitation_sum: float
    weather_code: int


class WeatherResponse(BaseModel):
    destination: str
    latitude: float
    longitude: float
    timezone: str
    forecast: list[WeatherDay]
```

### What goes in

Data from the weather provider.

### What comes out

Our own standardized weather structure.

---

# 4. `app/services/geocoding_service.py`

The weather API needs latitude and longitude, so first we convert:

```text
Goa
 ↓
15.3, 74.1
```

Create:

```text
app/services/geocoding_service.py
```

```python
import httpx


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


async def get_coordinates(city: str) -> dict:
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            GEOCODING_URL,
            params=params,
        )

        response.raise_for_status()

        data = response.json()

    results = data.get("results", [])

    if not results:
        raise ValueError(
            f"Could not find location: {city}"
        )

    location = results[0]

    return {
        "name": location["name"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": location.get("timezone", "UTC"),
    }
```

Open-Meteo's geocoding endpoint accepts `name`, `count`, `language`, and `format`, and returns latitude, longitude, timezone and location metadata. [Open-Meteo](https://open-meteo.com/en/docs/geocoding-api?utm_source=chatgpt.com)

---

# 5. `app/services/weather_service.py`

This is the main weather integration.

Create:

```text
app/services/weather_service.py
```

```python
import httpx

from app.schemas.weather import (
    WeatherDay,
    WeatherResponse,
)

from app.services.geocoding_service import (
    get_coordinates,
)


WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


async def get_weather(
    destination: str,
    forecast_days: int = 7,
) -> WeatherResponse:

    # ----------------------------------------
    # Step 1: Convert city → coordinates
    # ----------------------------------------

    location = await get_coordinates(
        destination
    )

    # ----------------------------------------
    # Step 2: Prepare weather API parameters
    # ----------------------------------------

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
                "precipitation_sum",
                "weather_code",
            ]
        ),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    # ----------------------------------------
    # Step 3: Call weather API
    # ----------------------------------------

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            WEATHER_URL,
            params=params,
        )

        response.raise_for_status()

        data = response.json()

    # ----------------------------------------
    # Step 4: Extract daily data
    # ----------------------------------------

    daily = data["daily"]

    forecast = []

    # ----------------------------------------
    # Step 5: Normalize provider response
    # ----------------------------------------

    for index, forecast_date in enumerate(
        daily["time"]
    ):
        forecast.append(
            WeatherDay(
                date=forecast_date,
                temperature_max=(
                    daily["temperature_2m_max"][index]
                ),
                temperature_min=(
                    daily["temperature_2m_min"][index]
                ),
                precipitation_probability=(
                    daily[
                        "precipitation_probability_max"
                    ][index]
                ),
                precipitation_sum=(
                    daily["precipitation_sum"][index]
                ),
                weather_code=(
                    daily["weather_code"][index]
                ),
            )
        )

    # ----------------------------------------
    # Step 6: Return our application schema
    # ----------------------------------------

    return WeatherResponse(
        destination=location["name"],
        latitude=location["latitude"],
        longitude=location["longitude"],
        timezone=location["timezone"],
        forecast=forecast,
    )
```

The forecast endpoint supports the daily variables used above and `forecast_days`; Open-Meteo currently documents up to 16 forecast days for this endpoint. [Open-Meteo](https://open-meteo.com/en/docs?past_days=1\&utm_source=chatgpt.com)

---

# 6. `app/api/routes/weather.py`

This exposes the weather functionality as an HTTP endpoint.

Create:

```text
app/api/routes/weather.py
```

```python
from fastapi import APIRouter, HTTPException, Query

from app.schemas.weather import WeatherResponse

from app.services.weather_service import (
    get_weather,
)


router = APIRouter(
    prefix="/weather",
    tags=["Weather"],
)


@router.get(
    "/{destination}",
    response_model=WeatherResponse,
)
async def weather(
    destination: str,
    forecast_days: int = Query(
        default=7,
        ge=1,
        le=16,
    ),
):

    try:
        return await get_weather(
            destination=destination,
            forecast_days=forecast_days,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail=(
                "Weather service is currently "
                "unavailable"
            ),
        )
```

---

# 7. `app/tools/weather_tools.py`

This is the interface we'll eventually expose to the AI agent.

Create:

```text
app/tools/weather_tools.py
```

```python
from app.schemas.weather import WeatherResponse

from app.schemas.weather_decision import (
    TravelWindowResponse,
)

from app.services.weather_service import (
    get_weather,
)

from app.services.weather_decision_service import (
    find_best_travel_windows,
)


async def weather_tool(
    destination: str,
    forecast_days: int = 7,
) -> WeatherResponse:

    return await get_weather(
        destination=destination,
        forecast_days=forecast_days,
    )


async def best_travel_dates_tool(
    destination: str,
    duration: int,
    forecast_days: int = 16,
) -> TravelWindowResponse:

    return await find_best_travel_windows(
        destination=destination,
        duration=duration,
        forecast_days=forecast_days,
    )
```

Think of this as:

```text
Agent
  │
  ├── weather_tool()
  │
  └── best_travel_dates_tool()
```

The agent does not need to know anything about:

```text
httpx
Open-Meteo
URLs
JSON mapping
```

It only sees the tool.

---

# 8. `app/schemas/weather_decision.py`

Now we define the data for the **travel-date decision engine**.

Create:

```text
app/schemas/weather_decision.py
```

```python
from datetime import date

from pydantic import BaseModel, Field


class TravelWindowRequest(BaseModel):
    destination: str = Field(
        min_length=2,
        max_length=100,
    )

    duration: int = Field(
        gt=0,
        le=16,
    )

    forecast_days: int = Field(
        default=16,
        ge=1,
        le=16,
    )


class TravelWindow(BaseModel):
    start_date: date
    end_date: date

    score: float

    average_temperature: float
    average_rain_probability: float
    total_precipitation: float


class TravelWindowResponse(BaseModel):
    destination: str
    duration: int

    recommended_window: TravelWindow | None

    alternatives: list[TravelWindow]
```

---

# 9. `app/services/weather_decision_service.py`

This is the **brain of Phase 3**.

Create:

```text
app/services/weather_decision_service.py
```

```python
from app.schemas.weather_decision import (
    TravelWindow,
    TravelWindowResponse,
)

from app.services.weather_service import (
    get_weather,
)


def calculate_weather_score(
    average_temperature: float,
    average_rain_probability: float,
    total_precipitation: float,
) -> float:

    # ----------------------------------------
    # Temperature score
    # Target temperature = 28°C
    # ----------------------------------------

    temperature_score = max(
        0,
        100
        - abs(
            28 - average_temperature
        ) * 10,
    )

    # ----------------------------------------
    # Rain probability score
    # ----------------------------------------

    rain_probability_score = max(
        0,
        100 - average_rain_probability,
    )

    # ----------------------------------------
    # Precipitation score
    # ----------------------------------------

    precipitation_score = max(
        0,
        100 - (
            total_precipitation * 5
        ),
    )

    # ----------------------------------------
    # Weighted final score
    # ----------------------------------------

    score = (
        temperature_score * 0.40
        + rain_probability_score * 0.40
        + precipitation_score * 0.20
    )

    return round(score, 2)


async def find_best_travel_windows(
    destination: str,
    duration: int,
    forecast_days: int = 16,
) -> TravelWindowResponse:

    # ----------------------------------------
    # Validate forecast length
    # ----------------------------------------

    if forecast_days < duration:
        raise ValueError(
            "forecast_days must be greater than "
            "or equal to duration"
        )

    # ----------------------------------------
    # Get weather forecast
    # ----------------------------------------

    weather = await get_weather(
        destination=destination,
        forecast_days=forecast_days,
    )

    forecast = weather.forecast

    # ----------------------------------------
    # Safety check
    # ----------------------------------------

    if len(forecast) < duration:
        return TravelWindowResponse(
            destination=weather.destination,
            duration=duration,
            recommended_window=None,
            alternatives=[],
        )

    windows = []

    # ----------------------------------------
    # Sliding window
    # ----------------------------------------

    for start_index in range(
        len(forecast) - duration + 1
    ):

        window = forecast[
            start_index:start_index + duration
        ]

        # ----------------------------------------
        # Average temperature
        # ----------------------------------------

        average_temperature = (
            sum(
                day.temperature_max
                for day in window
            )
            / duration
        )

        # ----------------------------------------
        # Average rain probability
        # ----------------------------------------

        average_rain_probability = (
            sum(
                day.precipitation_probability
                for day in window
            )
            / duration
        )

        # ----------------------------------------
        # Total precipitation
        # ----------------------------------------

        total_precipitation = sum(
            day.precipitation_sum
            for day in window
        )

        # ----------------------------------------
        # Calculate score
        # ----------------------------------------

        score = calculate_weather_score(
            average_temperature=average_temperature,
            average_rain_probability=(
                average_rain_probability
            ),
            total_precipitation=(
                total_precipitation
            ),
        )

        # ----------------------------------------
        # Create travel window
        # ----------------------------------------

        travel_window = TravelWindow(
            start_date=window[0].date,
            end_date=window[-1].date,
            score=score,
            average_temperature=round(
                average_temperature,
                2,
            ),
            average_rain_probability=round(
                average_rain_probability,
                2,
            ),
            total_precipitation=round(
                total_precipitation,
                2,
            ),
        )

        windows.append(
            travel_window
        )

    # ----------------------------------------
    # Rank windows
    # ----------------------------------------

    windows.sort(
        key=lambda window: window.score,
        reverse=True,
    )

    # ----------------------------------------
    # Best window
    # ----------------------------------------

    recommended_window = windows[0]

    # ----------------------------------------
    # Alternative windows
    # ----------------------------------------

    alternatives = windows[1:4]

    # ----------------------------------------
    # Return final result
    # ----------------------------------------

    return TravelWindowResponse(
        destination=weather.destination,
        duration=duration,
        recommended_window=recommended_window,
        alternatives=alternatives,
    )
```

---

# 10. Understand this service

The complete processing is:

```text
Goa
 │
 ▼
get_weather()
 │
 ▼
16-day forecast
 │
 ▼
Duration = 4
 │
 ▼
Create 4-day windows
 │
 ├── Day 1 → Day 4
 ├── Day 2 → Day 5
 ├── Day 3 → Day 6
 ├── Day 4 → Day 7
 ├── ...
 └── Day 13 → Day 16
 │
 ▼
Calculate each score
 │
 ▼
Sort by score
 │
 ▼
Best window + alternatives
```

---

# 11. `app/api/routes/travel_dates.py`

Now expose the decision engine through an API.

Create:

```text
app/api/routes/travel_dates.py
```

```python
from fastapi import APIRouter, HTTPException

from app.schemas.weather_decision import (
    TravelWindowRequest,
    TravelWindowResponse,
)

from app.services.weather_decision_service import (
    find_best_travel_windows,
)


router = APIRouter(
    prefix="/travel",
    tags=["Travel"],
)


@router.post(
    "/best-dates",
    response_model=TravelWindowResponse,
)
async def find_best_dates(
    request: TravelWindowRequest,
):

    try:
        return await find_best_travel_windows(
            destination=request.destination,
            duration=request.duration,
            forecast_days=request.forecast_days,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to calculate "
                "travel dates"
            ),
        )
```

---

# 12. `app/main.py`

This is an **updated existing file**, not a new file.

Make sure it contains:

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


app = FastAPI(
    title="Travel Agent API",
    version="0.1.0",
)


# ----------------------------------------
# Routers
# ----------------------------------------

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


@app.get("/")
def root():
    return {
        "message": "Travel Agent API is running"
    }
```

---

# 13. `app/services/__init__.py`

This can be empty:

```python
```

---

# 14. `app/tools/__init__.py`

Also empty:

```python
```

---

# 15. `app/schemas/__init__.py`

Empty is fine:

```python
```

---

# 16. `app/api/routes/__init__.py`

Empty:

```python
```

---

# 17. Weather API test

Create:

```text
tests/test_weather.py
```

For a proper unit/integration test, avoid depending on the live weather API.

We'll mock the weather service.

```python
from datetime import date

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.weather import (
    WeatherDay,
    WeatherResponse,
)


client = TestClient(app)


def test_weather_endpoint(monkeypatch):

    async def mock_get_weather(
        destination: str,
        forecast_days: int = 7,
    ):

        return WeatherResponse(
            destination=destination,
            latitude=15.49,
            longitude=73.82,
            timezone="Asia/Kolkata",
            forecast=[
                WeatherDay(
                    date=date(2026, 10, 2),
                    temperature_max=31.0,
                    temperature_min=25.0,
                    precipitation_probability=20,
                    precipitation_sum=0.5,
                    weather_code=2,
                )
            ],
        )

    monkeypatch.setattr(
        "app.api.routes.weather.get_weather",
        mock_get_weather,
    )

    response = client.get(
        "/weather/Goa"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["destination"] == "Goa"
    assert "latitude" in data
    assert "longitude" in data
    assert "timezone" in data
    assert "forecast" in data
    assert len(data["forecast"]) == 1
```

---

# 18. Decision engine test

Create:

```text
tests/test_weather_decision.py
```

```python
from datetime import date

from app.schemas.weather import (
    WeatherDay,
    WeatherResponse,
)

from app.services import (
    weather_decision_service,
)


def test_weather_score():

    score = (
        weather_decision_service
        .calculate_weather_score(
            average_temperature=28,
            average_rain_probability=10,
            total_precipitation=0,
        )
    )

    assert score == 96


def test_find_best_travel_windows(
    monkeypatch,
):

    async def mock_get_weather(
        destination: str,
        forecast_days: int = 16,
    ):

        return WeatherResponse(
            destination=destination,
            latitude=15.49,
            longitude=73.82,
            timezone="Asia/Kolkata",
            forecast=[
                WeatherDay(
                    date=date(2026, 10, 2),
                    temperature_max=28,
                    temperature_min=24,
                    precipitation_probability=10,
                    precipitation_sum=0,
                    weather_code=1,
                ),
                WeatherDay(
                    date=date(2026, 10, 3),
                    temperature_max=29,
                    temperature_min=24,
                    precipitation_probability=10,
                    precipitation_sum=0,
                    weather_code=1,
                ),
                WeatherDay(
                    date=date(2026, 10, 4),
                    temperature_max=28,
                    temperature_min=24,
                    precipitation_probability=10,
                    precipitation_sum=0,
                    weather_code=1,
                ),
                WeatherDay(
                    date=date(2026, 10, 5),
                    temperature_max=27,
                    temperature_min=23,
                    precipitation_probability=10,
                    precipitation_sum=0,
                    weather_code=1,
                ),
                WeatherDay(
                    date=date(2026, 10, 6),
                    temperature_max=35,
                    temperature_min=28,
                    precipitation_probability=80,
                    precipitation_sum=15,
                    weather_code=63,
                ),
            ],
        )

    monkeypatch.setattr(
        weather_decision_service,
        "get_weather",
        mock_get_weather,
    )

    result = (
        __import__(
            "asyncio"
        ).run(
            weather_decision_service
            .find_best_travel_windows(
                destination="Goa",
                duration=4,
                forecast_days=5,
            )
        )
    )

    assert result.destination == "Goa"

    assert result.duration == 4

    assert (
        result.recommended_window
        is not None
    )

    assert (
        result.recommended_window.start_date
        == date(2026, 10, 2)
    )

    assert (
        result.recommended_window.end_date
        == date(2026, 10, 5)
    )
```

There is a cleaner async-test version later when we introduce `pytest-asyncio`; for now this keeps the dependency set smaller.

---

# 19. You already have these files from previous phases

Don't recreate these if you already have them:

```text
app/database/base.py
app/database/session.py
app/models/trip.py
app/models/__init__.py
app/schemas/trip.py
app/services/trip_service.py
app/api/routes/travel.py
app/api/routes/health.py
app/core/config.py
```

Phase 3 only adds the weather/decision functionality.

---

# 20. Complete Phase 3 flow

Now the entire feature works like this:

```text
                 USER
                   │
                   │
                   │ "Goa"
                   │ "4 days"
                   ▼
          POST /travel/best-dates
                   │
                   ▼
       TravelWindowRequest
                   │
                   ▼
     weather_decision_service
                   │
                   ▼
            get_weather()
                   │
                   ▼
        geocoding_service
                   │
                   ▼
        Open-Meteo Geocoding
                   │
                   ▼
            coordinates
                   │
                   ▼
          Open-Meteo Forecast
                   │
                   ▼
          WeatherResponse
                   │
                   ▼
          Sliding Window
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      4 days     4 days     4 days
      window     window     window
        │          │          │
        ▼          ▼          ▼
      Score      Score      Score
        │          │          │
        └──────────┼──────────┘
                   ▼
              Sort scores
                   │
                   ▼
          Recommended window
                   │
                   ▼
              Alternatives
                   │
                   ▼
                  USER
```

---

# 21. Two APIs now exist

### Weather API

```http
GET /weather/Goa
```

This answers:

> What is the weather forecast?

---

### Best dates API

```http
POST /travel/best-dates
```

Request:

```json
{
    "destination": "Goa",
    "duration": 4
}
```

This answers:

> Given this forecast, which consecutive dates should the application evaluate as the best-weather windows according to our scoring rules?

---

# 22. What each layer is responsible for

| File | Responsibility |
|---|---|
| `schemas/weather.py` | Weather data structure |
| `geocoding_service.py` | City → coordinates |
| `weather_service.py` | Call weather provider + normalize response |
| `weather.py` | HTTP weather endpoint |
| `schemas/weather_decision.py` | Decision input/output structures |
| `weather_decision_service.py` | Sliding window + scoring + ranking |
| `travel_dates.py` | HTTP endpoint for best dates |
| `weather_tools.py` | Agent-facing weather capabilities |
| `main.py` | Register routes |

The key architecture is:

```text
Route
  ↓
Service
  ↓
External API
```

and for decision making:

```text
Route
  ↓
Decision Service
  ↓
Weather Service
  ↓
External API
```

Later, the agent will use:

```text
Agent
  ↓
Tool
  ↓
Service
  ↓
External API
```

---

# 23. Test everything

Start PostgreSQL:

```powershell
docker compose up -d
```

Activate environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
uvicorn app.main:app --reload
```

Run tests:

```powershell
pytest
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
```

---

# 24. Test `/weather/{destination}`

```http
GET http://127.0.0.1:8000/weather/Goa
```

Or:

```http
GET http://127.0.0.1:8000/weather/Goa?forecast_days=16
```

---

# 25. Test `/travel/best-dates`

```http
POST http://127.0.0.1:8000/travel/best-dates
```

Body:

```json
{
    "destination": "Goa",
    "duration": 4,
    "forecast_days": 16
}
```

Expected structure:

```json
{
    "destination": "Goa",
    "duration": 4,
    "recommended_window": {
        "start_date": "2026-10-02",
        "end_date": "2026-10-05",
        "score": 90.25,
        "average_temperature": 28.5,
        "average_rain_probability": 12.5,
        "total_precipitation": 0.8
    },
    "alternatives": []
}
```

The actual dates and weather values will change with the live forecast.

---

# 26. One important architectural point

Our current score is a **project-defined heuristic**:

```text
40% → temperature
40% → rain probability
20% → precipitation
```

It is **not** a universal weather-quality standard. We are deliberately keeping the first version simple so that you understand the decision engine before introducing more sophisticated rules.

Later we can improve it with things such as:

```text
Weather code
Wind speed
Humidity
Feels-like temperature
Sunshine duration
Activities
Destination type
User preferences
```

But **do not add those yet**.

---

# Phase 3 Definition of Done

You are finished with this phase when these all work:

```text
✅ /weather/Goa
✅ Geocoding
✅ Weather API
✅ Weather normalization
✅ Weather schema
✅ Weather service
✅ Weather tool

✅ /travel/best-dates
✅ Travel window schema
✅ Sliding window algorithm
✅ Temperature scoring
✅ Rain scoring
✅ Precipitation scoring
✅ Ranking
✅ Recommended window
✅ Alternative windows
✅ Tests
```

At that point, the backend has completed:

```text
PHASE 0
Project Setup
      ↓
PHASE 1
Trip API + PostgreSQL
      ↓
PHASE 2
Weather Integration
      ↓
PHASE 3
Weather Decision Engine
      ↓
PHASE 4
Flight Search
```

**Do not move to Flight Search until `/weather/Goa` and `/travel/best-dates` both work and `pytest` passes.**
