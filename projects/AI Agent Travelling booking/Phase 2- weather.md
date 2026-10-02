# Phase 3 — Weather Decision Engine

Now that **Phase 2 gives us a weather forecast**, the next step is to make the backend **reason over that forecast** and find a good consecutive travel window.

We are **not using an LLM or LangGraph yet**.

The flow is:

```text
User
 │
 │ destination = Goa
 │ duration = 4 days
 ▼
Weather Decision Engine
 │
 ▼
Get 7/16-day forecast
 │
 ▼
Evaluate every 4-day window
 │
 ├── Rain probability
 ├── Rain amount
 ├── Temperature
 └── Weather conditions
 │
 ▼
Calculate score
 │
 ▼
Rank windows
 │
 ▼
Recommended travel dates
```

---

# 1. What are we building?

Suppose the forecast is:

```text
Oct 2   Rain 20%
Oct 3   Rain 15%
Oct 4   Rain 10%
Oct 5   Rain 15%
Oct 6   Rain 80%
Oct 7   Rain 70%
Oct 8   Rain 20%
```

The user wants:

```text
Trip duration = 4 days
```

The engine evaluates:

```text
Window 1
Oct 2 → Oct 5
Score = 88

Window 2
Oct 3 → Oct 6
Score = 65

Window 3
Oct 4 → Oct 7
Score = 48

Window 4
Oct 5 → Oct 8
Score = 55
```

Then returns:

```text
Recommended:
Oct 2 → Oct 5

Score:
88
```

The important part is that **we are not just looking at one day**.

We need to find a **consecutive window matching the trip duration**.

---

# 2. New architecture

We'll add:

```text
app/
├── schemas/
│   ├── trip.py
│   └── weather.py
│
├── services/
│   ├── weather_service.py
│   ├── geocoding_service.py
│   └── weather_decision_service.py
│
├── tools/
│   └── weather_tools.py
│
└── api/
    └── routes/
        ├── travel.py
        └── weather.py
```

The new file is:

```text
app/services/weather_decision_service.py
```

And we'll add schemas for the decision engine.

---

# 3. Step 1 — Create the decision schemas

Create:

```text
app/schemas/weather_decision.py
```

```python
from datetime import date
from pydantic import BaseModel, Field


class TravelWindowRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=100)
    duration: int = Field(gt=0, le=16)
    forecast_days: int = Field(default=16, ge=1, le=16)


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

# 4. Understand the schemas

## `TravelWindowRequest`

This is what the user sends:

```json
{
    "destination": "Goa",
    "duration": 4
}
```

It means:

```text
Destination → Goa
Trip length → 4 days
```

---

## `TravelWindow`

This represents one possible trip.

Example:

```json
{
    "start_date": "2026-10-02",
    "end_date": "2026-10-05",
    "score": 87.5,
    "average_temperature": 30.2,
    "average_rain_probability": 18.5,
    "total_precipitation": 1.2
}
```

---

## `TravelWindowResponse`

This gives us:

```text
Recommended window
+
Alternative windows
```

This will become useful later when the **LLM/agent needs choices**.

---

# 5. Step 2 — Create the decision engine

Create:

```text
app/services/weather_decision_service.py
```

```python
from datetime import date
from app.schemas.weather_decision import (
    TravelWindow,
    TravelWindowResponse,
)
from app.services.weather_service import get_weather


def calculate_weather_score(
    average_temperature: float,
    average_rain_probability: float,
    total_precipitation: float,
) -> float:

    temperature_score = max(
        0,
        100 - abs(28 - average_temperature) * 10
    )

    rain_probability_score = max(
        0,
        100 - average_rain_probability
    )

    precipitation_score = max(
        0,
        100 - (total_precipitation * 5)
    )

    score = (
        temperature_score * 0.4
        + rain_probability_score * 0.4
        + precipitation_score * 0.2
    )

    return round(score, 2)


async def find_best_travel_windows(
    destination: str,
    duration: int,
    forecast_days: int = 16,
) -> TravelWindowResponse:

    weather = await get_weather(
        destination=destination,
        forecast_days=forecast_days,
    )

    forecast = weather.forecast

    if len(forecast) < duration:
        return TravelWindowResponse(
            destination=weather.destination,
            duration=duration,
            recommended_window=None,
            alternatives=[],
        )

    windows = []

    for start_index in range(
        len(forecast) - duration + 1
    ):

        window = forecast[
            start_index:start_index + duration
        ]

        average_temperature = sum(
            day.temperature_max
            for day in window
        ) / duration

        average_rain_probability = sum(
            day.precipitation_probability
            for day in window
        ) / duration

        total_precipitation = sum(
            day.precipitation_sum
            for day in window
        )

        score = calculate_weather_score(
            average_temperature=average_temperature,
            average_rain_probability=average_rain_probability,
            total_precipitation=total_precipitation,
        )

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

        windows.append(travel_window)

    windows.sort(
        key=lambda window: window.score,
        reverse=True,
    )

    recommended_window = windows[0]

    alternatives = windows[1:4]

    return TravelWindowResponse(
        destination=weather.destination,
        duration=duration,
        recommended_window=recommended_window,
        alternatives=alternatives,
    )
```

---

# 6. What is happening here?

This is the most important part of Phase 3.

Suppose we have:

```text
10 days of forecast
```

and:

```text
duration = 4
```

The engine creates:

```text
Window 1
Day 1 → Day 4

Window 2
Day 2 → Day 5

Window 3
Day 3 → Day 6

Window 4
Day 4 → Day 7

Window 5
Day 5 → Day 8

Window 6
Day 6 → Day 9

Window 7
Day 7 → Day 10
```

This is a **sliding window** algorithm.

---

# 7. Sliding window

The core code is:

```python
for start_index in range(
    len(forecast) - duration + 1
):
```

For:

```text
forecast = 10 days
duration = 4
```

we get:

```text
10 - 4 + 1
= 7 windows
```

So:

```text
┌─────────────────────┐
│ Day 1 Day 2 Day 3 Day 4 │
└─────────────────────┘

   ┌─────────────────────┐
   │ Day 2 Day 3 Day 4 Day 5 │
   └─────────────────────┘

      ┌─────────────────────┐
      │ Day 3 Day 4 Day 5 Day 6 │
      └─────────────────────┘
```

This is a very common algorithmic pattern.

---

# 8. Step 3 — Weather scoring

We need to convert weather into a numerical score.

Currently we use three factors:

```text
Temperature       → 40%
Rain probability  → 40%
Rain amount       → 20%
```

Therefore:

```text
Final Score =
    Temperature Score × 0.40
  + Rain Probability Score × 0.40
  + Precipitation Score × 0.20
```

---

# 9. Temperature score

We currently consider around:

```text
28°C
```

as the target temperature.

The formula:

```python
temperature_score = max(
    0,
    100 - abs(28 - average_temperature) * 10
)
```

For example:

```text
Temperature = 28°C

difference = 0

score = 100
```

But:

```text
Temperature = 30°C

difference = 2

score = 80
```

And:

```text
Temperature = 35°C

difference = 7

score = 30
```

This is intentionally a **simple first version**.

Later we can make this destination-specific.

---

# 10. Rain probability score

We use:

```python
rain_probability_score = max(
    0,
    100 - average_rain_probability
)
```

So:

| Rain probability | Score |
|---:|---:|
| 0% | 100 |
| 10% | 90 |
| 20% | 80 |
| 50% | 50 |
| 80% | 20 |
| 100% | 0 |

---

# 11. Precipitation score

We also consider actual expected precipitation:

```python
precipitation_score = max(
    0,
    100 - (total_precipitation * 5)
)
```

This prevents the engine from looking only at probability.

For example:

```text
Rain probability = 20%
Rain amount = 20 mm
```

is very different from:

```text
Rain probability = 20%
Rain amount = 0.2 mm
```

---

# 12. Step 4 — Create the API route

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
            status_code=404,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to calculate travel dates",
        )
```

---

# 13. Register the route

In:

```text
app/main.py
```

add:

```python
from app.api.routes.travel_dates import router as travel_dates_router
```

and:

```python
app.include_router(travel_dates_router)
```

Your `main.py` will now have multiple routers:

```python
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.travel import router as travel_router
from app.api.routes.weather import router as weather_router
from app.api.routes.travel_dates import router as travel_dates_router


app = FastAPI(
    title="Travel Agent API",
    version="0.1.0",
)


app.include_router(health_router)
app.include_router(travel_router)
app.include_router(weather_router)
app.include_router(travel_dates_router)


@app.get("/")
def root():
    return {
        "message": "Travel Agent API is running"
    }
```

---

# 14. Test the API

Start the server:

```powershell
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

You should see:

```text
POST /travel/best-dates
```

Send:

```json
{
    "destination": "Goa",
    "duration": 4
}
```

---

# 15. Expected response

Something like:

```json
{
    "destination": "Goa",
    "duration": 4,
    "recommended_window": {
        "start_date": "2026-10-04",
        "end_date": "2026-10-07",
        "score": 82.41,
        "average_temperature": 29.7,
        "average_rain_probability": 17.5,
        "total_precipitation": 2.4
    },
    "alternatives": [
        {
            "start_date": "2026-10-05",
            "end_date": "2026-10-08",
            "score": 79.21,
            "average_temperature": 29.5,
            "average_rain_probability": 21.2,
            "total_precipitation": 3.1
        }
    ]
}
```

The exact values will depend on the live forecast.

---

# 16. Step 5 — Add the weather decision tool

We already created:

```text
app/tools/weather_tools.py
```

Now extend it.

```python
from app.schemas.weather import WeatherResponse
from app.schemas.weather_decision import TravelWindowResponse

from app.services.weather_service import get_weather
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

Now we have **two different tools**:

```text
weather_tool
     ↓
Raw/normalized forecast


best_travel_dates_tool
     ↓
Weather forecast
     ↓
Decision engine
     ↓
Recommended travel window
```

This distinction becomes important later when we introduce the agent.

---

# 17. Why do we need both Service and Tool?

This architecture is intentional.

### Service

```text
weather_service.py
```

Answers:

> How does my application communicate with the weather provider?

---

### Decision service

```text
weather_decision_service.py
```

Answers:

> How does my application decide which dates are better?

---

### Tool

```text
weather_tools.py
```

Answers:

> What capability can an agent invoke?

Later:

```text
LLM Agent
    │
    ├── weather_tool()
    │
    ├── flight_tool()
    │
    ├── hotel_tool()
    │
    └── cab_tool()
```

That's why we are creating the tool layer **before** introducing the actual agent.

---

# 18. Add tests

Create:

```text
tests/test_weather_decision.py
```

```python
from datetime import date

from app.services.weather_decision_service import (
    calculate_weather_score,
)


def test_weather_score():

    score = calculate_weather_score(
        average_temperature=28,
        average_rain_probability=10,
        total_precipitation=0,
    )

    assert score == 96
```

Why `96`?

```text
Temperature score = 100

Rain score = 90

Precipitation score = 100


Final:

100 × 0.40
+ 90 × 0.40
+ 100 × 0.20

= 40 + 36 + 20

= 96
```

Run:

```powershell
pytest
```

---

# 19. Important: What we have built

At this point our architecture is becoming:

```text
                         ┌──────────────┐
                         │   FastAPI    │
                         └──────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              /weather/Goa          /travel/best-dates
                    │                       │
                    ▼                       ▼
             Weather Service        Decision Service
                    │                       │
                    ▼                       ▼
              Geocoding API          Weather Service
                    │                       │
                    ▼                       ▼
              Weather API             Forecast
                                            │
                                            ▼
                                      Sliding Window
                                            │
                                            ▼
                                        Scoring
                                            │
                                            ▼
                                      Ranked Dates
```

---

# 20. Current project structure

After Phase 3:

```text
BACKEND/
│
├── app/
│   ├── __init__.py
│   │
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

# 21. Phase 3 flow — remember this

The complete flow is:

```text
                    USER
                     │
                     │
             "Goa for 4 days"
                     │
                     ▼
              FastAPI Route
                     │
                     ▼
        TravelWindowRequest
                     │
                     ▼
       Weather Decision Service
                     │
                     ▼
             Weather Service
                     │
                     ▼
             Geocoding Service
                     │
                     ▼
              Weather Provider
                     │
                     ▼
               16-day forecast
                     │
                     ▼
             Sliding Windows
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Window 1   Window 2   Window 3 ...
          │          │          │
          ▼          ▼          ▼
        Score      Score      Score
          │          │          │
          └──────────┼──────────┘
                     ▼
                  SORT
                     │
                     ▼
          Recommended + Alternatives
                     │
                     ▼
                   USER
```

---

# 22. What Phase 3 teaches you

This phase is important because you're now moving from a simple API to **decision-making logic**.

You have learned:

- **Sliding window algorithm**
- **Feature aggregation**
- **Scoring functions**
- **Ranking**
- **Business rules**
- **Service separation**
- **Tool abstraction**
- **Structured decision output**

And importantly:

```text
Weather API
     ↓
Data
     ↓
Decision Engine
     ↓
Recommendation
```

There is still **no AI involved**.

That's deliberate.

---

# 23. Definition of Done — Phase 3

Before moving to Phase 4, verify:

- [ ] `weather_decision.py` created
- [ ] `weather_decision_service.py` created
- [ ] Sliding-window logic works
- [ ] Temperature score works
- [ ] Rain probability score works
- [ ] Precipitation score works
- [ ] Windows are ranked
- [ ] `/travel/best-dates` works
- [ ] Alternatives are returned
- [ ] `best_travel_dates_tool()` exists
- [ ] Tests pass
- [ ] Existing `/weather/Goa` still works
- [ ] Existing `/travel/plan` still works

### After this, Phase 4 is **Flight Search**:

```text
Recommended Weather Dates
          │
          ▼
    Flight Search
          │
          ▼
Origin → Destination
          │
          ▼
Available Flights
          │
          ▼
Price + Duration + Stops
          │
          ▼
Rank Flights
```

We'll initially use a **mock flight provider**, then replace it with a real flight API.
