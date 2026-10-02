# Phase 5 — Hotel Search

We'll now add the **hotel/accommodation search capability**.

For this phase, I'm changing the earlier plan slightly: `randomapi.dev` currently has a generic **room/booking fixture API**, but not a hotel-search endpoint. For a travel agent, that's not enough because we need properties, dates, occupancy, ratings, amenities, and pricing. StayingAPI currently provides a dedicated `/v1/search` accommodation endpoint with a deterministic `stay_test_...` sandbox mode that costs 0 credits. It accepts location, dates, occupancy, property type, price/rating filters, platforms, and limit, and returns a normalized property schema with location, rating, amenities, and price. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

So Phase 5 will use:

```text
StayingAPI
    ↓
stay_test_... sandbox
```

The sandbox is deterministic and free, while the live mode requires a live key/credits. [StayingAPI](https://stayingapi.com/docs/try-it?utm_source=chatgpt.com)

---

# 1. Phase 5 goal

We want:

```text
                Hotel Search Request
                        │
                        ▼
                  FastAPI Route
                        │
                        ▼
                HotelSearchRequest
                        │
                        ▼
                  Hotel Service
                        │
                        ▼
                    HTTP GET
                        │
                        ▼
                    StayingAPI
                        │
                        ▼
                 Provider response
                        │
                        ▼
                   Normalize
                        │
                        ▼
               HotelSearchResponse
                        │
                        ▼
                      User
```

---

# 2. Final files for Phase 5

### New files

```text
app/
├── api/
│   └── routes/
│       └── hotels.py
│
├── schemas/
│   └── hotel.py
│
├── services/
│   └── hotel_service.py
│
└── tools/
    └── hotel_tools.py

tests/
└── test_hotel.py
```

### Modified files

```text
app/main.py
app/core/config.py
.env
.env.example
```

No database migration is required.

---

# 3. Final relevant structure

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
│   │       ├── flights.py
│   │       └── hotels.py              ← NEW
│   │
│   ├── core/
│   │   └── config.py                  ← MODIFIED
│   │
│   ├── schemas/
│   │   ├── trip.py
│   │   ├── weather.py
│   │   ├── weather_decision.py
│   │   ├── flight.py
│   │   └── hotel.py                   ← NEW
│   │
│   ├── services/
│   │   ├── trip_service.py
│   │   ├── geocoding_service.py
│   │   ├── weather_service.py
│   │   ├── weather_decision_service.py
│   │   ├── flight_service.py
│   │   └── hotel_service.py           ← NEW
│   │
│   └── tools/
│       ├── weather_tools.py
│       ├── flight_tools.py
│       └── hotel_tools.py             ← NEW
│
├── tests/
│   ├── test_health.py
│   ├── test_trip.py
│   ├── test_weather.py
│   ├── test_weather_decision.py
│   ├── test_flight.py
│   └── test_hotel.py                  ← NEW
│
├── .env                               ← MODIFIED
├── .env.example                       ← MODIFIED
└── ...
```

---

# 4. Get the StayingAPI sandbox key

StayingAPI documents a `stay_test_...` sandbox key that returns deterministic fixtures at zero credits. Their Try-It console also uses a shared sandbox key for test calls. [StayingAPI](https://stayingapi.com/docs/try-it?utm_source=chatgpt.com)

For our backend, put **your test key** in `.env`.

Example:

```env
STAYINGAPI_KEY=stay_test_YOUR_KEY_HERE
```

Do **not** commit this key to Git.

---

# 5. Update `app/core/config.py`

You currently have something like:

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

Update it to:

```python
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):

    APP_NAME: str = "Travel Agent API"

    DEBUG: bool = True

    DATABASE_URL: str

    SECRET_KEY: str

    STAYINGAPI_KEY: str


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
```

---

# 6. Update `.env`

Add:

```env
STAYINGAPI_KEY=stay_test_YOUR_KEY_HERE
```

Your `.env` should now contain something like:

```env
APP_NAME=Travel Agent API
DEBUG=true

DATABASE_URL=postgresql+psycopg://travel_user:travel_password@localhost:5432/travel_db

SECRET_KEY=change-this-to-a-random-secret-key

STAYINGAPI_KEY=stay_test_YOUR_KEY_HERE
```

Replace:

```text
stay_test_YOUR_KEY_HERE
```

with your actual sandbox key.

---

# 7. Update `.env.example`

Use:

```env
APP_NAME=Travel Agent API
DEBUG=true

DATABASE_URL=postgresql+psycopg://travel_user:travel_password@localhost:5432/travel_db

SECRET_KEY=change-this-to-a-random-secret-key

STAYINGAPI_KEY=stay_test_your_key_here
```

---

# 8. `app/schemas/hotel.py`

Create:

```text
app/schemas/hotel.py
```

Full code:

```python
from datetime import date
from decimal import Decimal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


class HotelSearchRequest(BaseModel):

    location: str = Field(
        min_length=2,
        max_length=200,
        description="City, country or lat,lng",
    )

    check_in: date

    check_out: date

    adults: int = Field(
        default=2,
        ge=1,
        le=20,
    )

    children: int = Field(
        default=0,
        ge=0,
        le=10,
    )

    rooms: int = Field(
        default=1,
        ge=1,
        le=10,
    )

    property_type: str = Field(
        default="hotel",
    )

    min_guest_rating: float | None = Field(
        default=None,
        ge=0,
        le=10,
    )

    price_max: Decimal | None = Field(
        default=None,
        gt=0,
    )

    platform: str = Field(
        default="booking",
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=40,
    )

    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Za-z]{3}$",
    )

    @model_validator(mode="after")
    def validate_dates(self):

        if self.check_out <= self.check_in:
            raise ValueError(
                "check_out must be after check_in"
            )

        allowed_property_types = {
            "hotel",
            "apartment",
            "house",
            "villa",
            "cottage",
            "other",
        }

        if self.property_type not in allowed_property_types:
            raise ValueError(
                "Invalid property_type"
            )

        return self


class HotelLocation(BaseModel):

    latitude: float = Field(
        alias="lat"
    )

    longitude: float = Field(
        alias="lng"
    )

    city: str

    country: str

    model_config = {
        "populate_by_name": True
    }


class HotelPrice(BaseModel):

    currency: str

    nightly_price: Decimal

    total_price: Decimal

    nights: int


class HotelResult(BaseModel):

    id: str

    platform: str

    platform_listing_id: str

    url: str | None = None

    name: str

    property_type: str

    location: HotelLocation

    star_rating: float | None = None

    guest_rating: float | None = None

    rating_scale: float | None = None

    review_count: int | None = None

    max_occupancy: int | None = None

    bedrooms: int | None = None

    bathrooms: int | None = None

    amenities: list[str] = []

    price: HotelPrice


class HotelSearchResponse(BaseModel):

    location: str

    check_in: date

    check_out: date

    adults: int

    children: int

    rooms: int

    provider: str

    hotels: list[HotelResult]

    total_results: int
```

---

# 9. Why these fields?

StayingAPI's search response currently includes normalized properties with fields such as:

```text
id
platform
platformListingId
url
name
propertyType
location
starRating
guestRating
ratingScale
reviewCount
maxOccupancy
bedrooms
bathrooms
amenities
price
```

and the price contains:

```text
currency
nightlyPrice
totalPrice
nights
```

according to its documented search schema. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

---

# 10. `check_out` validation

For hotel stays:

```text
Check-in:
2026-10-05

Check-out:
2026-10-08
```

means:

```text
Oct 5
Oct 6
Oct 7
Oct 8 checkout
```

So:

```text
3 nights
```

The provider also validates that `checkOut` is after `checkIn`. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

---

# 11. `app/services/hotel_service.py`

Create:

```text
app/services/hotel_service.py
```

Complete code:

```python
import httpx

from app.core.config import settings

from app.schemas.hotel import (
    HotelLocation,
    HotelPrice,
    HotelResult,
    HotelSearchRequest,
    HotelSearchResponse,
)


HOTEL_API_URL = (
    "https://api.stayingapi.com/v1/search"
)

PROVIDER_NAME = "stayingapi"


def _map_location(
    data: dict,
) -> HotelLocation:

    return HotelLocation(
        lat=data["lat"],
        lng=data["lng"],
        city=data["city"],
        country=data["country"],
    )


def _map_price(
    data: dict,
) -> HotelPrice:

    return HotelPrice(
        currency=data["currency"],
        nightly_price=data["nightlyPrice"],
        total_price=data["totalPrice"],
        nights=data["nights"],
    )


def _map_hotel(
    data: dict,
) -> HotelResult:

    return HotelResult(
        id=data["id"],
        platform=data["platform"],
        platform_listing_id=data[
            "platformListingId"
        ],
        url=data.get("url"),
        name=data["name"],
        property_type=data[
            "propertyType"
        ],
        location=_map_location(
            data["location"]
        ),
        star_rating=data.get(
            "starRating"
        ),
        guest_rating=data.get(
            "guestRating"
        ),
        rating_scale=data.get(
            "ratingScale"
        ),
        review_count=data.get(
            "reviewCount"
        ),
        max_occupancy=data.get(
            "maxOccupancy"
        ),
        bedrooms=data.get(
            "bedrooms"
        ),
        bathrooms=data.get(
            "bathrooms"
        ),
        amenities=data.get(
            "amenities",
            []
        ),
        price=_map_price(
            data["price"]
        ),
    )


async def _request_hotels(
    request: HotelSearchRequest,
) -> list[HotelResult]:

    params = {
        "location": request.location,
        "checkIn": request.check_in.isoformat(),
        "checkOut": request.check_out.isoformat(),
        "adults": request.adults,
        "children": request.children,
        "rooms": request.rooms,
        "propertyType": request.property_type,
        "platforms": request.platform,
        "limit": request.limit,
        "currency": request.currency.upper(),
    }

    if request.min_guest_rating is not None:
        params["minGuestRating"] = (
            request.min_guest_rating
        )

    if request.price_max is not None:
        params["priceMax"] = (
            request.price_max
        )

    headers = {
        "Authorization": (
            f"Bearer {settings.STAYINGAPI_KEY}"
        )
    }

    async with httpx.AsyncClient(
        timeout=20.0
    ) as client:

        response = await client.get(
            HOTEL_API_URL,
            params=params,
            headers=headers,
        )

        response.raise_for_status()

        payload = response.json()

    records = payload.get(
        "data",
        []
    )

    return [
        _map_hotel(record)
        for record in records
    ]


async def search_hotels(
    request: HotelSearchRequest,
) -> HotelSearchResponse:

    hotels = await _request_hotels(
        request
    )

    return HotelSearchResponse(
        location=request.location,
        check_in=request.check_in,
        check_out=request.check_out,
        adults=request.adults,
        children=request.children,
        rooms=request.rooms,
        provider=PROVIDER_NAME,
        hotels=hotels,
        total_results=len(hotels),
    )
```

The documented StayingAPI search endpoint uses:

```text
GET https://api.stayingapi.com/v1/search
Authorization: Bearer ...
```

and supports location, check-in/check-out, adults, children, rooms, property type, ratings, price, platforms, limit and currency. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

---

# 12. Understand the hotel service

The service does:

```text
HotelSearchRequest
       │
       ▼
Build query parameters
       │
       ▼
Authorization header
       │
       ▼
StayingAPI
       │
       ▼
{
  "data": [...]
}
       │
       ▼
Map provider fields
       │
       ▼
HotelResult
```

---

# 13. Why the service maps the fields

Provider:

```json
{
  "platformListingId": "abc",
  "propertyType": "hotel",
  "guestRating": 9.1
}
```

Application:

```python
platform_listing_id
property_type
guest_rating
```

This gives us a stable internal model.

Later:

```text
StayingAPI
      ↓
Real hotel provider
```

can be changed without forcing every other layer to understand provider-specific field names.

---

# 14. `app/api/routes/hotels.py`

Create:

```text
app/api/routes/hotels.py
```

```python
import httpx

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.schemas.hotel import (
    HotelSearchRequest,
    HotelSearchResponse,
)

from app.services.hotel_service import (
    search_hotels,
)


router = APIRouter(
    prefix="/hotels",
    tags=["Hotels"],
)


@router.post(
    "/search",
    response_model=HotelSearchResponse,
)
async def search(
    request: HotelSearchRequest,
):

    try:

        return await search_hotels(
            request=request
        )

    except httpx.HTTPStatusError as error:

        status_code = (
            error.response.status_code
        )

        if status_code == 401:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Hotel provider "
                    "authentication failed"
                ),
            )

        if status_code == 400:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Hotel provider rejected "
                    "the search request"
                ),
            )

        raise HTTPException(
            status_code=502,
            detail=(
                "Hotel provider returned "
                f"HTTP {status_code}"
            ),
        )

    except httpx.HTTPError:

        raise HTTPException(
            status_code=502,
            detail=(
                "Hotel provider is "
                "currently unavailable"
            ),
        )
```

---

# 15. `app/tools/hotel_tools.py`

Create:

```text
app/tools/hotel_tools.py
```

```python
from app.schemas.hotel import (
    HotelSearchRequest,
    HotelSearchResponse,
)

from app.services.hotel_service import (
    search_hotels,
)


async def hotel_search_tool(
    request: HotelSearchRequest,
) -> HotelSearchResponse:

    return await search_hotels(
        request=request
    )
```

---

# 16. Why the hotel tool exists

Our agent will eventually have:

```text
                 TRAVEL AGENT
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
      Weather       Flight       Hotel
       Tool          Tool         Tool
          │           │            │
          ▼           ▼            ▼
      Service       Service      Service
          │           │            │
          ▼           ▼            ▼
      Provider      Provider     Provider
```

At this point, the tools are just wrappers around our services.

The actual agent comes later.

---

# 17. Update `app/main.py`

Add:

```python
from app.api.routes.hotels import (
    router as hotels_router,
)
```

Then register:

```python
app.include_router(
    hotels_router
)
```

Your complete `main.py` becomes:

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

from app.api.routes.hotels import (
    router as hotels_router,
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

app.include_router(
    hotels_router
)


@app.get("/")
def root():

    return {
        "message": "Travel Agent API is running"
    }
```

---

# 18. Test the provider first

StayingAPI's current Try-It documentation uses a sandbox request such as:

```text
GET /v1/search
?location=Split, HR
&checkIn=2026-10-13
&checkOut=2026-10-20
&platforms=airbnb,booking
&limit=5
```

and returns a deterministic `200` response from the sandbox at zero credits. [StayingAPI](https://stayingapi.com/docs/try-it?utm_source=chatgpt.com)

For our Python backend, test with your sandbox key.

PowerShell:

```powershell
$headers = @{
    Authorization = "Bearer YOUR_STAY_TEST_KEY"
}

Invoke-RestMethod `
  "https://api.stayingapi.com/v1/search?location=Split%2C%20HR&checkIn=2026-10-13&checkOut=2026-10-20&adults=2&platforms=booking&limit=5" `
  -Headers $headers
```

---

# 19. Test our FastAPI endpoint

Start backend:

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

You should now see:

```text
GET  /health

POST /travel/plan

GET  /weather/{destination}

POST /travel/best-dates

POST /flights/search

POST /hotels/search          ← NEW
```

---

# 20. Test `/hotels/search`

Use:

```http
POST /hotels/search
```

Body:

```json
{
  "location": "Split, HR",
  "check_in": "2026-10-13",
  "check_out": "2026-10-20",
  "adults": 2,
  "children": 0,
  "rooms": 1,
  "property_type": "hotel",
  "platform": "booking",
  "limit": 5,
  "currency": "USD"
}
```

This matches the shape of the provider's documented search contract. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

---

# 21. Expected response

Our API will normalize the provider response into approximately:

```json
{
  "location": "Split, HR",
  "check_in": "2026-10-13",
  "check_out": "2026-10-20",
  "adults": 2,
  "children": 0,
  "rooms": 1,
  "provider": "stayingapi",
  "total_results": 5,
  "hotels": [
    {
      "id": "stays_booking_...",
      "platform": "booking",
      "platform_listing_id": "...",
      "url": "https://...",
      "name": "Example Hotel",
      "property_type": "hotel",
      "location": {
        "lat": 43.51,
        "lng": 16.44,
        "city": "Split",
        "country": "HR"
      },
      "star_rating": 4,
      "guest_rating": 9.1,
      "rating_scale": 10,
      "review_count": 142,
      "max_occupancy": 4,
      "bedrooms": 2,
      "bathrooms": 1,
      "amenities": [
        "pool",
        "wifi"
      ],
      "price": {
        "currency": "USD",
        "nightly_price": 303,
        "total_price": 2122,
        "nights": 7
      }
    }
  ]
}
```

The documented StayingAPI search result uses this normalized property shape and price structure. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

---

# 22. Price filtering

Our API supports:

```json
{
  "location": "Split, HR",
  "check_in": "2026-10-13",
  "check_out": "2026-10-20",
  "adults": 2,
  "price_max": 150,
  "platform": "booking"
}
```

This becomes:

```text
priceMax=150
```

The provider documents `priceMax` as a supported search filter. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

---

# 23. Rating filtering

Request:

```json
{
  "location": "Split, HR",
  "check_in": "2026-10-13",
  "check_out": "2026-10-20",
  "adults": 2,
  "min_guest_rating": 8
}
```

becomes:

```text
minGuestRating=8
```

The API documents guest-rating filtering against its normalized rating scale. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

---

# 24. Test multiple platforms

You can later make:

```text
platforms=airbnb,booking
```

and the provider can fan out across multiple accommodation platforms into one normalized schema. The documented search endpoint supports the `platforms` parameter and merges results into the same property shape. [GitHub](https://github.com/stayingapi/hotel-api/blob/main/endpoints/search.md?utm_source=chatgpt.com)

For this phase, keeping:

```text
platform = booking
```

is simpler.

---

# 25. `tests/test_hotel.py`

Create:

```text
tests/test_hotel.py
```

Use:

```python
from datetime import date
from decimal import Decimal

import pytest

from app.schemas.hotel import (
    HotelLocation,
    HotelPrice,
    HotelResult,
    HotelSearchRequest,
)

from app.services.hotel_service import (
    search_hotels,
)


def create_mock_hotel() -> HotelResult:

    return HotelResult(
        id="stays_booking_test_001",
        platform="booking",
        platform_listing_id="test_001",
        url="https://example.com/hotel",
        name="Test Hotel",
        property_type="hotel",
        location=HotelLocation(
            lat=43.51,
            lng=16.44,
            city="Split",
            country="HR",
        ),
        star_rating=4,
        guest_rating=9.1,
        rating_scale=10,
        review_count=150,
        max_occupancy=4,
        bedrooms=2,
        bathrooms=1,
        amenities=[
            "wifi",
            "pool",
        ],
        price=HotelPrice(
            currency="USD",
            nightly_price=Decimal(
                "120"
            ),
            total_price=Decimal(
                "840"
            ),
            nights=7,
        ),
    )


@pytest.mark.anyio
async def test_search_hotels(
    monkeypatch,
):

    mock_hotel = create_mock_hotel()

    async def mock_request_hotels(
        request,
    ):
        return [mock_hotel]

    monkeypatch.setattr(
        "app.services.hotel_service._request_hotels",
        mock_request_hotels,
    )

    request = HotelSearchRequest(
        location="Split, HR",
        check_in=date(
            2026,
            10,
            13,
        ),
        check_out=date(
            2026,
            10,
            20,
        ),
        adults=2,
        children=0,
        rooms=1,
        property_type="hotel",
        platform="booking",
        limit=5,
        currency="USD",
    )

    result = await search_hotels(
        request
    )

    assert result.location == (
        "Split, HR"
    )

    assert result.adults == 2

    assert result.children == 0

    assert result.rooms == 1

    assert result.provider == (
        "stayingapi"
    )

    assert result.total_results == 1

    assert len(
        result.hotels
    ) == 1

    assert (
        result.hotels[0].name
        == "Test Hotel"
    )


def test_invalid_checkout():

    with pytest.raises(
        ValueError
    ):

        HotelSearchRequest(
            location="Split, HR",
            check_in=date(
                2026,
                10,
                20,
            ),
            check_out=date(
                2026,
                10,
                13,
            ),
        )


def test_invalid_property_type():

    with pytest.raises(
        ValueError
    ):

        HotelSearchRequest(
            location="Split, HR",
            check_in=date(
                2026,
                10,
                13,
            ),
            check_out=date(
                2026,
                10,
                20,
            ),
            property_type="castle",
        )


def test_invalid_adults():

    with pytest.raises(
        ValueError
    ):

        HotelSearchRequest(
            location="Split, HR",
            check_in=date(
                2026,
                10,
                13,
            ),
            check_out=date(
                2026,
                10,
                20,
            ),
            adults=0,
        )
```

---

# 26. Why the hotel tests don't call the internet

Exactly like the flight tests:

```text
pytest
   ↓
mock provider call
   ↓
test our logic
```

not:

```text
pytest
   ↓
Internet
   ↓
StayingAPI
```

Otherwise your tests could fail because of:

```text
network
provider outage
rate limits
authentication
```

The actual provider connection should be tested separately.

---

# 27. Test the API route

You can also add one route-level test.

Add to `tests/test_hotel.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_hotel_search_endpoint(
    monkeypatch,
):

    async def mock_search_hotels(
        request,
    ):

        return {
            "location": request.location,
            "check_in": request.check_in,
            "check_out": request.check_out,
            "adults": request.adults,
            "children": request.children,
            "rooms": request.rooms,
            "provider": "stayingapi",
            "hotels": [],
            "total_results": 0,
        }

    monkeypatch.setattr(
        "app.api.routes.hotels.search_hotels",
        mock_search_hotels,
    )

    response = client.post(
        "/hotels/search",
        json={
            "location": "Split, HR",
            "check_in": "2026-10-13",
            "check_out": "2026-10-20",
            "adults": 2,
            "children": 0,
            "rooms": 1,
            "property_type": "hotel",
            "platform": "booking",
            "limit": 5,
            "currency": "USD",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["location"] == (
        "Split, HR"
    )

    assert data["adults"] == 2

    assert data["provider"] == (
        "stayingapi"
    )
```

---

# 28. Run all tests

First make sure the key exists in `.env`.

Then:

```powershell
python -m pytest
```

Your suite should now contain:

```text
test_trip.py
test_weather.py
test_weather_decision.py
test_flight.py
test_hotel.py
```

The exact total number depends on the tests you've retained from the earlier phases.

Most importantly:

```text
FAILED = 0
```

---

# 29. Test the real sandbox separately

The deterministic sandbox is useful for confirming the entire integration:

```text
FastAPI
   ↓
hotel_service
   ↓
HTTP
   ↓
StayingAPI sandbox
   ↓
real JSON
   ↓
mapping
   ↓
your API response
```

A documented sandbox search example is:

```text
location = Split, HR
checkIn = 2026-10-13
checkOut = 2026-10-20
platforms = airbnb,booking
limit = 5
```

and the sandbox returns a genuine `200` with zero credits. [StayingAPI](https://stayingapi.com/docs/try-it?utm_source=chatgpt.com)

---

# 30. Important distinction: API test vs unit test

You now have two types of testing.

### Unit tests

```text
pytest
 ↓
mock
 ↓
test our code
```

### Integration test

```text
FastAPI
 ↓
StayingAPI
 ↓
real sandbox response
```

That's a good architecture because you don't need the internet for every test run.

---

# 31. Current hotel architecture

```text
                       USER
                         │
                         ▼
                POST /hotels/search
                         │
                         ▼
                HotelSearchRequest
                         │
                         ▼
                  Hotel API Route
                         │
                         ▼
                  Hotel Service
                         │
                         ▼
                StayingAPI /v1/search
                         │
                         ▼
                  JSON response
                         │
                         ▼
                    Normalize
                         │
                         ▼
                    HotelResult
                         │
                         ▼
               HotelSearchResponse
                         │
                         ▼
                       USER
```

---

# 32. Agent architecture now

We have three external capabilities:

```text
                     TRAVEL AGENT
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
     Weather Tool    Flight Tool     Hotel Tool
          │               │               │
          ▼               ▼               ▼
      Weather        Flight Service  Hotel Service
      Service             │               │
          │               │               │
          ▼               ▼               ▼
     Open-Meteo      randomapi.dev    StayingAPI
```

This is the foundation for the next stage.

---

# 33. How Phase 5 connects to Phase 3

Suppose Phase 3 finds:

```text
Goa
Oct 5 → Oct 8
```

Then eventually:

```text
Weather Decision
       │
       ▼
Oct 5 → Oct 8
       │
       ├───────────────┐
       ▼               ▼
Flight Search      Hotel Search
       │               │
       ▼               ▼
Flight options     Hotel options
```

Eventually the planner will combine all of them.

---

# 34. What we're NOT doing yet

Do **not** add:

```text
❌ Hotel ranking
❌ Cheapest hotel selection
❌ Best hotel selection
❌ Flight + hotel combination
❌ Booking
❌ Payment
❌ LLM
❌ LangGraph
❌ Memory
```

Phase 5's responsibility is only:

> Search and return accommodation options for a destination and date range.

---

# 35. Phase 5 Definition of Done

```text
✅ StayingAPI sandbox configured
✅ STAYINGAPI_KEY in environment
✅ Hotel search schema
✅ Check-in validation
✅ Check-out validation
✅ Occupancy validation
✅ Room validation
✅ Property type validation
✅ Price filtering
✅ Guest-rating filtering
✅ Platform filtering
✅ Currency
✅ Hotel service
✅ Provider HTTP integration
✅ Provider response normalization
✅ Hotel tool
✅ Hotel route
✅ Error handling
✅ Hotel tests
✅ Full test suite passes
```

---

# 36. Current project progress

```text
Phase 0 ✅ Infrastructure
       ↓
Phase 1 ✅ Trip + PostgreSQL
       ↓
Phase 2 ✅ Weather Integration
       ↓
Phase 3 ✅ Weather Decision Engine
       ↓
Phase 4 ✅ Flight Search
       ↓
Phase 5 🔵 Hotel Search    ← NOW
       ↓
Phase 6 → Cab Search
       ↓
Phase 7 → Connect everything
       ↓
Phase 8 → Recommendation Engine
       ↓
Phase 9 → LangGraph
       ↓
Phase 10 → Parallel execution
       ↓
Phase 11 → LLM
       ↓
Phase 12 → Memory
       ↓
Phase 13 → Human approval
       ↓
Phase 14 → Booking
```

**One important correction from the earlier roadmap:** I would keep **hotel ranking/recommendation out of Phase 5**. First make the provider integration solid and independently testable. Then we can build a separate recommendation layer that considers **weather dates + flights + hotels together**, which is much closer to the actual travel-agent problem.
