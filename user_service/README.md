# user_service

Standalone "who is this user / which travel plans do they own" service.

It stores **only** two things:

- `users` – one row per authenticated Supabase subject
- `travel_plans` – many per user, itinerary payload kept in a JSONB column

It is completely independent from the research code in the repository root
(no shared modules, no shared env vars, no shared Docker Compose file).

---

## 1. Authentication model

```
Google  ->  Supabase Auth  ->  JWT (access_token)  ->  FastAPI (this service)
```

- FastAPI never renders a Google login page and never stores Google passwords.
- The frontend signs the user in with Supabase, then sends:

```
Authorization: Bearer <supabase_access_token>
```

- The token **signature is always verified**:
  - new Supabase projects: JWKS at
    `${USER_SERVICE_SUPABASE_URL}/auth/v1/.well-known/jwks.json`
    (RS256 / ES256 / EdDSA), plus `aud` and `iss` checks;
  - legacy projects: shared secret, HS256.
- On the first request from a given subject, the local `users` row is created
  automatically. Later requests reuse it.

### DEV_MODE (local testing only)

With `USER_SERVICE_DEV_MODE=1` the service also accepts:

```
Authorization: Bearer dev:test-user
```

`USER_SERVICE_DEV_MODE` **defaults to `0`**.
**Never enable it in production** – it bypasses signature verification.
Tokens that are not prefixed with `dev:` are still fully verified even when
DEV_MODE is on.

---

## 2. API

Base path: `/api/v1`

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe (no auth) |
| GET | `/api/v1/users/me` | Current user, auto-created on first login |
| PATCH | `/api/v1/users/me` | Update own `display_name` / `avatar_url` / `locale` |
| GET | `/api/v1/travel-plans?limit=&offset=` | List only the caller's plans |
| POST | `/api/v1/travel-plans` | Create a plan (HTTP 201) |
| GET | `/api/v1/travel-plans/{plan_id}` | Read one own plan |
| PATCH | `/api/v1/travel-plans/{plan_id}` | Patch supplied fields only |
| DELETE | `/api/v1/travel-plans/{plan_id}` | Delete own plan (HTTP 204) |

Rules:

- `user_id` is **never** accepted from the client (`extra="forbid"` -> HTTP 422).
  Ownership always comes from the verified JWT.
- A plan owned by somebody else returns **404**, not 403.
- Unknown fields in any request body are rejected with HTTP 422.

### Example

```bash
curl -X POST http://localhost:8001/api/v1/travel-plans \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "title": "台南三天兩夜",
        "destination": "台南",
        "content": {"schema_version": 1, "days": []}
      }'
```

---

## 3. Data model

```
users
  id                uuid pk
  supabase_user_id  varchar unique      <- JWT "sub"
  email             varchar null
  display_name      varchar null
  avatar_url        text    null
  locale            varchar null
  created_at / updated_at

travel_plans
  id           uuid pk
  user_id      uuid fk -> users.id ON DELETE CASCADE
  title        varchar not null
  destination  varchar null
  start_date   date    null
  end_date     date    null
  status       varchar null
  content      jsonb   not null        <- planner payload
  created_at / updated_at
```

`days` / `slots` / `restaurants` / … intentionally stay inside `content`.
The planner format is still moving, so no sub-tables are created yet.

---

## 4. Run it

```bash
cp user_service/.env.example user_service/.env
# fill in USER_SERVICE_SUPABASE_URL (and the JWT secret only for legacy projects)

docker compose -f user_service/docker-compose.yml up -d --build
```

- Health: <http://localhost:8001/health>
- Swagger UI: <http://localhost:8001/docs>
- PostgreSQL is exposed on host port **5433** (container port 5432);
  FastAPI on host port **8001**.

Stop / wipe:

```bash
docker compose -f user_service/docker-compose.yml down
docker compose -f user_service/docker-compose.yml down -v
```

### Without Docker

```bash
pip install -r user_service/requirements.txt
set USER_SERVICE_DATABASE_URL=postgresql+psycopg://user_service:user_service@localhost:5433/user_service
uvicorn user_service.main:app --reload --port 8001
```

---

## 5. Tests

No network access, no Google, no Supabase: tests use API-level DEV tokens and
an in-memory SQLite database.

```bash
pytest user_service -v
```

---

## 6. Schema management (prototype)

Tables are created at startup with `Base.metadata.create_all()`.
That is fine for the first version. Before a real production deployment,
replace it with Alembic migrations (add `alembic`, generate an initial
revision, drop the `create_all()` call in `user_service/main.py`).

---

## 7. Manual setup checklist

1. Create a Supabase project.
2. Enable the **Google** provider in Authentication → Providers.
3. Add the site URL / redirect URLs (the Google OAuth callback is handled by
   Supabase, typically `https://<project-ref>.supabase.co/auth/v1/callback`).
4. Put the project URL in `USER_SERVICE_SUPABASE_URL`
   (e.g. `https://abcdefgh.supabase.co`).
5. Only for legacy HS256 projects: set `USER_SERVICE_SUPABASE_JWT_SECRET`
   (Project Settings → API → JWT Secret).
6. Point the frontend at this service and send the Supabase access token as a
   bearer token.
