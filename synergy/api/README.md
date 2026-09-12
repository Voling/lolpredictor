# api

FastAPI service over the trained model. Reads the parquet tables and the pickled model from
`DATA_DIR`; writes nothing except cache entries.

## Running

```bash
uvicorn synergy.api.server:app --host 0.0.0.0 --port 8000
```

The module is `server.py`, not `app.py`: a submodule named `app` shadows the `app` object and
`uvicorn synergy.api:app` silently resolves to the module.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Model readiness, corpus size, training metrics, cache state |
| `GET /api/players?q=&limit=` | Profiled players, most-seen first |
| `GET /api/players/{riot_id}` | Profile, trait percentiles, tendencies |
| `GET /api/partners/{riot_id}?limit=` | Best and worst modelled partners |
| `GET /api/pair?a=&b=` | One pairing: score, synergy, drivers, shared play |
| `POST /api/team` | `{"players": [...]}`, two to five, pairwise matrix and group score |
| `GET /api/outsider/{riot_id}` | Feature percentiles for a player outside the corpus |
| `GET /api/outsider-pair?a=&b=` | Two such players, largest style gaps, teammate and opponent games |
| `GET /api/corpus` | Corpus row count and the feature columns compared against |
| `POST /api/reload` | Reload model and profiles from disk, drop caches |

## Two things to know

`score` is `null` and `reliable` is `false` whenever `synergy_gain_sigma` falls below
`MIN_SYNERGY_GAIN_SIGMA` (2.0). The raw `synergy` value is still returned. Do not convert it to a
score client-side.

The `outsider` endpoints exist for players below the corpus LP floor, who have no profile and whose
games never enter `participations`. They read that player's stored matches directly and rank the
features against cached corpus quantiles. Redis caches both the quantiles and each profile; without
Redis everything still works, just slower.
