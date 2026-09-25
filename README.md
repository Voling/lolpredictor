# lolpredictor

Ranks which friends a League of Legends player should queue with. Each friend gets one score for the two of you as a duo, built from how each of you plays in other games, how your styles fit, and how you did together at 20 minutes when you have played together, calibrated on matches the model never saw. Run everything from the repo root with the virtualenv active and `RIOT_API_KEY` in `.env`.

| Command | What it does |
|---|---|
| `docker compose up -d postgres redis` | Start the database and cache |
| `python -m synergy crawl` | Pull ranked matches from the Riot API, resumable |
| `python -m synergy pipeline` | Rebuild features and models into a new run, then serve it |
| `python -m synergy runs` | List runs and which one is served |
| `python -m synergy promote --id RUN` | Serve another run, for rollback |
| `uvicorn synergy.api.server:app` | API on :8000 |
| `cd frontend && npm run dev` | Site on :3000 |
| `cd frontend && npm run e2e` | Browser tests against a running stack |
| `pytest` | Unit tests |
| `bash deploy/deploy.sh` | Deploy the served run to AWS with Terraform |
