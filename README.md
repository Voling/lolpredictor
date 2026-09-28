# lolpredictor

Ranks which friends a League of Legends player should queue with. Each friend gets one score for the two of you as a duo, built from how each of you plays in other games, how your styles fit, and how you did together at 20 minutes when you have played together, calibrated on matches the model never saw. Run everything from the repo root with the virtualenv active and `RIOT_API_KEY` in `.env`.

| Command | What it does |
|---|---|
| `docker compose up -d postgres redis broker` | Start the database, the cache and the Celery broker |
| `python -m synergy worker` | Celery workers for the pipeline's parallel jobs, keep running during a pipeline |
| `python -m synergy crawl` | Pull ranked matches from the Riot API, resumable |
| `python -m synergy pipeline` | Rebuild features and models into a new run, then serve it |
| `python -m synergy pipeline --batches crawl` | Train on the crawl alone, or on a comma list of batches from a past manifest |
| `python -m synergy contributed` | Import the Master+ games waiting in `CONTRIBUTED_STORE` as a new batch and empty the store, after you confirm |
| `python -m synergy runs` | List runs and which one is served |
| `python -m synergy promote --id RUN` | Serve another run, for rollback |
| `uvicorn synergy.api.server:app` | API on :8000 |
| `cd frontend && npm run dev` | Site on :3000 |
| `cd frontend && npm run e2e` | Browser tests against a running stack |
| `pytest` | Unit tests |
| `bash deploy/deploy.sh` | Deploy the served run, the site, Cognito sign in and the WAF to AWS with Terraform, copying `RIOT_API_KEY` into SSM |
| `bash deploy/publish_model.sh` | Upload the served run to the models bucket so the next backend deploy serves it |
| GitHub Actions `backend` and `frontend` | Deploy each side on its own when its files change on main, or by hand from the Actions tab |
