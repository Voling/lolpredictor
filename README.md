# lolpredictor

Scores League of Legends duos. The score predicts how far ahead the duo's two positions will be in gold at 20 minutes, from each player's other games and any games they played together.

Run everything from the repo root with the virtualenv active and `RIOT_API_KEY` in `.env`.

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
| `python -m synergy fits` | Save the corpus fitted worlds, kappas and tendency models after a build; the pipeline saves them itself from now on |
| `python -m synergy bundle` | Copy those fits and the encoders next to the served run for the evaluator, then publish |
| `python -m synergy pack` | Pack the served run's style vectors; the pipeline does this on promote, older runs need it before publishing |
| `uvicorn synergy.api.server:app` | API on :8000 |
| `cd frontend && npm run dev` | Site on :3000 |
| `cd frontend && npm run e2e` | Browser tests against a running stack |
| `pytest` | Unit tests |
| `bash deploy/deploy.sh` | Deploy the served run, the site, Cognito sign in and the WAF to AWS with Terraform, copying `RIOT_API_KEY` into SSM |
| `bash deploy/publish_model.sh` | Upload the served run to the models bucket; the API fetches it on its next cold start, no deploy needed |
| GitHub Actions `backend` and `frontend` | Deploy each side on its own when its files change on main, or by hand from the Actions tab |
