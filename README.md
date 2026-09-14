# lolpredictor

Scores how well two League of Legends players fit together, 0 to 100, from how they actually play
rather than from win rate alone. A seeded crawl walks outward from one ranked player through the other
participants in their Diamond to Master solo queue games, match and timeline JSON are stored raw,
behavioural features are extracted per player, and a two-stage model separates individual strength from
the part of a team's result that only a specific pairing explains.

## Pipeline

1. **Crawl** ([seeder.py](synergy/ingest/seeder.py)) resolves the seed Riot ID, pulls their ranked match
   ids, stores each match and its timeline, then pushes every participant onto a priority frontier.
   Repeat encounters raise a player's priority, so the crawl densifies the seed's own circle instead of
   drifting into strangers. Players outside the tier window are skipped, the frontier lives in Postgres so
   a run resumes where it stopped, a token bucket enforces both Riot rate windows with 429 back-off, and
   `CRAWL_MAX_REQUESTS` caps what one run may spend. Matches are fetched eight at a time; a reserve of
   requests at the end resolves Riot IDs for the participants seen often enough to be profiled.
2. **Window** ([window.py](synergy/ingest/window.py)) derives a second raw layer holding only what
   happened between 0:00 and 15:00, frames and events alike. Laning phase is where behaviour is legible
   and comparable across games, and putting the cut in its own stage means the rule lives in one place
   rather than in every extractor. It also drops the corpus from 72 MB to 32 MB. End of game summary
   statistics are deliberately unused for behaviour, since they describe a result rather than a choice.
3. **Features** ([early.py](synergy/features/early.py)) read only that window: map territory shares over
   18 team relative zones, invade camps taken, takedowns split by where they happened, proximity to both
   junglers, damage given and taken per minute, the trade ratio between them, how many distinct enemies a
   player damaged, how many allies focused the same target, how outnumbered they were when they died,
   vision wards, and every purchase with its first back. Ward counts exclude champion traps: `WARD_PLACED`
   bundles Teemo mushrooms and other plants under `UNDEFINED`, 35 percent of the events, and filtering to
   real vision types moves agreement with the match summary from 0.28 to 1.00.
4. **Normalisation** ([normalise.py](synergy/features/normalise.py)) compares a player to others in the
   same role on the same champion, shrinking toward the role mean by `n / (n + 10)` when that champion
   cell is thin. This is what makes damage comparable: `totalDamageTaken` counts monsters, so a jungler
   reads as a poor trader against a bot laner until the champion and role baseline is removed.
5. **Tendencies** ([propensity.py](synergy/features/propensity.py)) treat invading as conditional rather
   than as a rate, because the chance to invade is handed to a player by the game. Every player minute
   becomes an opportunity row with the situation lagged one minute behind the decision, a logistic model
   predicts the chance any player would act, and each player's observed over expected ratio is shrunk
   with a Gamma-Poisson empirical Bayes prior fitted from how much players actually differ. Three
   tendencies are estimated separately: starting an invade, following one already underway, and
   answering an invasion of your own jungle.

| Tendency | Opportunity rows | Base rate | Player dispersion | Verdict |
|---|---|---|---|---|
| Initiate an invade | 105,575 | 1.6% | 0.45 | a real trait |
| Follow an invade | 6,754 | 2.4% | 0.93 | a real trait |
| Answer an invasion | 9,291 | 29.7% | below noise | not separable |

   Initiating and following correlate only 0.35, so they are worth estimating apart. Defending is not
   distinguishable from chance: given the same situation every player responds about the same, which
   says defence is dictated by where you already were rather than by disposition. Conditioning matters:
   the top 20 initiators split across all five roles rather than collapsing to junglers.
6. **Profiles** collapse the window features into six style axes learned from the data by a generalised
   eigendecomposition, `t1` to `t6`, reported as percentiles alongside the tendencies. They replaced nine
   hand named axes; two of those were exactly collinear, which left the basis rank deficient. A seeded crawl leaves most participants with one or two games, so each estimate is
   shrunk toward the population average by `games / (games + STYLE_SHRINKAGE_K)` and the profile reports
   the confidence weight it was given.
7. **Model** turns each match's ten same-team pairs into an interaction vector: the symmetric
   cross-products of the two players' style axes, the absolute style gaps, and the pair's own shared
   history. A logistic baseline predicts the result from team-level individual strength; a ridge then
   fits what the summed interaction vectors explain about the *residual*, which is what isolates fit from
   strength. Every profile input is leave-one-out. Role pairing is deliberately absent: both teams always
   field the same ten role combinations, so it cancels exactly in the team difference.
8. **Score** is `w . phi(pair)`, the pair's contribution to the team's win-rate residual, reported as a
   percentile against every pairing in the corpus, so 50 is an average pairing. It is gated: the model
   reports no score unless the pair terms beat their own noise by `synergy_gain_sigma` of 2, and on the
   win outcome they never have. See [docs/variance.md](docs/variance.md).

Frames and events live in Postgres 17 hypertables under TimescaleDB, everything else in ordinary
relational tables. See [docs/timescaledb.md](docs/timescaledb.md).

## What it measures, on 42,816 Master and Diamond matches

Out-of-fold AUC on predicting which team won, from a seeded crawl outward from bblskibs#gotg. The corpus
holds 43,835 matches, 12,795,420 frames and 48,863,986 events; 42,816 matches and 17,262 players survive
the tier and season filters.

| Predictor | AUC | Log loss |
|---|---|---|
| Playstyle sums alone | 0.5446 | |
| Individual strength, styles, rank, movement | 0.6745 | 0.6515 |
| Strength plus pair interaction terms | 0.6747 | 0.6515 |

Pair terms add 0.0002 AUC, and `synergy_gain_sigma` reads 0.04 against a gate of 2, so no pair score is
reported on this outcome. That null is informative rather than merely absent: simulating a known pair
effect through the real design shows the win outcome recovers an effect of 0.25, nearly as large as the
player effect itself, at only 1.5 null spreads. Win compresses each match into one bit shared across ten
pairs and arrives twenty minutes after the features, so it cannot see a pair effect of any plausible size.

Gold plus objectives at minute 15 can. It recovers that same planted effect at 13.3 null spreads, and on
real data it says two different things:

| Question | Result |
|---|---|
| Does pair identity have a variance component? | no, 0.1 null spreads, so any effect is under about 600 gold |
| Do the two players' styles interact beyond each player alone? | **yes, +0.0011 R², no draw of 40 nulls reached it** |

So a specific duo carries no consistent effect of its own, but how two playstyles combine does predict
board advantage, worth roughly 180 gold at minute 15 and about 2 percent of what the players individually
explain. The current gate is calibrated on win and therefore rejects it.

## Movement

Players are distinguishable by where they go next given where they are. Each player's region transition
matrix is shrunk toward the global one, and held out by whole player-games that beats the global matrix
by 0.096 nats per transition. Reduced to 12 components, the signature adds 0.0082 R² on gold at 15 beyond
rank, which no draw of 40 scrambled nulls reached. Averaging a player's position instead of conditioning
on it predicts nothing: position responds to what just happened, so a marginal average describes the
situations a player met rather than the player. See [docs/movement.md](docs/movement.md).

## Learned timeline representations

These numbers predate the current corpus and are kept for the comparison they make rather than as
current metrics.

`synergy/deep/` replaces the hand weighted axes with a representation learned from the timelines
themselves. Each player game becomes 45 minute steps carrying a map region token from 18 team relative
zones (three lanes by own, contested and enemy depth, four jungle quadrants, the two pits, both bases),
a second region token for whichever positioned event happened that minute, and 16 per minute numbers
covering gold, xp, cs, level, distance travelled, gold share and nine event counts. A 3 layer
transformer encodes that into 64 dimensions, conditioned on champion and role, and is trained with a
supervised contrastive loss where the label is simply the player id, which needs no win label and so
gets 10,687 training games rather than 1155 outcomes.

The test is whether a held out game retrieves its own player. One game is held back for each of the 823
players with three or more games, and the gallery holds every other player in the corpus.

| Representation | Recall@1, 6916 player gallery | Recall@1, 344 player gallery |
|---|---|---|
| Learned encoder | 0.041 | 0.241 |
| Champion pick alone | 0.005 | 0.204 |
| 54 raw z-scored statistics | 0.006 | 0.044 |
| Ten hand weighted axes | 0.000 | 0.012 |
| Chance | 0.00014 | 0.0029 |

Occluding the inputs of the trained encoder shows where that comes from. On the full gallery, champion
and role alone reach 0.009, the per minute numbers push it to 0.034, the region trace alone to 0.028 and
everything together to 0.041, so behaviour carries real identity beyond the pick. On the small gallery
champion alone already reaches 0.233 against 0.241 for the full model, meaning that easier task is
mostly answered by knowing who mains what.

Downstream the picture is narrower. Swapping the embeddings in for the ten axes lifts the individual
strength model from 0.538 to 0.556 AUC across all 1155 matches, is level on the 200 well observed ones
(0.613 against 0.614), and pair interaction terms still add nothing in either representation. The
representation improved; the pair synergy signal did not appear.

Two design notes worth keeping. Adversarially removing champion with a gradient reversal layer, the
obvious way to isolate playstyle, destroys the model: it lands at 0.012, exactly the hand weighted axes
and 20 times worse than conditioning on champion instead. Champion choice is playstyle, not a confound.
And the encoder memorises quickly without augmentation, so training crops a random contiguous 60 to 100
percent of the valid minutes and drops 10 percent of the remaining steps.

GPU training needs `pip install -r requirements-deep.txt`. The base install stays CPU only.

## Commands

Run from the repo root with the virtualenv active. Copy `.env.example` to `.env` and set `RIOT_API_KEY`.

| Command | What it does |
|---|---|
| `python -m synergy crawl` | Seeded crawl from `SEED_RIOT_ID` against the Riot API, resumable, budget capped |
| `python -m synergy window` | Cut raw timelines down to the first 15 minutes |
| `python -m synergy features` | Extract participation, pair and opportunity tables from the windows |
| `python -m synergy train` | Fit the baseline and synergy models, write profiles and the report |
| `python -m synergy reingest` | Rewrite frames and events from the raw timelines |
| `python -m synergy status` | Corpus counts, crawl frontier and model metrics |
| `python -m synergy pair "a#tag" "b#tag"` | Score one pairing with its drivers |
| `python -m synergy partners "a#tag"` | Best and worst modelled partners |
| `python -m synergy sequences` | Encode timelines into per player game tensors |
| `python -m synergy deep-train --epochs 300` | Train the timeline encoder, GPU if one is present |
| `python -m synergy deep-compare` | Score learned embeddings against the hand weighted axes |
| `python -m synergy synthetic --matches 4000 --players 200` | Synthetic corpus with a known synergy structure, for offline work |
| `python -m synergy all` | Synthetic corpus, features and training in one go |
| `python lolpredictordjango/manage.py runserver` | Serve the API on :8000 |
| `cd frontend && npm start` | Serve the dashboard on :3000, proxied to the API |
| `pytest` | Unit tests plus the synergy recovery check |
| `docker compose up --build` | API on :8000, dashboard on :3000 |
| `docker compose run --rm pipeline crawl` | Run any subcommand inside the API image |

The Django management command `manage.py synergy <subcommand>` wraps the same CLI.

## Riot API budget

A development key allows 20 requests per second and 100 per two minutes, so 3000 per hour, and expires
24 hours after issue. One crawled player costs 3 requests (rank, Riot ID, match list) and one new match
costs 2 (match, timeline), so `CRAWL_MAX_REQUESTS` is the honest knob: at the two-minute ceiling a run
sustains roughly 50 requests a minute, and the defaults in `.env.example` fit a single hour. Raise
`RIOT_RATE_*` for a production key.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/status/` | Model readiness, corpus size, training metrics |
| `GET /api/players/?q=&limit=` | Known players, most-seen first |
| `GET /api/players/<riot id or puuid>/` | Profile, style percentiles, behavioural traits, dashboard series |
| `GET /api/partners/<riot id or puuid>/?limit=` | Best and worst modelled partners |
| `GET /api/pair/?a=&b=` | One pairing: score, drivers, shared play, projected win rate |
| `POST /api/team/` | `{"players": [...]}` up to five, returns the pairwise matrix and group score |
| `POST /api/reload/` | Reload profiles and model from disk after a retrain |

## Data on disk

`DATA_DIR` (default `./data`) holds `raw/matches/*.json.gz`, `raw/timelines/*.json.gz`, the parquet
tables under `processed/`, and the pickled model plus `training_report.json` under `models/`. The crawl
frontier, matches, frames and events live in Postgres, not on disk. Nothing in `data/` is committed.
