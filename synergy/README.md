# synergy

The pipeline and model. Crawls ranked games, cuts them to the first fifteen minutes, extracts
behavioural features, and fits a two-stage model that separates individual strength from whatever a
specific pairing adds.

## Stages

| Stage | Module | Does |
|---|---|---|
| Crawl | [ingest/seeder.py](ingest/seeder.py) | Walks a priority frontier from the apex ladders, stores match and timeline JSON, respects both Riot rate windows |
| Window | [ingest/window.py](ingest/window.py) | Cuts every timeline to 0:00-15:00 so laning behaviour is comparable across games |
| Features | [features/build.py](features/build.py) | Runs 15 extractors over each match in parallel, streaming 16 tables to parquet |
| Posterior | [features/posterior.py](features/posterior.py) | Position between frames as a time weighted mixture over the bracketing known points, read as mass over regions; wave state derived from it |
| Traits | [features/traits.py](features/traits.py) | Learns the style basis by maximising between-player over within-player variance |
| Model | [ml/model.py](ml/model.py) | Ridge on rank and playstyle predicts advantage at 15, a second ridge on the residual uses pair terms only |

## The corpus filter

`qualifying_matches` drops any match with fewer than two ranked participants or an average below
`CORPUS_MIN_AVERAGE_LP` (Master 0 LP, 2800), and any match whose patch falls outside `CRAWL_SEASON`.
The season rule is decided on the patch string and stays on even when the rank floor cannot decide.
Games below the floor stay on disk and remain queryable, they simply never reach `participations`.
That is how sub-Master players can be evaluated without entering the corpus.

## Position certainty

Frames are 60 seconds apart, so a champion's position between them is barely constrained: median
error against known positions is 1,324 units, about one vision radius. Three things narrow it.

`anchors.py` extracts positions the data makes certain: a kill victim is at the kill, a shop event
means the fountain, plates and objectives pin their killer. Claims from killers and assists are
checked against a physical reachability bound first, which is what filters out global abilities;
Gangplank, Shen, Karthus and Soraka top the rejection list without the code knowing anything about
champions. Deaths open a window, from the level-based respawn timer, during which the recorded
position is a corpse rather than a player, and `alive_at` excludes those from proximity features.

68.7% of accepted anchors land in a region neither surrounding frame shows, so most of this
certainty is invisible to frame sampling alone. Where a position is needed between frames, the
event walk reads a posterior over regions built from these anchors rather than a snapped point;
the top level README's Match walk section carries the measurements behind its form.

## What it measures

Nothing trains on the end of game result. The target is advantage at 15, gold plus objectives,
blue minus red. The pair term is `advantage_gain`, the held out R² the pair block adds over the
rank and playstyle baseline, and `informative` is true only when none of 20 permutation nulls
reaches it; otherwise `pair_score` withholds the score. Every input is a player's leave one out
playstyle from their other games, never win rate, not even as a strength control. The current
value lives in `data/models/training_report.json` and the top level README carries the full three
rung test.

Individual signal is solid by contrast. Trait reliability runs 0.59 to 0.91 split-half, `policyvec`
keeps 18 conditional action cells above 0.50, and action coupling between bot and mid sits at 40x
its shuffled control. On the win target the pair effect never appeared; on advantage at 15 the
style cross term is small and clear of its null.

## Commands

| Command | Does |
|---|---|
| `python -m synergy crawl --leaderboard` | Seed from the apex ladders and crawl outward |
| `python -m synergy crawl --riot-id "x#tag" --no-discover` | Crawl one named player, add nobody to the frontier |
| `python -m synergy window` | Cut timelines to fifteen minutes |
| `python -m synergy features --workers 8` | Extract every table, 8 processes |
| `python -m synergy train` | Fit traits, profiles and the pair model |
| `python -m synergy status` | Corpus counts, frontier state, model metrics |
| `python -m synergy pair "a#tag" "b#tag" --left-position top --right-position jungle` | Score one pairing in the positions they will play |
| `python -m synergy lineup --top ... --jungle ... --mid ... --bot ... --support ...` | Score a full five |
| `python -m synergy partners "a#tag"` | Best and worst modelled partners |
| `python -m synergy sequences` / `deep-train` | The timeline encoder, see below |

`--workers 1` forces the serial build, which takes about five times longer.

## The deep branch

`deep/` trains a transformer over timeline sequences with a contrastive loss on player identity. It
is not wired into the model, and a measurement is the reason. With champion and role injected into
the CLS token it retrieves the same player at 40x chance, but a fresh linear probe recovers champion
from the embedding at 100%, so the retrieval was reading the pick. With `identity_conditioning`
off, champion recovery falls to 13% and player retrieval falls to 3x chance. The simple conditional
action distributions in `policyvec.py` are the stronger representation.
