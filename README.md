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
   becomes an opportunity row with the situation lagged one minute behind the decision, including the
   filtered lane priority of top, mid and bot separately from the 10 second track, so a jungler's gank
   is conditioned on which lane holds priority rather than on a team average, a logistic model predicts
   the chance any player would act, and each player's observed over expected ratio is shrunk
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
   history. A ridge on the role indexed controls predicts advantage at 15 from individual strength; a
   second ridge then fits what the summed interaction vectors explain about the *residual*, which is what
   isolates fit from strength. Every profile input is leave-one-out. Role pairing is deliberately absent:
   both teams always field the same ten role combinations, so it cancels exactly in the team difference.
8. **Score** is `w . phi(pair)`, the pair's contribution to the team's advantage residual, reported as a
   percentile against every pairing in the corpus, so 50 is an average pairing. It is gated: the model
   reports no score unless the pair terms add held out R² that none of 20 permutation nulls reaches.
   See [docs/variance.md](docs/variance.md).

Frames and events live in Postgres 17 hypertables under TimescaleDB, everything else in ordinary
relational tables. See [docs/timescaledb.md](docs/timescaledb.md).

## What it measures, on 44,786 Master and Diamond matches

The target is advantage at 15, gold plus objectives, blue minus red. Nothing trains on the end of game
result: win arrives twenty minutes after the features, compresses a match into one bit shared across ten
pairs, and in a power simulation recovers a planted pair effect at only 1.5 null spreads where advantage
at 15 recovers it at 13.3. The corpus holds 44,919 matches, 13,110,860 frames and 50,087,274 events
across 39,433 players, every one season 16, patches 16.1 to 16.18. Mixing seasons is refused:
`qualifying_matches` drops anything outside the configured season on the patch string, and the rule stays
on even when the rank floor cannot decide.

Every input is a player's playstyle from their *other* games, leave one out, and none of it is an
average. It is learned per position: a seat's cells come from the player's other games in the position
they held in that match, shrunk toward that position's corpus row, so a top main autofilled to jungle
is described by their jungle games, which for most players means the jungle corpus row. A player who
plays two positions is two players here, since they neither stand in the same places nor answer the
same triggers in each. Each block keeps the player's distribution over outcomes within a situation,
Dirichlet shrunk toward the position's corpus row with a strength fitted from the data, exactly as the
movement transitions are:

| block | situations | outcomes | columns |
|---|---|---|---|
| reaction to a trigger | kill, plate, objective or building, by a teammate or an enemy, near, mid or far from the responder | converged, held, left, present, absent | 120 |
| objectives | dragon or grubs, taken by us or them, near, mid or far | died, fought, committed, rotated, approached, absent | 72 |
| wards | early, mid, late game | own side of a lane, its middle, the enemy side, own jungle, enemy jungle, river, other | 21 |
| jungle openings | first gank, first invade, sides | by 3, by 5, by 8, late, never; crossed, stayed | 12 |
| lane priority | every tick, the minute before an objective | own lane depth × opponent's depth × farming, five by six by two, plus off lane and dead | 124 |
| tendencies | which of top, mid and bot hold priority, eight patterns, × own or enemy half | observed over expected log ratio per action kind, initiate, follow, dive, fight_join and the rest | 144 |
| habit, movement, identity signature | conditional distributions and a learned embedding, as before | | 34 |

Never champion pool, never win rate, never a summary of the match being scored. A player has a couple
hundred games and a handful with any given duo, so the duo subset cannot carry a playstyle.

The pair question is asked in numpyro against the right baseline. Gold compounds, so two players who are
each ahead produce more than the sum, and a cross term picks that up with no synergy present. The test is
therefore three nested rungs on held out matches, every fit run to an ELBO plateau, each rung given at
least the steps the rung below it needed, and the pair term set aside if it fits the training matches
worse than solo at its posterior median, which is what a cross term shrunk to nothing looks like. The
ELBO itself is not the nesting check: on 527 cells the richer rung carries thousands more latent
loadings, so its ELBO sits a few nats above the poorer one even when the optimiser did its job. The
equal budget matters: on the per position cells solo held out 0.0367 after 40,000 steps and 0.0371
after 96,000, a move larger than any pair gain, so a pair rung that stopped earlier than solo would
have read as synergy.

| rung | held out R², per position | same, shallower corpus | pooled over positions | 16 principal components |
|---|---|---|---|---|
| linear, Σ w·z over the five seats, blue minus red | 0.0517 | 0.0368 | 0.0452 | 0.0303 |
| plus each player's own quadratic, z^T S z | 0.0516 | 0.0371 | 0.0452 | 0.0297 |
| plus the pair cross term, Σ z_a^T C z_b over the ten same team pairs | 0.0517 | 0.0370 | 0.0457 | 0.0310 |

Per position, which is how a seat is described now, the pair term finds nothing. On 46,619 matches with
9,324 held out it earns **+0.000002 R²** beyond each player alone, and the own quadratic earns −0.000007,
so all three rungs sit within one part in a hundred thousand of each other. Every rung was run to
96,000 steps and reached a drift near 1e-06; a 16,000 step fit of the same data gave the same three
numbers to four decimal places, so this is convergence, not a stopping artefact.

The scores it does produce are noise, and two fits show it. Between the 16,000 and 96,000 step fits of
the identical corpus, pair scores changed sign and crossed the whole percentile range: one top and
jungle pair went from the 98th percentile to the 4th, a top and mid pair from the 96th to the 29th. A
quantity that reorders itself under a change that leaves the held out R² unmoved is measuring optimiser
noise. This is why `pair_between` reports `reliable: false` and why the percentile must not be shown to
a user as a finding.

Deepening the corpus is what moved the linear rung, from 0.0368 to 0.0517, after three players were
crawled from roughly 150 games each to between 400 and 800. Better descriptions of individuals, no pair
effect. That is the cleanest statement of where the evidence stands.

The middle column is the same cells with a player pooled across every position they play, where the pair
term earned +0.00048 against 10 nulls that reached at most +0.000007. That fit was measuring the wrong
thing. 49% of the variance of its pair scores was the pair's main position combination alone, two players
of the same main position averaging −0.0028 against +0.0009 for different mains, because two top mains
share a team only when one is autofilled, the cells carried no position, and the cross term was the only
place that damage could go. Splitting by position removes the configuration, and the increment goes with
it. Part of the fall is power rather than structure: the linear rung dropped from 0.0452 to 0.0368 and
the share of seats with every column present from 0.79 to 0.67, since a player's evidence is now divided
among their positions. Deepening the corpus is the answer to that, not pooling positions back together.

The permutation null, when there is a positive gain to test, reshuffles seats across matches within role
and side, keeping the target and every seat's own distribution but breaking who was actually together.
Both rungs then fit nothing and the gain between them measures optimiser noise alone. That bounds the
noise in the gain, not the leakage of individual effects into the cross term, which is what the solo rung
is there to absorb.

On the previous representation, 69 leave one out scalars reduced to 16 principal components, the pair
term earned +0.0013 R² in its final run, and the same increment appeared four times through different
constructions: +0.0011 from an earlier held out ridge on style cross products against the same target,
+0.0011 on 60 columns before the tendency block was repaired, +0.0010 on 69, and +0.0013 there, with
none of 10 null draws reaching it. Decomposed, though, that `C` was unreadable, its weight sat on habit
and movement components rather than on anything named. On the named cells every cross term has a name,
but a pair's score is spread across thousands of them: for the reference roster the largest single term
contributes about one percent of the pair's score, so a pair is read through each player's cells rather
than through a handful of drivers.

The representation above replaces it. `z` is now the 527 named cells with no compression, and the
comparison is a rank 8 factored form, `C = U diag(s) Uᵀ` on the named cells directly, with `solo` the
same shape on each player's own cells. Fewer parameters than the dense 16 by 16 had per column, every
cross term readable as "this reaction on one player with that reaction on the other", and the pair
score `z_a^T C z_b` decomposes into its strongest named cross terms, which `pair` reports as `drivers`
alongside the `interaction` percentile. `C` is still symmetric, so a one directional reaction, one
player converging on the other's plays but not the reverse, remains the hinge's to show.

The first fit of these cells pooled a player across positions, and 49% of the variance of its pair
scores was the pair's main position combination alone, two players of the same main position averaging
−0.0028 against +0.0009 for different mains. Two top mains share a team only when one is autofilled,
the cells carried no position, and the cross term was the only place the autofill damage could go, so
it read duplicated profiles as anti synergy. The per position cells remove that configuration from
training, and the rungs above are the per position fit. A second effect stays by construction: every
team holds exactly one pair of each position combination, so the fit cannot pin the combinations'
baselines against each other, and a pair's percentile is therefore taken within its own position pair.

`pair` takes the position each player will play. The same position for both is refused, a player with
no games in the asked position is refused with a message naming the position. Each player's cells
carry a fitted share of their own evidence, the exposure weighted posterior weight of their own games
against the position's corpus row under the shrinkage strengths the blocks fitted; when that share is
under a half the answer warns by name, share and games, since the description is then more corpus row
than player. The answer carries each player's games and evidence share in that position either way.
`lineup` names one player per position, sums the ten pairs, ranked against every corpus team, and
carries the same warnings.

What this replaced tells you why. The same test with `z` pooled out of a transformer trained on the
outcome gave linear 0.671, solo 0.827, pair 0.819: a quality of play summary of the match itself, where
"both were doing well" is the whole story and the pair term had nothing left. And the first version of
that fit reported its −0.008 as a finding when neither rung had converged. Both are kept in
`data/models/` as the record of what not to do.

The shipped ridge, the path that actually produces `pair_score`, asks the same question through 49
columns of learned style cross products, style gaps, complementarity and shared history, on a baseline
of 387 role indexed controls that carry rank and playstyle but no win rate. It finds nothing:
`advantage_gain` −0.0002 R² with 1 of 20 permutation nulls above it, and `ridge_alpha` pinned at the top
of its grid, meaning the residual ridge wants the pair block shrunk harder than it is allowed to. So
`informative` is false and the score is withheld, while the numpyro test on the named blocks is the one
that finds the increment. Which of the two becomes the product's scoring path is an open decision;
until it is made, `pair_score` reports the ridge's withheld score alongside the interaction percentile
and the hinge.

Alongside the score, every pair with shared games gets a direct observable: the responder's rate of
converging on, being present for, or being near a teammate's kill, plate, objective or building when
*that* teammate was the actor, against the responder's own baseline, shrunk with the Gamma-Poisson prior.
That is `hinge`, built from 5.9M responder rows into 500,752 responder actor pairs with five or more
triggers.

## Learned representation

The strongest player signal is not hand written. A transformer over the first 16 minutes,
conditioned on champion and role as embeddings rather than as separate code paths, produces a
64 dimensional signature that adds 0.019 to 0.027 R2 on gold at 15 beyond rank across five
independent splits, with no draw of 40 permutation nulls ever reaching it. Fed into the team
controls out of fold it lifts `advantage_base_r2` from 0.0509 to 0.0622, and with every block
loaded, the win rate controls removed and the tendency block repaired, the shipped baseline reads
0.0729. It is the only feature in this project that clears both a permutation null and
replication. See [docs/representation.md](docs/representation.md).

## Match walk

`synergy/deep/` also reads a match as an ordered event sequence rather than as per minute summaries.
Every event under minute 15 becomes one token carrying its kind, the acting and suffering seats, the
region it happened in, the clock, the actor's position at that instant as a posterior over the 18
regions, and the actor's wave state as probabilities derived from that posterior. A 4 layer transformer
over up to 1024 tokens regresses advantage at 15, with the running gold lead zeroed out of its inputs so
it can only read actions. Building the sequence is `python -m synergy stream`; training is
`python -m synergy walk-train`.

Frames are 60 seconds apart, so between them a position is uncertain and the honest object is a
distribution, not a snapped point. Measured on 40,167 kill, plate and objective anchors that fell
strictly inside a frame gap and passed the reachability check, players do not move linearly: the anchor
sits nearer the earlier frame with probability close to one minus its time fraction, the nearer frame is
a median 1,547 units away against 2,147 for the linear interpolant, and the interpolant is the closer
guess only 40 percent of the time. So [posterior.py](synergy/features/posterior.py) uses a two component
mixture, one Gaussian around each bracketing known point weighted by time, with a per role spread that
grows with the time distance to that point, about 700 units three seconds from a frame and 1,700 to
2,500 at the middle of the minute, calibrated on those anchors. Known points are frames while alive,
kill victims at the kill, shop events at the fountain, plausible killer and assist claims, and the
fountain at respawn; a dead actor has no position. On the full corpus 91 percent of the 28.0M actor
tokens carry a posterior, the rest are dead, the top region holds 0.69 of the mass on average and the
top three hold 0.88, so close to half of all tokens are genuinely split between regions. The walk mixes
its region embeddings by that mass.
Wave state is the same push, defensive, neutral, off lane rule the feature tables use, applied to the
posterior: mass in the role's lane, the chance lane progress is past 1.0 or short of 0.85 under the
mixture, and whether at least four lane minions fell in the previous minute. A latent wave model was
tried before and retired at plate AUC 0.55. Minute resolution features stay exact because frames are
exact there; the movement transitions now exclude frames inside a death window.

Read the 0.0003 the wave channel adds inside the walk carefully. It says the channel is redundant with
position and CS in a model that already sees both and predicts gold, not that lane priority is
worthless. The place a skill lives is the player's leave one out playstyle, where the pair test can ask
whether it combines with a partner's. [priority.py](synergy/features/priority.py) samples every
player's position posterior on a 10 second grid, 90 ticks per player per match, 40.4 million rows for
the corpus in `positions_10s.parquet`, each row carrying the most likely region as an index into
`REGIONS`, −1 while dead, with the posterior mass on it, and derives lane
priority per tick as the chance the lane front, the midpoint of the two sides' lane coordinates under
their posteriors, sits on the enemy half; an empty enemy lane hands priority to whoever is there, and
a mid minute posterior gives a probability rather than a verdict, about a third of laner ticks come out
decisive and a fifth near a coin flip. From that track each laner gets priority share, priority in the
minute before an objective fell, the lane gap, push and defensive shares on the fine grid, and how often
a reset followed a push, all leave one out across their other games and shrunk toward the population.
The frames are still 60 seconds apart, so the grid is a common time base for the two sides and the
objective windows, not new information, and nothing finer than the frames and anchors is ever observed.
The block enters the pair test's `z` alongside the others.

Priority plays two roles and they must not share an information set. As a description of what
happened, the share of a game a lane was theirs, it is the smoothed estimate, the posterior that also
uses the frame after each tick. As context for what a player does next, a prior for the event, it must
be the filtered estimate, built only from frames and anchors at or before that tick, otherwise the
frame that records where the player went is already inside the condition and the reaction is scored
against a prior that contains it, the same error as reading a gold frame that came after the event.
The track carries both, `prio` and `prio_filtered`; on 27,292 laner ticks they differ by 0.14 on
average. Traits read the smoothed one, anything that conditions a tendency reads the filtered one,
lagged like every other situation covariate. Inferring priority from a roam and then scoring the roam
given priority is the circular form and is never done.

| | held out R² |
|---|---|
| walk, actions only, position as a posterior | **0.9486** |
| same walk on raw frames at the floored minute | 0.9453 |
| gold lead at the last event before 15, one number | 0.9033 |
| walk with `actor` shuffled across matches | −0.211 |
| walk with `region` shuffled | 0.596 |
| walk with the position posterior shuffled | 0.918 |
| walk with `kind` shuffled | 0.837 |
| walk with `wave` shuffled | 0.948 |

Shuffle who did each thing and it drops below zero, so it is reading who did what where. The posterior
is the channel the switch changed: shuffling the actor's region cost 0.004 R² when it was a raw frame
and costs 0.030 now, and skill over the gold lead baseline rose from 0.434 to 0.468. The wave channel
adds almost nothing on top of it, 0.0003, which is expected when the target is gold and the wave state
is derived from the same posterior. It is an accounting model for how actions become gold, not a
synergy model, and its per seat encodings are deliberately not the pair test's input for the reason
above. Earlier figures of 0.8511 AUC were from a version trained on the end of game result and are
withdrawn.

## Movement

Players are distinguishable by where they go next given where they are. Each player's region transition
matrix, per position, is shrunk toward their position's matrix, and held out by whole player-games that
beats the position matrix by 0.005 nats per transition once frames inside a death window are excluded.
Against one global matrix the same players were worth 0.102 nats, so most of what looked like personal
movement was where each position goes. Reduced to 12 components, the signature adds 0.0082 R² on gold at 15 beyond
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

GPU training needs `pip install -r requirements-deep.txt`. The base install stays CPU only. The
notebooks under `notebooks/` need `pip install -r requirements-notebooks.txt`, the repo root on
`PYTHONPATH`, and `DATA_DIR` set to the absolute data path when the kernel's working directory is not
the repo root, which is what `jupyter nbconvert --execute` does. `python -m ipykernel install --user
--name lolpredictor` registers the virtualenv as a kernel so any launcher picks it up.

## Commands

Run from the repo root with the virtualenv active. Copy `.env.example` to `.env` and set `RIOT_API_KEY`.

| Command | What it does |
|---|---|
| `python -m synergy crawl` | Seeded crawl from `SEED_RIOT_ID` against the Riot API, resumable, budget capped |
| `python -m synergy window` | Cut raw timelines down to the first 15 minutes |
| `python -m synergy features` | Extract participation, pair and opportunity tables from the windows |
| `python -m synergy blocks` | Build the movement, embedding, orphan, tendency, habit and hinge tables |
| `python -m synergy train` | Fit the baseline and synergy models, write profiles and the report |
| `python -m synergy variance` | Pair identity variance component on advantage at 15, gated at 5 shared games |
| `python -m synergy interaction --scores` | The three rung pair test on leave one out playstyle with its seat reshuffle null, then every player's style vector and every corpus pair score |
| `python -m synergy db-load` | Load the raw archive on disk into Postgres |
| `python -m synergy reingest --missing` | Rewrite frames and events from the raw timelines |
| `python -m synergy validate` | 21 corpus integrity checks, non-zero exit on any failure |
| `python -m synergy profiles` | Rebuild `player_profiles` from the corpus, no API calls |
| `python -m synergy ranks` | Look up rank for players that have none |
| `python -m synergy priors` | Conditional base rates for a behaviour given the matchup |
| `python -m synergy status` | Corpus counts, crawl frontier and model metrics |
| `python -m synergy pair "a#tag" "b#tag" --left-position top --right-position jungle` | Score one pairing in the positions they will play |
| `python -m synergy lineup --top "a#tag" --jungle "b#tag" --mid "c#tag" --bot "d#tag" --support "e#tag"` | Score a full five |
| `python -m synergy partners "a#tag"` | Best and worst modelled partners |
| `python -m synergy sequences` | Encode timelines into per player game tensors |
| `python -m synergy deep-train --epochs 300` | Train the timeline encoder, GPU if one is present |
| `python -m synergy stream` | Write the per match event sequence to `stream.npz` |
| `python -m synergy walk-train` | Regress advantage at 15 from the event sequence, actions only |
| `python -m synergy synthetic --matches 4000 --players 200` | Synthetic corpus with a known synergy structure, for offline work |
| `python -m synergy all` | Synthetic corpus, features and training in one go |
| `uvicorn synergy.api.server:app --reload` | Serve the API on :8000 |
| `cd frontend && npm start` | Serve the dashboard on :3000, proxied to the API |
| `pytest` | Unit tests plus the synergy recovery check |
| `docker compose up --build` | API on :8000, dashboard on :3000 |
| `docker compose run --rm pipeline crawl` | Run any subcommand inside the API image |

`blocks` runs after `features` and, for the embedding block, after `deep-train`. The loaders return an
empty frame when a table is missing, so skipping it trains a narrower model without complaining.

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
| `GET /api/pair/?a=&b=&a_position=&b_position=` | One pairing in the given positions: a 0 to 100 score where 50 is the average pair in that position pair, the synergy as gold at 15, and below them the interaction percentile, each player's reading, evidence warnings, games together and the hinge from shared games |
| `POST /api/team/` | `{"players": [...]}` up to five, returns the pairwise matrix and group score |
| `POST /api/lineup/` | `{"players": {"top": ..., "jungle": ..., "mid": ..., "bot": ..., "support": ...}}`, the ten pairs with their scores, then the team's score, gold at 15 and rank against every corpus team |
| `POST /api/reload/` | Reload profiles and model from disk after a retrain |

## Data on disk

`DATA_DIR` (default `./data`) holds `raw/matches/*.json.gz`, `raw/timelines/*.json.gz`, the parquet
tables and `stream.npz` under `processed/`, and the pickled model, `match_walk.pt` and the reports under
`models/`. Matches, frames, events, players and the crawl frontier live in Postgres, not on disk.
Nothing in `data/` is committed.

Every derived artefact is rewritten whole rather than versioned: a parquet table is truncated and
replaced, `stream.npz` is regenerated from scratch, and a recrawled match deletes its old
`participations`, `frames` and `events` rows before inserting. Rebuilding therefore needs nothing
cleaned up first. The one exception is Redis, which keys `corpus_quantiles` and outsider profiles
without the build digest, so those survive a rebuild for `CACHE_TTL_SECONDS` unless `POST /api/reload/`
clears them.
