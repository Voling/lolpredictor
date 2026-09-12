# Pair variance component

Asks whether a pair of players has an effect on winning beyond the two players
individually. One observation per match, both teams entering as a difference, fitted with
numpyro SVI in [variance.py](../synergy/ml/variance.py):

```
logit P(blue wins) = mean + Σ player[blue] − Σ player[red] + Σ pair[blue] − Σ pair[red]
```

There is no match random effect. `win` takes exactly one distinct value per team-match, so
a team-match term is a saturated model of the response: it fits the outcome perfectly,
leaves nothing for the players or the pair, and its scale runs away instead of converging.
Pairs below `MIN_PAIR_GAMES` are pinned to zero rather than dropped, because dropping their
rows also drops the other nine pairs sharing that match.

Every scale is reported against a null from permuting `win` across matches, fitted with
identical settings. A raw scale is not interpretable: mean-field SVI puts `pair_scale` at
0.263 with `LEARNING_RATE` 0.02 and 0.50 with 0.1 on the same data, and the posterior
interval is narrower than the spread between null draws.

| quantity | real | null mean | null sd | z | nulls above real |
|---|---|---|---|---|---|
| `player_scale`, floor 5 | 0.30646 | 0.27153 | 0.00243 | **14.37** | 0 of 8 |
| `pair_scale`, floor 5 | 0.26255 | 0.25017 | 0.00882 | 1.40 | 1 of 8 |
| `pair_scale`, floor 5, 10 nulls | 0.26255 | 0.25396 | 0.00620 | 1.39 | 0 of 10 |
| `pair_scale`, floor 12 | 0.16382 | 0.15872 | 0.00576 | 0.89 | 1 of 10 |

Individual skill separates from its null at z=14, so the model detects what it should. A
pair effect does not, and the z falls as the pair floor rises, which is the opposite of a
real effect emerging from cleaner data. Corpus is 42,816 matches and 35,857 players, with
1,924 pairs at 5 or more games together.
