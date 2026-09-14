# Pair variance component

Asks whether a pair of players has an effect beyond the two players individually. One
observation per match, both teams entering as a difference, in
[variance.py](../synergy/ml/variance.py) for `win` and [gold.py](../synergy/ml/gold.py) for
the minute 15 outcomes:

```
outcome = mean + Σ player[blue] − Σ player[red] + Σ pair[blue] − Σ pair[red]
```

There is no match random effect: `win` takes one distinct value per team-match, so a term at
that level is a saturated model of the response and its scale runs away instead of
converging. Pairs below `MIN_PAIR_GAMES` are pinned to zero rather than dropped, because
dropping their rows also drops the other nine pairs sharing that match.

Read every scale against its permutation null, never against zero, and check the power row
before reading a null as evidence. A raw `pair_scale` sits near its prior when nothing is
there, and it also moves with the optimiser: `LEARNING_RATE` 0.02 gives 0.263 where 0.1
gives 0.50 on identical data.

| outcome | `player_scale` z | `pair_scale` z | detects true 0.15 | detects true 0.25 |
|---|---|---|---|---|
| `win` | +14.37 | +1.39 | z=+0.97 | z=+1.50 |
| gold @15 | **+12.61** | **+0.10** | **z=+6.98** | **z=+13.34** |
| gold + objectives @15 | +10.17 | +0.38 | — | — |

The `win` result is uninformative. Simulating a known pair effect through the real design
recovers z=+1.50 for an effect of 0.25, nearly as large as the player effect, so that test
cannot see a pair effect of any plausible size and its null says nothing either way.

Gold at minute 15 is 7 to 9 times more sensitive, tightening the null spread from ±0.00904
to ±0.00218, because a continuous outcome measured when the features are measured carries
far more than one bit shared across ten pairs. Its null is therefore a real bound: **a pair
effect above roughly 0.15 sd, about 600 gold per pairing or 1,900 across a team, is ruled
out.** Below about 0.10 the estimator sits on its prior floor and still cannot separate.

Adding objectives priced from the learned evaluation weights (a dragon at 1,228 gold, a grub
at 100) raises the outcome sd from 4,165 to 5,273 and moves `pair_scale` from z=+0.10 to
z=+0.38, which is not a detection. Corpus is 42,689 matches, 35,857 players and 1,924 pairs
with 5 or more games together. See [complement.md](complement.md), which reaches the same
conclusion from the policy columns.
