# Movement

Players are distinguishable by where they go next given where they are, and that predicts
board advantage. [movement.py](../synergy/ml/movement.py) counts region transitions per
player from the frames, shrinks each player's rows toward the global matrix, and scores the
result on held-out player-games.

Shrinkage strength `kappa` comes from a bounded search on the closed-form
Dirichlet-multinomial marginal likelihood over all 169,952 rows; NUTS runs on a 20,000 row
subsample only for the interval. Sampling the full set is not worth it: one `kappa` gradient
touches every row, and two attempts at it burned five hours each without finishing.

Hold out whole player-games rather than single transitions. Steps inside one game are
correlated, so splitting transitions leaks; here it inflated the lift from 0.0958 to 0.0999.

| measurement | value |
|---|---|
| `kappa` | 30.01, interval [29.85, 30.87], r̂ 1.0053, n_eff 652 |
| held-out loglik, global matrix | -1.91590 |
| held-out loglik, per player raw | -75.42687 |
| held-out loglik, per player shrunk | **-1.82012** |
| lift over the global matrix | **+0.09579 nats per transition** |
| signature R², movement alone | +0.01027 |
| signature R², rank alone | +0.01808 |
| signature R², both | **+0.02628** |
| movement beyond rank | **+0.00820** |
| scrambled signature null, 40 draws | -0.00060 ± 0.00092, range -0.00277 to +0.00144 |
| null draws reaching the real value | **0 of 40**, so exact p ≤ 1/41 |

Raw per-player matrices score -75.4 because they give unseen transitions zero probability,
which is what the shrinkage is for. `kappa` of 30 against rows averaging 30 transitions means
a player's own history and the global prior carry about equal weight.

The signature is the log ratio of a player's shrunk matrix to the global one, reduced to 12
components, fitted on 21,917 matches and scored on 21,918 disjoint ones so no player's own
matches inform the outcome they are judged against. It adds 0.0082 R² on top of rank, which
on an outcome with sd 5,273 is a predicted component of roughly 480 gold.

Test that increment by scrambling which player gets which signature, leaving rank and the
outcome intact. Permuting the outcome instead destroys rank along with movement and never
asks whether movement adds anything beyond it. Report the count of null draws reaching the
real value, not a z: a ratio to a spread estimated from a few dozen draws says nothing about
a tail that was never sampled.

The same positional data carries nothing as pair chemistry: a pair's positional history
predicts a new match at 0.6 null spreads, inside the null. Position is an individual disposition here, and only in the
conditional form. Averaging a player's position across matches describes the situations they
met, not the player. See [variance.md](variance.md).
