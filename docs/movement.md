# Movement

Players are distinguishable by where they go next given where they are, and that predicts
board advantage. [movement.py](../synergy/ml/movement.py) counts region transitions per
player and position from the frames, skipping any frame that falls inside a death window so a
corpse never counts as a step, shrinks each player's rows toward their position's matrix, and
scores the result on held-out player-games. The signature R² rows at the bottom of the table were
measured before the death masking and the per position split, and stand until that test is rerun.

Shrinkage strength `kappa` comes from a bounded search on the closed-form
Dirichlet-multinomial marginal likelihood over all 156,836 rows, and its interval from a Laplace
approximation on log `kappa` under a log normal prior, mean 3 and sd 1.5 on the log scale. NUTS
used to give that interval from a 20,000 row subsample; after the per position split it ran 48
minutes without finishing one fit, and one `kappa` gradient touches every row, so it was replaced.

Hold out whole player-games rather than single transitions. Steps inside one game are
correlated, so splitting transitions leaks; here it inflated the lift from 0.0958 to 0.0999.

Splitting by position changed the reading of the lift. Against one global matrix a player's own
transitions were worth +0.102 nats per transition; against their position's matrix they are worth
+0.005, so most of what looked like personal movement was where junglers, laners and supports
each go, and `kappa` rose from 28 to 307 because a position row is already close to its players.

| measurement | value |
|---|---|
| `kappa` | 307.10, Laplace interval [300.76, 313.48] |
| held-out loglik, position matrix | -1.60732 |
| held-out loglik, per player and position raw | -68.90624 |
| held-out loglik, per player and position shrunk | **-1.60254** |
| lift over the position matrix | **+0.00478 nats per transition**; the earlier +0.10195 was against one global matrix |
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
