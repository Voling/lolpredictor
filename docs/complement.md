# Policy complement columns

`policy_joint`, `policy_solo` and `policy_distance` in
[complement.py](../synergy/features/complement.py) score how two players' action priors
combine. None of them survives a permutation null.

Judge every z against the shuffled column, never against zero. Pair rows carry ten
correlated observations per team-match, so a logistic fit that treats rows as independent
understates its standard errors: permuting the outcome at the team-match level puts the
`policy_distance` null at **+1.21 ± 1.08**, not at zero, and one null seed reached +3.08.

| model, 159,908 held-out rows | policy_joint | policy_solo | policy_distance | rows_min |
|---|---|---|---|---|
| joint alone | +2.69 | | | |
| joint controlling solo | -1.10 | +2.19 | | |
| joint, solo, distance | -1.52 | +2.67 | +7.20 | |
| distance alone | | | +7.01 | |
| distance controlling games volume | | | +2.00 | +1.14 |
| all four | -1.95 | +2.84 | +3.23 | +10.53 |
| shuffled null, 8 seeds | -2.00 … +0.48 | | +1.21 ± 1.08 | |

`policy_joint` reaches +2.69 alone and flips to -1.10 once `policy_solo` is controlled,
because the two are 94% collinear. An earlier +4.16 came from a `policy_solo` that summed
the same player twice and never included the partner, leaving the partner's solo
contribution inside `policy_joint`. `policy_distance` reaches +7.01 alone and falls to
+2.00 against games volume, which is inside its own null; it correlates 0.34 with the
smaller player's row count, so most of it is how much data a player has rather than how
differently the two behave.

Fitted on the 42,816 match corpus with priors from the training matches and evaluation on
the value model's holdout. See [variance.md](variance.md), which reaches the same
conclusion from the variance components.
