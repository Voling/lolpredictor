# Learned representation

A 3 layer transformer over the first 16 minutes carries a map region token, an event region
token and 24 numbers per minute, conditioned on champion and role as embeddings rather than
as separate code paths. It is trained in [deep/](../synergy/deep/) with a contrastive loss on
player identity, so it needs no win label and no hand written features.

Its 64 dimensional output beats every hand built alternative at predicting gold plus
objectives at minute 15, and it is the only thing in this project that clears both bars: 40
permutation nulls and replication across independent splits.

| split | embedding alone | rank alone | beyond rank | nulls above |
|---|---|---|---|---|
| 0 | +0.024572 | +0.015432 | **+0.019257** | 0 of 40 |
| 1 | +0.023354 | +0.018566 | +0.019917 | 0 of 40 |
| 2 | +0.033560 | +0.023055 | +0.027143 | 0 of 40 |
| 3 | +0.029750 | +0.020823 | +0.024456 | 0 of 40 |
| 4 | +0.029162 | +0.017356 | +0.025841 | 0 of 40 |

Signatures come from one random half of matches and are scored on the disjoint half, so no
player's own game informs the outcome they are judged against. Inside the model the
signature enters `CONTROL_COLUMNS` out of fold through
[embedding.py](../synergy/ml/embedding.py), five folds against a basis fitted once so
components stay comparable, and lifts `advantage_base_r2` from 0.050938 to 0.062173.

Read that gain on gold, not on wins. When the pair model still trained on the end of game
result, the same change moved its AUC by -0.0003 and log loss by -0.0006, because a binary
whole game outcome shared across ten pairs discards most of what the representation knows.
Every improvement measured that way had the same shape, which is one of the reasons win is
no longer a training target anywhere in the project.

The encoder still trains on every game including the halves it is scored against. Its loss
never sees the outcome, only player identity, so the target is not exposed, but the clean
version fits the encoder on one half of matches and scores the other.

Hand written features aimed at one role do not survive the same tests. Jungle camp order is
close to universal: a per jungler transition matrix over 14 camps learned from 100,791 clears
lifts prediction of the next camp by 0.00434 nats where general movement lifts 0.09579, and
its signature loses to 16 of 40 nulls on the outcome. What varies between junglers is timing
and deviation, not sequence. See [movement.md](movement.md).
