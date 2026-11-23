# LID-Adaptive Pairing ADERH

`LIDAdaptiveADERH` estimates MLE-LID once for every training observation and
uses the sampled anchors' LID ranks to decide which anchors receive local
pairing constraints.

Two directions are available:

- `high_local`: high-LID anchors are more likely to use local pairing;
- `high_random`: low-LID anchors are more likely to use local pairing, leaving
  high-LID anchors in the random component.

Two strategies are available:

- `rank`: exactly `round(alpha * n)` anchors are local, selected by LID rank;
- `continuous`: each anchor makes a Bernoulli decision from a rank-based
  probability. The probability mapping has mean `alpha`, so `alpha` is the
  expected local proportion. `alpha=0` is fully random and `alpha=1` fully
  local.

After local positions are chosen, partners are assigned with the same
one-to-one sampled-anchor assignment used by the Local and Hybrid variants.

```python
from pairing.lid_adaptive_pairing.lid_adaptive_aderh import LIDAdaptiveADERH

model = LIDAdaptiveADERH(
    n_estimators=256,
    n=18,
    contamination=0.1,
    random_state=42,
    k_neighbors=5,
    alpha=0.5,
    lid_k=20,
    direction="high_local",
    strategy="continuous",
)
model.fit(X_train)
```

The method ranks LID within each estimator's sampled anchors. It does not yet
learn `alpha`, `lid_k`, or the direction from data.
