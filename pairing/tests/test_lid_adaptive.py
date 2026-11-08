
import numpy as np

from sklearn.datasets import make_blobs

# Ajusta este import ao módulo onde criaste a classe.
from pairing.lid_adaptive_pairing.lid_adaptive_aderh import LIDAdaptiveADERH


# Dataset artificial para testar o funcionamento.
X, _ = make_blobs(
    n_samples=200,
    n_features=5,
    centers=3,
    random_state=42,
)

common_params = {
    "n_estimators": 10,
    "n": 18,
    "contamination": 0.1,
    "random_state": 42,
    "k_neighbors": 5,
    "alpha": 0.5,
    "lid_k": 20,
}

# Testar as duas direções.
model_high_local = LIDAdaptiveADERH(
    **common_params,
    direction="high_local",
)

model_high_random = LIDAdaptiveADERH(
    **common_params,
    direction="high_random",
)

model_high_local.fit(X)
model_high_random.fit(X)

# Verificar os resultados.
scores_local = model_high_local.decision_scores_
scores_random = model_high_random.decision_scores_

assert model_high_local.lid_values_.shape == (200,)
assert model_high_random.lid_values_.shape == (200,)

assert np.all(np.isfinite(scores_local))
assert np.all(np.isfinite(scores_random))

assert len(model_high_local._centers) == 10
assert len(model_high_random._centers) == 10

print("LID Adaptive ADERH - Testes concluídos!")
print("LID shape:", model_high_local.lid_values_.shape)
print("Scores high_local:", scores_local.shape)
print("Scores high_random:", scores_random.shape)
print("Scores diferentes:", not np.allclose(scores_local, scores_random))
