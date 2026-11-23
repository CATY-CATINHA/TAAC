# Checkpoint 1 · Escolha global por LID

O CP0 (`a15791d`) conserva o ADERH original e o pre-subspace fixo, incluindo
os resultados observados nas seeds sintéticas 0–29 e no painel real. No CP1
testamos a hipótese de que a LID mediana do treino indica quando ativar os
subespaços. A regra e o protocolo são guardados antes das novas medições.

## Regra congelada

1. Ajustar o MinMaxScaler no treino, tal como no CP0.
2. Até dez features, usar o original. Acima de dez features, estimar a
   MLE-LID de cada ponto de treino com os 20 vizinhos mais próximos do
   mesmo treino, excluindo o próprio ponto.
3. Usar o pre-subspace se a mediana das estimativas finitas for estritamente
   superior a 10. O limiar é uma hipótese fixada pelo orçamento de dez
   features do candidato. Usar o original nos restantes casos, incluindo
   splits com até 20 observações ou com menos de metade das estimativas
   válidas. Uma vizinhança degenerada produz uma estimativa indefinida.
4. Treinar um único ensemble com 256 estimadores e 18 âncoras. Aplicar a
   decisão do treino a todos os pontos do teste. Guardar a decisão, a
   mediana e o número de estimativas válidas por split. A ausência de mediana
   indica que o cálculo foi dispensado ou que todas as estimativas falharam.

## Comparação fixada antes da execução

- Quatro políticas recebem o mesmo split e seed. As referências são o ADERH
  original, o pre-subspace fixo e uma regra de controlo que usa o pre-subspace
  sempre que o número de features excede 10. Esta última permite medir o
  que a LID acrescenta à simples contagem de features. As duas regras devolvem
  exatamente os scores do ramo escolhido. Podemos assim partilhar os scores
  dos dois ensembles em cada execução experimental.
- Sintético com oito dimensões e vinte seeds novas (30–49). Em cada dimensão
  medimos o delta emparelhado por seed contra cada controlo. Nos onze datasets
  ADBench repetimos os três splits e cinco seeds do CP0 e verificamos as
  métricas das duas referências face ao arquivo. A unidade da análise real
  é o dataset, após tirar a média das suas 15 execuções. Este painel já foi
  observado durante o desenvolvimento do projeto.
- AP é a métrica principal e ROC-AUC a complementar. Usamos IC marginal
  bootstrap de 95% para o delta médio e teste bilateral dos sinais com
  empates a 12 casas. As diferenças são sempre CP1 menos controlo.
  Holm abrange todos os 48 contrastes sintéticos ou os 6 contrastes reais,
  separadamente. A comparação direta com o pre-subspace mede o efeito da
  alteração; a comparação com o controlo por dimensão testa o contributo
  específico da LID.
- Guardar os resultados num checkpoint posterior. Qualquer alteração de
  limiar, vizinhança ou estratégia passa a uma versão experimental seguinte.

## Reprodução a partir da raiz

```bash
python -m pairing.experiments.pre_subspace fetch
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pairing.experiments.lid_gate real
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pairing.experiments.lid_gate synthetic
```

Cada comando de avaliação grava os CSV brutos, efeitos emparelhados e
metadados em `pairing/experiments/results/current/lid_gate_v1/`. A opção
`--output` permite escolher outra pasta.
