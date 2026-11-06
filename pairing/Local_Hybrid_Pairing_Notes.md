
# TAAC — Análise do ADERH e plano para Local/Hybrid Pairing

## Objetivo

Este documento reúne o que foi confirmado até agora no projeto TAAC, os comandos usados para o investigar e o plano de trabalho da nossa parte: implementar e avaliar **Local Pairing** e **Fixed Hybrid Pairing**, preparando a integração posterior com o estimador de LID desenvolvido pelo Baía.

O objetivo é estudar estratégias alternativas para escolher os pares de pontos usados na construção das hiperesferas do ADERH, mantendo uma comparação controlada com o baseline original.

O plano global do grupo é:

1. **ADERH original** — baseline de referência.
2. **Local Pairing ADERH** — avaliar se a seleção local de parceiros ajuda, por si só.
3. **Fixed Hybrid Pairing ADERH** — combinar seleção local e aleatória, testando `alpha = 0.25, 0.5, 0.75`.
4. **LID-Aware Adaptive Pairing ADERH** — proposta principal do grupo: adaptar `alpha` à dimensionalidade intrínseca local, através de uma função do tipo `alpha(x) = f(LID(x))`.
5. **XAI / Interpretability** — extensão secundária, caso haja tempo, para explicar o LID, os valores de `alpha`, a origem local/aleatória dos pares e a contribuição das hiperesferas.

### Divisão de trabalho

- **A minha parte:** implementar e avaliar Local Pairing e Fixed Hybrid Pairing, incluindo testes, experiências e documentação.
- **Baía:** investigar e implementar o estimador de LID e analisar a dimensionalidade intrínseca local.
- **Trabalho conjunto posterior:** integrar o estimador de LID com as estratégias de pairing, de modo a implementar e avaliar o LID-Aware Adaptive Pairing.
- **XAI:** extensão secundária, a considerar depois de validar a experiência principal.

> **Estado atual:** Local Pairing e Fixed Hybrid Pairing ainda não foram implementados nem avaliados. A integração com LID também está por fazer. As ideias descritas abaixo são um plano de trabalho, não resultados experimentais.

---

## 1. Estrutura relevante do projeto

O repositório contém, entre outros, estes elementos:

```text
TAAC/
├── baseline_experiment/
│   ├── docs/
│   │   ├── aderh_internal_analysis.md
│   │   └── full_experiment_report.md
│   ├── figures/
│   ├── results/
│   ├── src/
│   │   └── aderh_distance/
│   │       ├── __init__.py
│   │       ├── analysis.py
│   │       ├── data_generation.py
│   │       ├── experiments.py
│   │       ├── metrics.py
│   │       └── visualization.py
│   ├── tests/
│   │   ├── test_aderh_integration.py
│   │   ├── test_analysis.py
│   │   ├── test_data_generation.py
│   │   ├── test_experiments.py
│   │   └── test_metrics.py
│   ├── pyproject.toml
│   └── requirements.txt
├── docs/
├── experiments/
├── lid_analysis/
├── pairing/
│   ├── hybrid_pairing/
│   │   └── README.md
│   ├── lid_adaptive_pairing/
│   │   └── README.md
│   ├── local_pairing/
│   │   └── README.md
│   └── README.md
├── results/
└── xai/
```

A estrutura foi observada anteriormente através da listagem recursiva do projeto. A branch indicada é `2/local-hybrid-pairing`. Os diretórios `pairing/local_pairing/` e `pairing/hybrid_pairing/` já existem, mas os respetivos README estavam vazios quando foram verificados. Confirmar o estado atual antes de editar.

### Comando usado para procurar a lógica de pairing no código do projeto

```bash
grep -RniE "random|pair|hypersphere|sphere|n_estimators" baseline_experiment/src
```

A pesquisa encontrou referências a pares e distâncias usadas nos diagnósticos, mas não encontrou a implementação interna da seleção dos pares do ADERH. A implementação do algoritmo está no pacote Python instalado.

---

## 2. Ambiente Python e dependência oficial

O pacote oficial foi instalado no ambiente virtual do projeto com:

```bash
python -m pip install -e ".[dev]"
```

A importação foi confirmada com:

```bash
python -c "import aderh; print(aderh.__file__)"
```

O resultado indicou que o pacote está no ambiente virtual do repositório TAAC, em `.../TAAC/.venv/Lib/site-packages/aderh/__init__.py`.

A classe `ADERH` está definida em `_aderh.py`. Para descobrir o caminho exato:

```bash
python -c "import inspect; from aderh import ADERH; print(inspect.getfile(ADERH))"
```

Resultado observado:

```text
.../TAAC/.venv/Lib/site-packages/aderh/_aderh.py
```

Para inspecionar o código-fonte da classe:

```bash
python -c "import inspect; from aderh import ADERH; print(inspect.getsource(ADERH))"
```

A implementação devolvida por este comando foi analisada e está resumida na secção seguinte.

---

## 3. O que faz a implementação oficial do ADERH

A classe tem estes parâmetros no construtor:

```python
def __init__(self, n_estimators=256, n=18, contamination=0.1,
             random_state=None):
    self.n_estimators = n_estimators
    self.n = n
    self.contamination = contamination
    self.random_state = random_state
```

Significado dos parâmetros:

- `n_estimators=256`: número de membros do ensemble.
- `n=18`: número de anchors selecionados por membro do ensemble.
- `contamination=0.1`: proporção esperada de anomalias, usada para definir o limiar e os labels depois do `fit`.
- `random_state`: controla a aleatoriedade e permite resultados reproduzíveis.

### 3.1 Seleção de anchors e cálculo das distâncias

O código oficial faz, dentro de `fit()`, o seguinte:

```python
anchor_idx = rnd.choice(n_samples, n_anchors, replace=False)
anchors = X[anchor_idx]

perm = rnd.choice(n_anchors, n_anchors, replace=False)
for j in range(n_anchors):
    if perm[j] == j:
        k = (j + 1) % n_anchors
        perm[j], perm[k] = perm[k], perm[j]

sq_pair_dist = np.sum((anchors - X[perm]) ** 2, axis=1)
```

Interpretação:

1. Seleciona `n` anchors distintos do conjunto de treino.
2. Cria uma permutação aleatória de índices.
3. Tenta remover pontos fixos da permutação através de trocas adjacentes.
4. Calcula as distâncias quadráticas entre `anchors` e `X[perm]`.

### 3.2 Detalhe importante: cálculo das distâncias versus construção dos centros

O cálculo das distâncias usa:

```python
sq_pair_dist = np.sum((anchors - X[perm]) ** 2, axis=1)
```

Contudo, a construção dos centros usa `anchors[perm]`, como mostrado na secção seguinte.

Isto significa que **o cálculo das distâncias e a construção dos centros não usam exatamente a mesma expressão para os parceiros**:

- Para as distâncias, o código usa `X[perm]`, que indexa a matriz de entrada através da permutação.
- Para os centros, o código usa `anchors[perm]`, que indexa o conjunto de anchors selecionados.

Por isso, não devemos resumir o comportamento oficial dizendo simplesmente que os parceiros usados na construção dos centros são `X[perm]`.

Este detalhe é importante para reproduzir o baseline. A implementação de referência deve preservar o comportamento oficial, incluindo esta diferença. Nas variantes Local e Hybrid, qualquer alteração na seleção dos parceiros ou na associação entre pares, centros e raios deve ser explícita, documentada e testada.

### 3.3 Construção das hiperesferas

A implementação guarda os centros e os raios quadráticos:

```python
self._centers[i] = np.concatenate(
    [anchors, anchors[perm]], axis=0)

self._sq_radii[i] = np.repeat(
    sq_pair_dist.reshape(-1, 1) / 2, 2, axis=1).reshape(-1)
```

Pontos importantes:

- Os centros são concatenados pela ordem `[anchors; anchors[perm]]`.
- Para cada distância quadrática calculada, o valor guardado como raio ao quadrado é `d_squared / 2`.
- Os raios quadráticos são duplicados numa ordem intercalada pelo código acima.
- Devido à diferença entre `X[perm]` no cálculo das distâncias e `anchors[perm]` na construção dos centros, não se deve presumir que cada centro fica associado ao raio intuitivamente correspondente ao par.

O baseline deve reproduzir a ordem oficial dos centros e dos raios exatamente. Se a implementação experimental usar uma representação mais explícita e coerente dos pares, essa diferença deve ser documentada, pois pode alterar o comportamento do algoritmo para além da escolha dos parceiros.

### 3.4 Cálculo dos scores

A função `_raw_scores()`:

- calcula as distâncias quadráticas dos pontos aos centros;
- identifica quais pontos ficam dentro das hiperesferas;
- determina a hiperesfera relevante mais próxima para cada ponto coberto;
- combina os componentes descritos no código como `Pitch` e `NDensity`;
- mantém score `1` para pontos não cobertos por nenhuma hiperesfera do membro do ensemble;
- calcula a média dos scores pelos membros do ensemble.

Em termos gerais:

- O `Pitch` relaciona a distância quadrática do ponto ao centro com o raio quadrático da hiperesfera selecionada.
- O `NDensity` considera as atribuições dos pontos às hiperesferas e a normalização pela densidade, de acordo com a implementação oficial.
- A contribuição dos pontos cobertos combina estes componentes.
- Os pontos não cobertos recebem contribuição `1`.

A direção dos scores é: **valores mais altos significam maior anomalia**.

O `decision_function()` também é dependente do conjunto de observações passado em cada chamada, porque a implementação recalcula contagens de atribuições usadas no `NDensity`. Para comparar variantes, devemos manter consistente a forma como os conjuntos de teste são enviados ao método de scoring.

A implementação oficial também disponibiliza `decision_function()`, `score_samples()`, `predict()` e `fit_predict()`.

### 3.5 Implicações para a implementação das variantes

A classe oficial não expõe um parâmetro público para escolher a estratégia de pairing. A seleção acontece internamente em `fit()`. Por isso, não basta passar um novo parâmetro a `ADERH`.

A implementação experimental deverá ficar separada, sem editar o pacote instalado em `site-packages`. O objetivo é alterar a estratégia de seleção dos pares mantendo o restante algoritmo tão comparável quanto possível.

Antes de implementar Local ou Hybrid, é necessário definir como representar os pares e associar cada par aos centros e raios. O modo de referência tem de reproduzir o comportamento oficial, incluindo as particularidades descritas acima. As variantes podem usar uma construção de pares mais explícita, mas qualquer diferença adicional deve ser documentada e considerada na interpretação dos resultados.

---

## 4. Configuração do projeto (`pyproject.toml`)

Comando usado:

```bash
cat pyproject.toml
```

Conteúdo relevante observado:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "aderh-distance-concentration"
version = "0.1.0"
description = "Reproducible study of distance concentration in the original ADERH algorithm"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "aderh==1.0.0",
    "matplotlib>=3.8,<4",
    "numpy>=1.24,<3",
    "pandas>=2.0,<3",
    "scikit-learn>=1.3,<2",
]

[project.optional-dependencies]
dev = ["pytest>=8,<10"]

[project.scripts]
aderh-experiment = "aderh_distance.experiments:main"
aderh-analyze = "aderh_distance.analysis:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
```

Conclusões:

- A versão do pacote oficial está fixada em `aderh==1.0.0`.
- O projeto usa `src/` como diretório de código Python.
- O pipeline de experiências pode ser iniciado através de `aderh-experiment`.
- A análise de resultados tem o comando `aderh-analyze`.
- O projeto já inclui `pytest` como dependência opcional de desenvolvimento.

---

## 5. O pipeline experimental e o baseline já avaliados

O ficheiro analisado foi `baseline_experiment/src/aderh_distance/experiments.py`.

O pipeline atual executa estas etapas:

1. Gera datasets sintéticos com `make_nested_datasets()`.
2. Usa `StratifiedShuffleSplit` para criar um split de treino/teste.
3. Reutiliza os mesmos índices de split entre dimensões, dentro de cada seed.
4. Ajusta `MinMaxScaler` apenas nos dados de treino e transforma treino e teste.
5. Calcula `relative_contrast()` como diagnóstico geométrico separado.
6. Cria o modelo oficial com `ADERH(random_state=seed)`.
7. Ajusta o modelo e mede o tempo de treino.
8. Calcula scores no conjunto de teste e mede o tempo de scoring.
9. Calcula ROC-AUC, Average Precision e diagnósticos dos scores.
10. Guarda os resultados brutos em CSV após cada configuração concluída.
11. Agrega os resultados por dimensionalidade e cria gráficos.
12. Guarda metadados, incluindo versões das bibliotecas, seeds, dimensões e configuração do dataset.

### Configurações importantes

```python
SMOKE_DIMENSIONS = (2, 50)
FULL_SEEDS = tuple(range(10))
```

O modo `smoke` usa as dimensões 2 e 50 com uma seed. O modo `full` usa as dimensões definidas em `DEFAULT_DIMENSIONS` e dez seeds.

O modelo é criado atualmente com:

```python
model = ADERH(random_state=seed)
```

Isto usa os valores por defeito do pacote oficial para os restantes hiperparâmetros.

### Métricas registadas

O resumo inclui:

- `roc_auc`
- `average_precision`
- `relative_contrast`
- `test_score_std`
- `test_unique_score_count`
- `test_uncovered_fraction`
- `normal_uncovered_fraction`
- `anomaly_uncovered_fraction`
- `fit_time_seconds`
- `scoring_time_seconds`

Também são guardados detalhes de prevalência de anomalias, número de amostras, hiperparâmetros, scores médios e outros diagnósticos.

### Interpretação de `score == 1.0`

De acordo com o comentário em `score_diagnostics()`, um score final exatamente igual a `1.0` indica que a observação não foi coberta por nenhuma hiperesfera em nenhum membro do ensemble, segundo a semântica de scoring do ADERH 1.0.0.

### Geração dos datasets

A implementação de `data_generation.py` foi inspecionada anteriormente. Os datasets são aninhados nas dimensões:

```python
(2, 5, 10, 20, 50, 100, 200, 500)
```

A configuração observada usa 2 000 amostras, contaminação de `0.05`, duas dimensões informativas, dados normais gerados a partir de `N([0, 0], I)`, anomalias deslocadas para `N([4, 4], I)` nas dimensões informativas e dimensões adicionais irrelevantes com distribuição independente `N(0, 1)` para ambas as classes. Confirmar estes valores no código se a geração dos dados for alterada.

### Resultados do baseline já concluído

O relatório completo foi executado em 11 de outubro de 2026, com 80 configurações: oito dimensionalidades e dez seeds, usando `aderh==1.0.0` sem modificações, parâmetros por defeito e `random_state` explícito.

| Dimensões | ROC-AUC (média) | Average Precision (média) | Fração de teste não coberta (média) |
|---:|---:|---:|---:|
| 2 | 0.9993 | 0.9833 | 0.0000 |
| 5 | 0.9945 | 0.8958 | 0.0000 |
| 10 | 0.9839 | 0.7978 | 0.0000 |
| 20 | 0.9506 | 0.6061 | 0.0002 |
| 50 | 0.8780 | 0.3023 | 0.0158 |
| 100 | 0.6990 | 0.0840 | 0.4725 |
| 200 | 0.5018 | 0.0502 | 0.9967 |
| 500 | 0.5000 | 0.0500 | 1.0000 |

Os valores da tabela são médias sobre dez seeds; o relatório completo inclui também desvios-padrão e outros diagnósticos.

Neste setup sintético, o desempenho degradou-se com o aumento da dimensionalidade. Nas 500 dimensões, todas as observações de teste receberam score `1.0`, pelo que o ROC-AUC ficou em `0.5` e a Average Precision em `0.05`, igual à prevalência de anomalias.

Estes resultados justificam investigar estratégias alternativas, mas **não demonstram antecipadamente** que Local ou Hybrid Pairing irão melhorar o desempenho. As conclusões são específicas à configuração testada e aos hiperparâmetros utilizados; não devem ser generalizadas automaticamente a todos os datasets.

Os resultados originais estão em `baseline_experiment/results/` e as figuras em `baseline_experiment/figures/`. Não devem ser sobrescritos pelas experiências das variantes.

---

## 6. Testes já existentes

Comando usado para listar os testes:

```bash
ls tests
```

Resultado observado:

```text
__pycache__/
test_aderh_integration.py
test_analysis.py
test_data_generation.py
test_experiments.py
test_metrics.py
```

Comando usado para procurar testes relacionados com ADERH, experiências, diagnósticos e pares:

```bash
grep -RniE "ADERH|run_experiment|score_diagnostics|pair" tests
```

Referências encontradas:

```text
tests/test_aderh_integration.py:
    test_official_aderh_defaults_and_anomaly_score_direction

tests/test_analysis.py:
    analyze_results

tests/test_data_generation.py:
    DEFAULT_DIMENSIONS, make_nested_datasets

tests/test_experiments.py:
    score_diagnostics

tests/test_metrics.py:
    relative_contrast
```

A pesquisa também encontrou correspondências nos ficheiros compilados dentro de `tests/__pycache__/`. Estas correspondências binárias não são testes adicionais e podem ser ignoradas.

### O que isto nos diz

- Já existe um teste de integração para os defaults do ADERH oficial e a direção dos scores.
- Existem testes para a geração dos dados, análise, métricas e diagnósticos experimentais.
- Não foram identificados testes dedicados à seleção interna dos pares do ADERH ou às variantes Local/Hybrid.

Antes de implementar, devemos ler os testes relevantes completos e executar a suite existente para estabelecer uma referência.

---

## 7. Variantes a implementar

### 7.1 Local Pairing ADERH

**Objetivo:** avaliar se selecionar parceiros localmente, com base nos vizinhos mais próximos (kNN), melhora os resultados relativamente ao ADERH original.

Antes de codificar, definir e documentar:

- a métrica de distância usada para encontrar vizinhos;
- como são escolhidos os candidatos a parceiro para cada anchor;
- se a própria anchor é excluída da lista de candidatos;
- se parceiros podem repetir-se;
- como lidar com empates e conjuntos pequenos;
- como controlar a aleatoriedade e garantir reprodutibilidade;
- como construir os pares e associar corretamente cada par aos centros e raios.

A seleção local deve usar apenas os dados de treino durante o `fit`; os dados de teste não podem influenciar a construção dos pares.

### 7.2 Fixed Hybrid Pairing ADERH

**Objetivo:** combinar a seleção local e a seleção aleatória, para avaliar se uma estratégia híbrida é vantajosa.

O plano acordado é testar:

```python
ALPHA_VALUES = (0.25, 0.5, 0.75)
```

A convenção proposta para a implementação é definir `alpha` como a proporção de pares locais:

- `alpha = 0.25`: 25% locais e 75% aleatórios;
- `alpha = 0.5`: 50% locais e 50% aleatórios;
- `alpha = 0.75`: 75% locais e 25% aleatórios.

Esta convenção deve ser confirmada e registada explicitamente no código e nos resultados. Também é necessário decidir como arredondar o número de pares de cada tipo, como evitar pares próprios ou duplicados se a especificação os excluir e como controlar a aleatoriedade.

### 7.3 Preparação para LID-Aware Adaptive Pairing

A proposta principal do grupo é adaptar a mistura de pares locais e aleatórios à dimensionalidade intrínseca local:

```text
alpha(x) = f(LID(x))
```

Esta fase depende do estimador de LID desenvolvido pelo Baía e de uma decisão conjunta sobre a função `f` e o significado dos seus valores.

Para facilitar a integração, a implementação de Local/Hybrid deve separar a estratégia de seleção de pares da lógica de scoring, documentar a interface necessária e permitir reutilizar a seleção local e aleatória já testadas. Não é necessário implementar o estimador de LID nesta fase.

### 7.4 XAI / Interpretability — extensão secundária

XAI é uma extensão secundária, a considerar depois de as variantes principais estarem implementadas e avaliadas. Os objetivos possíveis do grupo são:

- explicar o LID estimado para cada ponto;
- mostrar o valor de `alpha(x)`;
- identificar se os pares escolhidos foram locais ou aleatórios;
- identificar as hiperesferas que mais contribuíram para o anomaly score;
- opcionalmente, explorar SHAP ou perturbação de features.

Esta extensão não deve atrasar a implementação e avaliação principais.

---

## 8. Plano de trabalho concreto

### Fase 1 — Confirmar a referência atual

1. Confirmar a branch Git ativa e o estado do working tree.
2. Consultar os README de `pairing/local_pairing/` e `pairing/hybrid_pairing/`.
3. Ler `test_aderh_integration.py` e `test_experiments.py`.
4. Executar a suite de testes existente e registar o resultado.

Comandos úteis, executados a partir da raiz do repositório:

```bash
git branch --show-current
git status --short
cat pairing/local_pairing/README.md
cat pairing/hybrid_pairing/README.md
cat baseline_experiment/tests/test_aderh_integration.py
cat baseline_experiment/tests/test_experiments.py
python -m pytest baseline_experiment/tests
```

Se o ambiente virtual ou as dependências não estiverem ativos, ativar o `.venv` da raiz e confirmar que o comando usa o Python correto antes de interpretar falhas nos testes.

### Fase 2 — Definir a implementação experimental e reproduzir o original

1. Escolher onde ficará o código das variantes, sem editar `site-packages`.
2. Definir uma representação clara dos pares e a associação entre pares, centros e raios.
3. Implementar um modo de referência que reproduza fielmente a implementação oficial.
4. Criar um teste de equivalência num dataset pequeno e determinístico, comparando os scores ponto a ponto com os da implementação oficial.

Não basta obter métricas semelhantes: para testar equivalência, os scores devem coincidir sob a mesma configuração, seed, dados e batch de scoring.

Este teste é especialmente importante devido às particularidades de `X[perm]`, `anchors[perm]` e à ordem dos raios descritas na secção 3.

### Fase 3 — Implementar Local Pairing

1. Finalizar a regra de seleção local e documentar as decisões.
2. Implementar a seleção de parceiros locais.
3. Adicionar testes para a regra de seleção, reprodutibilidade, casos-limite e validade dos pares.
4. Confirmar que o scoring produz valores finitos e que valores maiores indicam maior anomalia.
5. Executar um `smoke test` antes das experiências completas.

### Fase 4 — Implementar Fixed Hybrid Pairing

1. Reutilizar a seleção local já testada e implementar a seleção aleatória.
2. Implementar a mistura com `alpha = 0.25, 0.5, 0.75`.
3. Testar a proporção efetiva de pares locais e aleatórios, incluindo o arredondamento.
4. Testar reprodutibilidade, casos-limite e ausência de pares próprios, se exigido pela especificação.
5. Executar um `smoke test` para cada valor de `alpha`.

### Fase 5 — Integrar no pipeline e comparar

Adicionar as variantes ao pipeline experimental sem alterar por defeito o comportamento do baseline. Guardar os resultados em ficheiros e prefixos próprios, sem sobrescrever os resultados originais.

A comparação deve usar, tanto quanto possível:

- as mesmas dimensões;
- os mesmos dados gerados;
- os mesmos índices de treino/teste;
- as mesmas seeds;
- o mesmo preprocessing;
- os mesmos hiperparâmetros do ensemble, exceto a estratégia de pairing;
- as mesmas métricas e convenções de scoring.

Executar primeiro experiências curtas de verificação. Só depois de confirmar que o código e os resultados estão corretos executar as experiências completas.

Analisar os resultados por dimensão, incluindo ROC-AUC, Average Precision, cobertura das hiperesferas e tempos de treino/scoring.

### Fase 6 — Preparar a integração com LID

Em conjunto com o Baía:

1. Definir a interface do estimador de LID.
2. Definir a função que transforma LID em `alpha(x)`.
3. Decidir em que etapa e para que pontos é calculado `alpha(x)`.
4. Integrar o estimador sem duplicar desnecessariamente o código de Local/Hybrid Pairing.
5. Avaliar a variante adaptativa com o mesmo protocolo experimental.

### Fase 7 — Documentar resultados

Registar a definição exata das variantes, hiperparâmetros usados, seeds, limitações, resultados por dimensão e diferenças relativamente ao baseline.

As conclusões devem refletir os resultados medidos, sem assumir previamente que Local, Hybrid ou LID-Aware Pairing são melhores.

---

## 9. Cuidados antes de começar a editar

- **Não modificar diretamente o pacote oficial instalado** em `site-packages`.
- **Não substituir os resultados existentes do baseline**; usar prefixos ou diretórios próprios.
- **Não alterar o modo de referência silenciosamente:** reproduzir a lógica oficial, incluindo o cálculo com `X[perm]`, os centros com `anchors[perm]` e a ordem dos raios.
- Não confundir `relative_contrast()` com a seleção de pares do ADERH: no pipeline atual, essa função é um diagnóstico geométrico separado.
- Manter as seeds, splits e preprocessing consistentes na comparação.
- Usar apenas os dados de treino para selecionar pares e evitar fuga de informação do teste.
- Documentar qualquer diferença inevitável entre as estratégias.
- Manter a implementação de Local/Hybrid independente do estimador de LID, para permitir trabalho em paralelo e integração posterior.
- Tratar XAI como extensão secundária, depois de validar a experiência principal.

---

## 10. Resumo do estado atual

### Confirmado

- O pacote `aderh==1.0.0` está instalado e a classe `ADERH` foi inspecionada.
- A seleção de pares ocorre dentro de `ADERH.fit()`, sem um parâmetro público para trocar a estratégia.
- O cálculo das distâncias usa `anchors` e `X[perm]`, enquanto a construção dos centros usa `anchors` e `anchors[perm]`; a ordem dos raios também faz parte do comportamento de referência.
- O pipeline de experiências está em `baseline_experiment/src/aderh_distance/experiments.py`.
- O baseline foi avaliado em oito dimensionalidades e dez seeds, com degradação acentuada neste setup sintético à medida que aumenta a dimensionalidade.
- O projeto já tem testes de integração e testes para os componentes do pipeline.
- Existem diretórios dedicados a Local Pairing, Hybrid Pairing e LID-adaptive pairing.
- O plano global do grupo coloca **LID-Aware Adaptive Pairing** como proposta principal; Local e Fixed Hybrid são etapas intermédias para avaliar as estratégias de pairing.

### Ainda por fazer na minha parte

1. Confirmar branch, estado do Git e testes de referência.
2. Consultar os README e os testes existentes.
3. Definir a interface experimental e provar equivalência com o ADERH original.
4. Implementar e testar Local Pairing.
5. Implementar e testar Fixed Hybrid Pairing para os três valores de `alpha`.
6. Integrar as variantes no pipeline e executar as comparações experimentais.
7. Documentar os resultados e preparar a integração com o LID do Baía.

### Próximo passo imediato

Confirmar a branch e o estado do Git; consultar os README das pastas de pairing e ler os testes existentes. Depois disso, executar os testes de referência antes de criar ou alterar ficheiros.
