# TAAC --- Análise do ADERH e plano para Local/Hybrid Pairing

## Objetivo

Este documento reúne o que foi confirmado até agora no projeto TAAC, os
comandos usados para o investigar e o plano proposto para implementar e
avaliar **Local Pairing** e **Hybrid Pairing**.

O objetivo é estudar estratégias alternativas para escolher os pares de
pontos usados na construção das hiperesferas do ADERH, sem perder a
possibilidade de comparar os resultados com o baseline original.

> **Estado atual:** ainda não foram implementadas nem avaliadas as
> variantes Local Pairing e Hybrid Pairing. As ideias descritas abaixo
> são um plano de trabalho, não resultados experimentais.

------------------------------------------------------------------------

## 1. Estrutura relevante do projeto

O repositório contém, entre outros, estes elementos:

``` text
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
│   ├── lid_adaptive_pairing/
│   ├── local_pairing/
│   └── README.md
├── results/
└── xai/
```

A estrutura foi observada anteriormente através da listagem recursiva do
projeto. Os diretórios `pairing/local_pairing/` e
`pairing/hybrid_pairing/` já existem como áreas dedicadas às
estratégias.

### Comando usado para procurar a lógica de pairing no código do projeto

``` bash
grep -RniE "random|pair|hypersphere|sphere|n_estimators" baseline_experiment/src
```

A pesquisa encontrou referências a pares e distâncias usadas nos
diagnósticos, mas não encontrou a implementação interna da seleção dos
pares do ADERH. A implementação do algoritmo está no pacote Python
instalado.

------------------------------------------------------------------------

## 2. Ambiente Python e dependência oficial

O pacote oficial foi instalado no ambiente virtual do projeto com:

``` bash
python -m pip install -e ".[dev]"
```

A importação foi confirmada com:

``` bash
python -c "import aderh; print(aderh.__file__)"
```

O resultado indicou que o pacote está no ambiente virtual do repositório
TAAC, em `.../TAAC/.venv/Lib/site-packages/aderh/__init__.py`.

A classe `ADERH` está definida em `_aderh.py`. Para descobrir o caminho
exato:

``` bash
python -c "import inspect; from aderh import ADERH; print(inspect.getfile(ADERH))"
```

Resultado observado:

``` text
.../TAAC/.venv/Lib/site-packages/aderh/_aderh.py
```

Para inspecionar o código-fonte da classe:

``` bash
python -c "import inspect; from aderh import ADERH; print(inspect.getsource(ADERH))"
```

A implementação devolvida por este comando foi analisada e está resumida
na secção seguinte.

------------------------------------------------------------------------

## 3. O que faz a implementação oficial do ADERH

A classe tem estes parâmetros no construtor:

``` python
def __init__(self, n_estimators=256, n=18, contamination=0.1,
             random_state=None):
    self.n_estimators = n_estimators
    self.n = n
    self.contamination = contamination
    self.random_state = random_state
```

Significado dos parâmetros:

-   `n_estimators=256`: número de membros do ensemble.
-   `n=18`: número de anchors selecionados por membro do ensemble.
-   `contamination=0.1`: proporção esperada de anomalias, usada para
    definir o limiar e os labels depois do `fit`.
-   `random_state`: controla a aleatoriedade e permite resultados
    reproduzíveis.

### 3.1 Seleção de anchors e parceiros

O código oficial faz, dentro de `fit()`, o seguinte:

``` python
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

1.  Seleciona `n` anchors distintos do conjunto de treino.
2.  Cria uma permutação aleatória de índices.
3.  Tenta remover pontos fixos da permutação através de trocas
    adjacentes.
4.  Calcula as distâncias quadráticas entre `anchors` e `X[perm]`.

### 3.2 Detalhe de implementação que deve ser preservado no baseline

O código usa:

``` python
X[perm]
```

e **não**:

``` python
anchors[perm]
```

Assim, os parceiros são selecionados a partir das linhas `0` a `n-1` do
conjunto de dados, indexadas pela permutação, e não necessariamente a
partir do conjunto de anchors escolhido aleatoriamente.

O próprio código contém comentários a identificar esta semântica como
comportamento de referência preservado. Não devemos alterar esse
comportamento silenciosamente quando reproduzirmos o baseline. Se as
variantes usarem outra regra de seleção, a diferença deve ser explícita
e documentada.

### 3.3 Construção das hiperesferas

A implementação guarda os centros e os raios quadráticos:

``` python
self._centers[i] = np.concatenate(
    [anchors, anchors[perm]], axis=0)

self._sq_radii[i] = np.repeat(
    sq_pair_dist.reshape(-1, 1) / 2, 2, axis=1).reshape(-1)
```

O código duplica os centros e os raios de cada par. O raio ao quadrado
corresponde a metade da distância quadrática entre os dois pontos do
par.

### 3.4 Cálculo dos scores

A função `_raw_scores()`:

-   calcula as distâncias quadráticas dos pontos aos centros;
-   identifica quais pontos ficam dentro das hiperesferas;
-   determina a hiperesfera relevante mais próxima para cada ponto
    coberto;
-   combina os componentes descritos no código como `Pitch` e
    `NDensity`;
-   mantém score `1` para pontos não cobertos por nenhuma hiperesfera do
    membro do ensemble;
-   calcula a média dos scores pelos membros do ensemble.

A direção dos scores é: **valores mais altos significam maior
anomalia**.

O código oficial também disponibiliza `decision_function()`,
`score_samples()`, `predict()` e `fit_predict()`.

------------------------------------------------------------------------

## 4. Configuração do projeto (`pyproject.toml`)

Comando usado:

``` bash
cat pyproject.toml
```

Conteúdo relevante observado:

``` toml
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

-   A versão do pacote oficial está fixada em `aderh==1.0.0`.
-   O projeto usa `src/` como diretório de código Python.
-   O pipeline de experiências pode ser iniciado através de
    `aderh-experiment`.
-   A análise de resultados tem o comando `aderh-analyze`.
-   O projeto já inclui `pytest` como dependência opcional de
    desenvolvimento.

------------------------------------------------------------------------

## 5. O pipeline experimental atual

O ficheiro analisado foi
`baseline_experiment/src/aderh_distance/experiments.py`, obtido com:

``` bash
cat src/aderh_distance/experiments.py
```

O pipeline atual executa estas etapas:

1.  Gera datasets sintéticos com `make_nested_datasets()`.
2.  Usa `StratifiedShuffleSplit` para criar um split de treino/teste.
3.  Reutiliza os mesmos índices de split entre dimensões, dentro de cada
    seed.
4.  Ajusta `MinMaxScaler` apenas nos dados de treino e transforma treino
    e teste.
5.  Calcula `relative_contrast()` como diagnóstico geométrico separado.
6.  Cria o modelo oficial com `ADERH(random_state=seed)`.
7.  Ajusta o modelo e mede o tempo de treino.
8.  Calcula scores no conjunto de teste e mede o tempo de scoring.
9.  Calcula ROC-AUC, Average Precision e diagnósticos dos scores.
10. Guarda os resultados brutos em CSV após cada configuração concluída.
11. Agrega os resultados por dimensionalidade e cria gráficos.
12. Guarda metadados, incluindo versões das bibliotecas, seeds,
    dimensões e configuração do dataset.

### Configurações importantes

``` python
SMOKE_DIMENSIONS = (2, 50)
FULL_SEEDS = tuple(range(10))
```

O modo `smoke` usa as dimensões 2 e 50 com uma seed. O modo `full` usa
as dimensões definidas em `DEFAULT_DIMENSIONS` e dez seeds.

O modelo é criado atualmente com:

``` python
model = ADERH(random_state=seed)
```

Isto usa os valores por defeito do pacote oficial para os restantes
hiperparâmetros.

### Métricas registadas

O resumo inclui:

-   `roc_auc`
-   `average_precision`
-   `relative_contrast`
-   `test_score_std`
-   `test_unique_score_count`
-   `test_uncovered_fraction`
-   `normal_uncovered_fraction`
-   `anomaly_uncovered_fraction`
-   `fit_time_seconds`
-   `scoring_time_seconds`

Também são guardados detalhes de prevalência de anomalias, número de
amostras, hiperparâmetros, scores médios e outros diagnósticos.

### Interpretação de `score == 1.0`

De acordo com o comentário em `score_diagnostics()`, um score final
exatamente igual a `1.0` indica que a observação não foi coberta por
nenhuma hiperesfera em nenhum membro do ensemble, segundo a semântica de
scoring do ADERH 1.0.0.

### Geração dos datasets

A implementação de `data_generation.py` foi inspecionada anteriormente.
Os datasets são aninhados nas dimensões:

``` python
(2, 5, 10, 20, 50, 100, 200, 500)
```

A configuração observada usa 2 000 amostras, contaminação de 0,05, duas
dimensões informativas, dados normais gerados a partir de
`N([0, 0], I)`, anomalias deslocadas para `N([4, 4], I)` nas dimensões
informativas e dimensões adicionais irrelevantes com distribuição
independente `N(0, 1)` para ambas as classes. Confirmar estes valores no
código se a geração dos dados for alterada.

------------------------------------------------------------------------

## 6. Testes já existentes

Comando usado para listar os testes:

``` bash
ls tests
```

Resultado observado:

``` text
__pycache__/
test_aderh_integration.py
test_analysis.py
test_data_generation.py
test_experiments.py
test_metrics.py
```

Comando usado para procurar testes relacionados com ADERH, experiências,
diagnósticos e pares:

``` bash
grep -RniE "ADERH|run_experiment|score_diagnostics|pair" tests
```

Referências encontradas:

``` text
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

A pesquisa também encontrou correspondências nos ficheiros compilados
dentro de `tests/__pycache__/`. Estas correspondências binárias não são
testes adicionais e podem ser ignoradas.

### O que isto nos diz

-   Já existe um teste de integração para os defaults do ADERH oficial e
    a direção dos scores.
-   Existem testes para a geração dos dados, análise, métricas e
    diagnósticos experimentais.
-   A pesquisa não revelou, pelo nome, testes dedicados à seleção
    interna dos pares do ADERH ou às variantes Local/Hybrid.

Antes de implementar, devemos ler os testes relevantes completos e
executar a suite existente para estabelecer uma referência.

------------------------------------------------------------------------

## 7. Proposta para Local Pairing e Hybrid Pairing

### Local Pairing

A ideia é selecionar parceiros com base na proximidade aos anchors, em
vez de usar apenas a estratégia aleatória do baseline.

Ainda é necessário definir precisamente:

-   a métrica de distância usada para escolher parceiros;
-   se a própria anchor deve ser excluída da lista de candidatos;
-   se parceiros podem repetir-se;
-   como lidar com empates e com conjuntos pequenos;
-   como controlar a aleatoriedade, quando existir.

### Hybrid Pairing

A ideia é combinar parceiros locais e aleatórios. Uma possibilidade
inicial é usar `alpha` como proporção de pares locais; por exemplo,
`alpha = 0.5` significaria uma combinação equilibrada.

Antes de codificar, devemos documentar a convenção escolhida para
`alpha` e como é feito o arredondamento quando o número total de pares
não é divisível pela proporção.

**Não há ainda evidência de que qualquer variante melhore o
desempenho.** A avaliação experimental deve determinar se melhora, piora
ou mantém as métricas, e em que dimensões.

------------------------------------------------------------------------

## 8. Plano de trabalho proposto

### Fase 1 --- Confirmar a referência atual

1.  Confirmar a branch Git ativa antes de editar ficheiros.
2.  Ler `tests/test_aderh_integration.py` e `tests/test_experiments.py`.
3.  Executar os testes existentes e registar o resultado.

Comandos úteis:

``` bash
git branch --show-current
git status --short
cat tests/test_aderh_integration.py
cat tests/test_experiments.py
python -m pytest
```

Os caminhos para os README assumem que o terminal está em
`baseline_experiment/`. Se o layout real for diferente, ajustar os
caminhos.

### Fase 2 --- Criar uma implementação experimental separada

Criar uma implementação experimental que permita selecionar os parceiros
através de uma estratégia configurável, mantendo o código do baseline e
a dependência oficial intactos.

A implementação deve preservar a lógica de scoring do ADERH. O primeiro
objetivo é provar que o modo de referência reproduz os scores da
implementação oficial sob a mesma configuração e seed. Não basta obter
métricas semelhantes: para um teste de equivalência rigoroso, comparar
os scores ponto a ponto num dataset pequeno e determinístico.

### Fase 3 --- Adicionar testes

Testes mínimos recomendados:

-   modo de referência reproduz os scores oficiais;
-   mesma seed produz resultados reproduzíveis;
-   Local Pairing seleciona parceiros segundo a regra documentada;
-   Hybrid Pairing respeita a proporção definida;
-   não há parceiros próprios, se a especificação os excluir;
-   entradas inválidas são rejeitadas com mensagens claras;
-   `decision_function()` mantém a direção dos scores (maior = mais
    anómalo);
-   pipeline experimental regista as mesmas métricas e metadados
    essenciais.

### Fase 4 --- Integrar no pipeline

Adicionar a seleção da estratégia de pairing sem alterar por defeito o
comportamento do pipeline baseline. Guardar os resultados das variantes
em ficheiros e prefixos separados, para não sobrescrever os resultados
originais.

### Fase 5 --- Comparação experimental

Executar primeiro um `smoke test` para confirmar que o pipeline
funciona. Só depois executar as experiências completas.

A comparação deve usar, tanto quanto possível:

-   as mesmas dimensões;
-   os mesmos dados gerados;
-   os mesmos índices de treino/teste;
-   as mesmas seeds;
-   o mesmo preprocessing;
-   os mesmos hiperparâmetros do ensemble, exceto a estratégia de
    pairing;
-   as mesmas métricas.

Os resultados devem ser analisados por dimensão, incluindo ROC-AUC,
Average Precision, cobertura das hiperesferas e tempos de
treino/scoring.

### Fase 6 --- Documentar resultados

Registar a definição exata das variantes, hiperparâmetros usados,
limitações, resultados por dimensão e diferenças relativamente ao
baseline. As conclusões devem refletir os resultados medidos, sem
assumir previamente que Local ou Hybrid Pairing são melhores.

------------------------------------------------------------------------

## 9. Cuidados antes de começar a editar

-   **Não modificar diretamente o pacote oficial instalado** em
    `site-packages`.
-   **Não substituir os resultados existentes do baseline**; usar
    prefixos ou diretórios próprios.
-   **Não corrigir silenciosamente** a regra original `X[perm]` no modo
    baseline.
-   Não confundir `relative_contrast()` com a seleção de pares do ADERH:
    no pipeline atual, essa função é um diagnóstico geométrico separado.
-   Manter as seeds, splits e preprocessing consistentes na comparação.
-   Documentar qualquer diferença inevitável entre as estratégias.

------------------------------------------------------------------------

## 10. Resumo do estado atual

**Confirmado:**

-   O pacote `aderh==1.0.0` está instalado e a classe `ADERH` foi
    inspecionada.
-   A seleção de pares ocorre dentro de `ADERH.fit()`.
-   O baseline atual usa a regra `X[perm]` para obter os parceiros.
-   O pipeline de experiências está centralizado em
    `src/aderh_distance/experiments.py`.
-   O projeto já tem testes de integração e testes para os componentes
    do pipeline.
-   Existem diretórios dedicados a Local Pairing e Hybrid Pairing.

**Ainda por fazer:**

1.  Confirmar a branch e o estado do Git.
2.  Ler os README das áreas de pairing e os testes existentes.
3.  Executar a suite de testes de referência.
4.  Desenhar a implementação experimental e o teste de equivalência.
5.  Implementar e testar Local Pairing.
6.  Implementar e testar Hybrid Pairing.
7.  Integrar as variantes e executar a comparação experimental.

**Próximo passo imediato:** confirmar a branch com
`git branch --show-current`, consultar os README de pairing e ler os
testes existentes antes de criar ou alterar ficheiros.
