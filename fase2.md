# Fase 2 — Recuperar geometria útil e superar a baseline

## Objetivo

Encontrar uma extensão do ADERH que melhore de forma reproduzível o ADERH
original no regime de transição dimensional, sem degradar materialmente os
casos de baixa dimensão e sem escolher hiperparâmetros através do conjunto de
teste.

A evidência atual indica que alterar apenas a identidade dos parceiros não é
suficiente. O Local Pairing aproxima demasiado os pares, reduz os raios e
destrói a cobertura. A Fase 2 separa, por isso, três decisões que estavam
acopladas:

1. que parceiro é escolhido;
2. que escala é usada para o raio;
3. como um ponto não coberto contribui para o score.

## Estado de referência

- A baseline do ADERH original está concluída e validada em oito
  dimensionalidades e dez seeds.
- A análise MLE-LID está concluída; LID acompanha de forma estável o regime em
  que contraste, cobertura e desempenho se degradam.
- Local, Hybrid e LID-Adaptive Pairing estão implementados.
- Os resultados completos das variantes são históricos: foram produzidos antes
  da correção das distâncias dos parceiros e do alinhamento entre centros e
  raios.
- Existe um controlo `corrected_random` que separa o efeito dessas correções do
  efeito da política de pairing.
- O smoke test corrigido é apenas diagnóstico: contém duas dimensionalidades,
  uma seed e foi produzido com o working tree ainda não consolidado.

## Hipótese de trabalho

O insucesso atual resulta principalmente da perda de cobertura causada por
raios demasiado pequenos e pela remoção da diversidade multiescala do ADERH.
Uma estratégia promissora deverá preservar alguma preferência local sem
minimizar agressivamente todas as distâncias e deverá calibrar a escala dos
raios usando apenas os dados de treino.

## Opções a investigar

### A. Calibração não supervisionada dos raios — prioridade alta

Aplicar uma escala aos raios construídos a partir dos pares:

```text
r_j² = gamma * distance(anchor_j, partner_j)² / 2
```

`gamma` não pode ser escolhido através de ROC-AUC ou Average Precision no
teste. Alternativas admissíveis:

- atingir uma cobertura-alvo no treino;
- igualar a cobertura do `corrected_random`;
- igualar quantis da distribuição de raios do `corrected_random`;
- distribuir várias escalas entre os estimadores.

### B. Raios multiescala — prioridade alta

Usar diferentes escalas ao longo do ensemble para recuperar a diversidade
multiescala do ADERH. Esta opção deve ser comparada com uma única escala
calibrada, mantendo o mesmo orçamento de estimadores.

### C. Localidade estocástica — prioridade alta

Substituir a atribuição que minimiza as distâncias por uma das seguintes
políticas:

- parceiro aleatório dentro dos `k` vizinhos permitidos;
- parceiro numa banda intermédia de distâncias;
- distribuição suave controlada por uma temperatura.

O objetivo é evitar pares sistematicamente demasiado próximos.

### D. Pairing com temperatura controlada por LID — prioridade média

Definir uma distribuição do tipo:

```text
P(partner | anchor) proporcional a exp(-distance² / tau(anchor))
```

e usar LID para controlar `tau`. Uma temperatura baixa favorece parceiros
locais; uma temperatura alta aproxima a política do pairing aleatório.

Esta opção só deve ser introduzida depois de existir uma política não
adaptativa capaz de competir com a baseline. Assim será possível medir o valor
acrescentado da LID.

### E. Quantidade global de pairing local — prioridade média

Separar a confiança global na geometria local da ordenação local por LID:

```text
alpha(x) = alpha_global * g(rank(LID(x)))
```

`alpha_global` poderá diminuir quando a LID média aumenta, o Relative Contrast
diminui ou a fidelidade dos vizinhos se deteriora.

### F. ADERH em subespaços — prioridade média/alta para 200D–500D

Construir cada membro do ensemble num subconjunto de features. Os subespaços
podem ser aleatórios, de alto contraste ou ter dimensão controlada por LID.
Esta opção altera o espaço no qual as distâncias são calculadas e é a candidata
mais plausível caso nenhuma política de pairing no espaço completo recupere os
regimes extremos.

### G. Score contínuo fora das hiperesferas — segunda linha

Substituir o empate `score == 1` dos pontos não cobertos por um fallback
contínuo baseado na distância normalizada à hiperesfera mais próxima. É uma
alteração mais profunda ao ADERH e só será considerada se a calibração dos
raios não recuperar cobertura suficiente.

### H. Representação antes do ADERH — alternativa de maior âmbito

Avaliar PCA, random projections ou uma representação aprendida. Esta direção
pode melhorar dimensões extremas, mas aumenta bastante o âmbito e enfraquece a
novidade se for apenas pré-processamento seguido de ADERH.

### I. Ensemble de segurança com o ADERH original

Combinar os scores original e adaptativo usando um peso dependente de cobertura
ou fiabilidade geométrica. Pode melhorar robustez, mas não é a contribuição
principal pretendida.

## Plano experimental

### Fase 2.0 — Diagnóstico geométrico

Antes de criar novas variantes, medir:

- fidelidade dos vizinhos do espaço completo relativamente às duas dimensões
  informativas;
- distribuição das distâncias dos pares;
- distribuição dos raios;
- correlação entre raio e excentricidade do centro;
- cobertura média por estimador;
- número de estimadores que cobrem cada ponto;
- fração sem cobertura.

Este diagnóstico deve usar apenas treino para qualquer decisão de calibração.
As dimensões informativas conhecidas são usadas apenas como diagnóstico do
dataset sintético, nunca pelo modelo.

### Fase 2.1 — Pairing e escala sem LID

Configuração de desenvolvimento:

```text
Dimensões: 2, 20, 50, 100
Seeds: 0, 1, 2, 3, 4
```

Comparar inicialmente:

1. ADERH original;
2. Corrected Random;
3. Local atual;
4. parceiro aleatório dentro de top-k;
5. parceiro numa banda intermédia;
6. pairing com temperatura fixa.

Para as políticas promissoras, comparar:

- raio bruto;
- raio calibrado no treino;
- ensemble multiescala.

Pergunta desta etapa:

> Existe uma combinação de parceiro e escala que supera o ADERH original sem
> usar LID?

### Fase 2.2 — Valor acrescentado da LID

Aplicar LID apenas às melhores políticas da Fase 2.1:

- LID controla a temperatura;
- LID controla a proporção local;
- opcionalmente, LID controla a escala;
- testar `high_local` e `high_random`.

Pergunta desta etapa:

> A adaptação por LID melhora uma política que já é competitiva sem LID?

### Fase 2.3 — Avaliação final

- oito dimensionalidades;
- dez seeds;
- comparação emparelhada por seed;
- ROC-AUC e Average Precision;
- cobertura, diversidade dos scores e geometria dos raios;
- runtime;
- datasets reais;
- intervalos de confiança e/ou teste emparelhado apropriado.

## Critério de sucesso

Uma variante só será considerada uma melhoria se:

1. melhorar ROC-AUC e/ou Average Precision em 50D e 100D;
2. a diferença for consistente entre seeds e superior à variabilidade normal;
3. não degradar materialmente 2D e 20D;
4. não usar labels de teste para escolher parâmetros;
5. preservar ou melhorar cobertura e diversidade dos scores;
6. superar tanto `original` como `corrected_random`;
7. apresentar um mecanismo verificável para a melhoria;
8. manter custo computacional justificável.

Os resultados em 200D e 500D serão tratados separadamente: se a fidelidade dos
vizinhos no espaço completo já tiver desaparecido, uma melhoria nesses regimes
exigirá provavelmente subespaços, transformação da representação ou alteração
do scoring.

## Primeira implementação

A primeira alteração de código desta fase é instrumentar o pipeline com os
diagnósticos da Fase 2.0. Só depois serão implementados pairing estocástico e
calibração de raios. Esta ordem evita selecionar uma nova variante sem saber
qual mecanismo está efetivamente a mudar.

### Primeira leitura do smoke diagnóstico

Foi executada a seed 0 em 2D e 50D apenas para validar a instrumentação. Estes
valores são evidência preliminar, não resultados finais:

| Diagnóstico | 2D | 50D |
|---|---:|---:|
| Sobreposição média dos 5 vizinhos com a geometria informativa | 1.0000 | 0.0069 |
| Cobertura média por estimador — original | 0.9707 | 0.2536 |
| Cobertura média por estimador — corrected random | 0.9721 | 0.1027 |
| Cobertura média por estimador — local | 0.6912 | 0.0145 |

Em 50D, a fidelidade dos vizinhos relativamente às duas dimensões informativas
é já quase nula. Isto põe em causa qualquer método que trate os vizinhos do
espaço completo como uma fonte fiável de estrutura local.

No mesmo smoke, o raio mediano do Local em 50D é aproximadamente `0.9225`,
contra `1.0395` no Corrected Random. Uma redução moderada na escala produz uma
redução muito maior na cobertura, como é esperado em espaços de dimensão
elevada.

A correlação entre raio e excentricidade do centro também muda de cerca de
`0.031` no original para `0.507` no Corrected Random e `0.561` no Local. Isto
reforça a necessidade de estudar separadamente a seleção dos centros, a escala
dos raios e a sua associação.

O pipeline disponibiliza agora um modo limitado:

```bash
python -m pairing.experiments.experiments phase1
```

Este modo executa as dimensões `2, 20, 50, 100`, seeds `0–4`, e apenas os
controlos não adaptativos enquanto as novas políticas ainda estão em
desenvolvimento.

### Diagnóstico limitado em cinco seeds

O modo `phase1` foi executado em diretórios temporários para confirmar a
hipótese antes de criar novas variantes. Os valores seguintes são médias das
seeds `0–4` e continuam a ser diagnóstico de desenvolvimento, não a avaliação
final.

#### Fidelidade dos cinco vizinhos

| Dimensionalidade | Sobreposição com os vizinhos da geometria informativa |
|---:|---:|
| 2 | 1.0000 |
| 20 | 0.0101 |
| 50 | 0.0079 |
| 100 | 0.0058 |

A vizinhança euclidiana do espaço completo deixa de representar a geometria
informativa já em 20D. Consequentemente, Local Pairing baseado diretamente em
kNN no espaço completo não tem uma fonte de localidade fiável neste dataset.

#### Average Precision

| Modelo | 2D | 20D | 50D | 100D |
|---|---:|---:|---:|---:|
| Original | 0.9885 | 0.6239 | 0.3319 | 0.0885 |
| Corrected Random | 0.7650 | 0.5029 | 0.1960 | 0.0601 |
| Hybrid | 0.7584 | 0.4596 | 0.1464 | 0.0574 |
| Local | 0.2377 | 0.1292 | 0.0595 | 0.0501 |

#### Cobertura média de treino por estimador

| Modelo | 2D | 20D | 50D | 100D |
|---|---:|---:|---:|---:|
| Original | 0.9743 | 0.7758 | 0.2226 | 0.0230 |
| Corrected Random | 0.9757 | 0.6471 | 0.1134 | 0.0150 |
| Hybrid | 0.9683 | 0.5284 | 0.0835 | 0.0139 |
| Local | 0.7354 | 0.0916 | 0.0148 | 0.0128 |

O raio mediano do Original e do Corrected Random é muito semelhante, mas o
Original mantém bastante mais cobertura em 20D e 50D. A diferença mais clara é
a associação entre raio e posição do centro: a correlação entre raio e
excentricidade do centro permanece aproximadamente `0.03` no Original e ronda
`0.50–0.59` nas representações corrigidas.

Isto sugere que a associação peculiar da implementação oficial distribui
raios grandes e pequenos por centros de forma quase independente da posição,
o que acidentalmente melhora a cobertura. A Fase 2 não deve restaurar o erro de
implementação, mas pode testar explicitamente uma versão justificável desta
ideia: centros definidos pelos pares, com escalas calibradas ou distribuídas
independentemente ao longo do ensemble.

#### Decisão resultante

A primeira nova variante deverá combinar:

1. pairing aleatório ou apenas suavemente local;
2. raios calibrados com dados de treino;
3. distribuição multiescala ou reassociação explícita de escalas;
4. nenhuma dependência de kNN do espaço completo enquanto a fidelidade dos
   vizinhos for próxima do acaso.

Subespaços passam a ter prioridade elevada para os regimes em que se pretende
recuperar localidade genuína, porque a informação dos dois eixos relevantes
está diluída pelas features de ruído no espaço completo.

## Primeiro controlo de raio implementado

Foram implementadas três variantes sem LID e sem acesso a labels:

1. `shuffled_radius`: mantém centros e distribuição de raios do Corrected
   Random, mas reassocia aleatoriamente os raios entre pares;
2. `coverage_calibrated_random`: mantém o alinhamento correto e aplica o menor
   multiplicador de raio quadrático `gamma >= 1` que atinge a cobertura média
   de treino do ADERH original;
3. `shuffled_coverage_calibrated`: combina reassociação e calibração.

A calibração usa amostras seeded de pontos e estimadores de treino. Não usa
labels, scores de teste, ROC-AUC ou Average Precision. `gamma` nunca é inferior
a um, para que uma geometria corrigida que já cumpra o alvo não seja
encolhida.

### Resultado limitado — cinco seeds

Average Precision média:

| Modelo | 2D | 20D | 50D | 100D |
|---|---:|---:|---:|---:|
| Original | 0.9885 | 0.6239 | 0.3319 | 0.0885 |
| Corrected Random | 0.7650 | 0.5029 | 0.1960 | 0.0601 |
| Shuffled Radius | 0.9738 | 0.6056 | 0.2629 | 0.0822 |
| Coverage Calibrated | 0.8169 | 0.5438 | 0.2589 | 0.0808 |
| Shuffled + Calibrated | 0.9882 | 0.6189 | 0.2763 | 0.0847 |

Diferença média de AP da melhor combinação relativamente ao Original:

| Dimensionalidade | Delta AP |
|---:|---:|
| 2D | -0.0003 |
| 20D | -0.0050 |
| 50D | -0.0556 |
| 100D | -0.0038 |

A reassociação dos raios recupera grande parte do desempenho perdido pelo
Corrected Random, confirmando que a associação centro–raio é um mecanismo
importante. Igualar a cobertura média de treino também recupera desempenho,
mas não é suficiente para superar a baseline. Nenhuma das três variantes vence
o Original de forma consistente entre seeds.

### Decisão após o controlo

- Manter `shuffled_radius` e `shuffled_coverage_calibrated` como ablações.
- Não continuar a procurar `gamma` através dos resultados de teste.
- Uma distribuição multiescala por estimador continua a ser uma experiência
  admissível, desde que seja definida através de cobertura de treino.
- A principal direção para obter melhoria passa a ser alterar o espaço onde a
  geometria é calculada: subespaços de features ou outra representação que
  recupere vizinhos informativos.
- Só depois de existir geometria local fiável deverá a LID controlar localidade
  ou temperatura de pairing.

## Segundo passo — ADERH em subespaços aleatórios

Foi implementado `RandomSubspaceADERH`. Cada um dos 256 estimadores:

1. escolhe anchors e parceiros através do Corrected Random;
2. amostra, sem reposição, no máximo dez features;
3. calcula centros, distâncias, raios, cobertura e scores apenas nesse
   subespaço;
4. na variante `random_subspace_10_shuffled`, mantém a distribuição de raios
   do estimador, mas reassocia-os aleatoriamente entre os pares.

Quando o dataset tem dez ou menos features, são usadas todas. Acima de 10D, a
dimensão efetiva de cada estimador permanece fixa em dez. O modelo não conhece
quais são as duas features informativas do gerador sintético e não usa labels
ou métricas de teste.

O valor `max_features=10` e a variante com reassociação de raios foram
escolhidos exclusivamente nas seeds de desenvolvimento `0–4`. Depois dessa
escolha, a configuração foi congelada e avaliada nas seeds `5–9`.

### Resultado de desenvolvimento — seeds 0–4

Na comparação de desenvolvimento, `random_subspace_10_shuffled` obteve os
seguintes deltas médios relativamente ao Original:

| Dimensionalidade | Delta ROC-AUC | Delta AP | Vitórias AP |
|---:|---:|---:|---:|
| 2D | -0.0005 | -0.0116 | 1/5 |
| 20D | +0.0005 | +0.0022 | 3/5 |
| 50D | +0.0151 | +0.0552 | 4/5 |
| 100D | +0.0892 | +0.1465 | 5/5 |

Este resultado serviu para selecionar a configuração, não para a validar.

### Validação separada — seeds 5–9

Após congelar a configuração, foi executado o modo:

```bash
python -m pairing.experiments.experiments subspace-validation
```

| Dimensão | ROC Original | ROC Subspace | Delta ROC | AP Original | AP Subspace | Delta AP |
|---:|---:|---:|---:|---:|---:|---:|
| 2D | 0.9991 | 0.9990 | -0.0001 | 0.9782 | 0.9761 | -0.0021 |
| 20D | 0.9463 | 0.9462 | -0.0002 | 0.5883 | 0.5688 | -0.0195 |
| 50D | 0.8702 | 0.8636 | -0.0066 | 0.2727 | 0.3074 | +0.0346 |
| 100D | 0.6828 | 0.8056 | +0.1227 | 0.0795 | 0.1957 | +0.1162 |

Em 100D, a melhoria ocorreu nas cinco seeds tanto em ROC-AUC como em AP. Em
50D, a AP melhorou em quatro das cinco seeds, embora a ROC-AUC média tenha
descido ligeiramente. Em 20D existe uma perda de AP que terá de ser considerada
num mecanismo adaptativo ou num ensemble de segurança. Cinco seeds dão pouca
potência a um teste estatístico isolado; a consistência e a magnitude em 100D
são a evidência principal desta validação.

### Comparação focada nas oito dimensionalidades

O candidato congelado foi depois comparado com o Original nas oito dimensões e
nas dez seeds através de `subspace-full`. Esta tabela combina desenvolvimento e
validação para caracterizar a curva completa; por isso, não deve ser descrita
como uma avaliação totalmente held-out.

O snapshot validado desta execução está disponível nos ficheiros
[`raw`](pairing/experiments/results/current/subspace-full_raw_results.csv),
[`summary`](pairing/experiments/results/current/subspace-full_summary.csv) e
[`metadata`](pairing/experiments/results/current/subspace-full_metadata.json).
Foi produzido a partir do commit `e917c79`, com o working tree limpo.

| Dimensão | ROC Original | ROC Subspace | Delta ROC | AP Original | AP Subspace | Delta AP |
|---:|---:|---:|---:|---:|---:|---:|
| 2D | 0.9993 | 0.9990 | -0.0003 | 0.9833 | 0.9765 | -0.0068 |
| 5D | 0.9945 | 0.9947 | +0.0002 | 0.8958 | 0.8949 | -0.0010 |
| 10D | 0.9839 | 0.9832 | -0.0007 | 0.7978 | 0.7912 | -0.0066 |
| 20D | 0.9506 | 0.9508 | +0.0002 | 0.6061 | 0.5975 | -0.0086 |
| 50D | 0.8780 | 0.8822 | +0.0042 | 0.3023 | 0.3472 | +0.0449 |
| 100D | 0.6990 | 0.8050 | +0.1060 | 0.0840 | 0.2153 | +0.1313 |
| 200D | 0.5018 | 0.6902 | +0.1884 | 0.0502 | 0.1277 | +0.0776 |
| 500D | 0.5000 | 0.6097 | +0.1097 | 0.0500 | 0.0990 | +0.0490 |

Em 100D e 200D, o candidato venceu nas dez seeds em ambas as métricas. Em
500D, venceu nas dez seeds em AP e em nove das dez em ROC-AUC. Testes de
Wilcoxon emparelhados nas dez seeds dão `p=0.0020` para ambas as métricas em
100D e 200D; em 500D dão `p=0.0039` para ROC-AUC e `p=0.0020` para AP. Estes
valores são exploratórios, pois cinco das dez seeds participaram na seleção do
candidato. Em 50D, a melhoria média de AP não é ainda estatisticamente estável
(`p=0.3223`).

### Mecanismo observado

O ganho não resulta de melhorar os vizinhos euclidianos no espaço completo.
Resulta de deixar de calcular cada distância em todas as features de ruído:

- cada estimador vê apenas dez features, limitando a concentração de
  distâncias;
- diferentes estimadores amostram vistas diferentes, pelo que parte do
  ensemble contém uma ou ambas as features informativas;
- a reassociação dos raios reduz o acoplamento entre raio e excentricidade do
  centro;
- a cobertura exata do teste passa a ser total nas oito dimensionalidades,
  enquanto o Original deixa 47.25% dos pontos sem cobertura em 100D, 99.67%
  em 200D e 100% em 500D;
- a cobertura média amostrada de treino por estimador do candidato mantém-se
  perto de 0.91 em alta dimensão.

Os diagnósticos confirmam que o ensemble não recebe ajuda para encontrar as
features informativas:

| Dimensão ambiente | Features por estimador | Estimadores com pelo menos uma feature informativa | Estimadores com ambas |
|---:|---:|---:|---:|
| 20D | 10 | 76.91% | 24.30% |
| 50D | 10 | 36.99% | 3.48% |
| 100D | 10 | 19.65% | 0.98% |
| 200D | 10 | 9.92% | 0.16% |
| 500D | 10 | 3.91% | 0.04% |

Apesar de a maioria dos estimadores em alta dimensão não conter uma feature
informativa, os 256 subespaços dão ao ensemble vistas suficientes para evitar
o colapso completo. Esta observação também indica o limite da estratégia: à
medida que a dimensão aumenta, a probabilidade de observar sinal útil continua
a diminuir.

Este mecanismo é verificável, mas tem uma limitação importante: subespaços
aleatórios são uma técnica conhecida. O resultado demonstra uma solução
eficaz para o failure mode do ADERH, não estabelece por si só a novidade da
contribuição. Também falta validação em datasets reais, onde não se conhecem as
features informativas.

### Avaliação face aos critérios de sucesso

1. **50D e 100D:** AP melhora em ambos; em 100D melhoram claramente as duas
   métricas. Em 50D, a evidência é positiva para AP, mas ainda instável.
2. **Consistência:** confirmada em 100D e nos regimes 200D–500D; parcial em
   50D.
3. **Baixa dimensão:** as perdas entre 2D e 20D são pequenas em média, mas a
   perda de AP em 20D nas seeds held-out merece mitigação.
4. **Sem leakage no ajuste:** o detector e os raios são ajustados sem labels.
   A escolha entre os candidatos usou as métricas das seeds de desenvolvimento
   `0–4`, mas não as seeds held-out `5–9`. Uma política final estritamente não
   supervisionada ainda deverá fixar a configuração a priori ou escolhê-la por
   um diagnóstico calculado apenas no treino.
5. **Cobertura e scores:** cumprido; deixa de haver o colapso para scores todos
   iguais a um.
6. **Baselines:** supera claramente o Original em alta dimensão e, por
   consequência, também o Corrected Random nesses regimes.
7. **Mecanismo:** suportado pelos diagnósticos de dimensão efetiva, cobertura e
   associação centro–raio.
8. **Custo:** comparável e, em 500D, inferior ao Original porque cada estimador
   calcula distâncias em apenas dez features.

## Próximo passo recomendado

O próximo desenvolvimento deve partir deste candidato funcional, em vez de
voltar ao kNN no espaço completo. A hipótese seguinte é usar LID para controlar
a dimensão ou a combinação dos subespaços:

1. manter `random_subspace_10_shuffled` como baseline positiva congelada;
2. estimar um sinal global de dimensionalidade apenas no treino;
3. comparar uma pequena política pré-definida de dimensões de subespaço, sem
   escolher valores por AP/ROC no teste;
4. avaliar se a LID acrescenta valor relativamente ao subespaço fixo;
5. testar um fallback para o Original nas dimensões em que a geometria global
   ainda é fiável, reduzindo a pequena perda em 2D–20D;
6. só promover a proposta final depois de validar também em dados reais.

Esta sequência preserva a pergunta científica inicial: a LID não precisa de
ser forçada a escolher parceiros locais; pode indicar quanta dimensionalidade
cada membro do ensemble deve observar.
