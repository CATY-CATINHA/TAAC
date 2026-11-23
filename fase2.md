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
