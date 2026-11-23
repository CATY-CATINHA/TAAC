# CP2 · Primeira seleção de subespaços por LID

O CP0 (`a15791d`) guarda os dois modelos de referência e os resultados
anteriores. O CP1 (`c7d2fd6`) fixa o modelo, o limiar e o protocolo antes
da avaliação. O CP2 guarda as medições obtidas a partir do CP1.
[Protocolo e comandos](../../lid_gate_protocol.md) · [Referências congeladas](../pre_subspace/README.md).

## Dados reais

Médias com peso igual para cada um dos onze datasets. Cada média por
dataset agrega os três splits e cinco seeds do CP0. O teste emparelhado
automático confirmou as 330 linhas das duas referências em AP, ROC-AUC e
fração sem cobertura, com tolerância absoluta de 10⁻¹².

| Política | AP | ROC-AUC |
|---|---:|---:|
| ADERH original | 0,5407 | 0,8882 |
| Pre-subspace fixo | 0,5503 | 0,8773 |
| Escolha por número de features | 0,5503 | 0,8773 |
| Escolha por LID (CP1) | 0,5364 | 0,8876 |

A escolha por LID perdeu 0,0139 de AP em média face ao pre-subspace
(IC 95% de −0,0490 a +0,0069) e ganhou 0,0103 de ROC-AUC (IC 95% de
−0,0009 a +0,0265). Em AP houve quatro vitórias, quatro empates e três
derrotas face ao pre-subspace. Os seis contrastes reais tiveram p de Holm
igual a 1 no teste bilateral dos sinais. Este painel já tinha sido usado
para caracterizar os dois modelos no CP0. [Efeitos emparelhados](real_effects.csv).

O ramo escolhido foi o mesmo nos três splits de cada dataset. O original
foi escolhido em 27 dos 33 splits e o pre-subspace nos seis restantes.

| Dataset | Modelo escolhido | Intervalo das medianas de LID nos três treinos |
|---|---|---:|
| Cardio | Original | 4,96–5,15 |
| Lymphography | Original | 4,97–5,21 |
| Musk | Original | 5,72–5,79 |
| Optdigits | Original | 8,82–8,91 |
| Pendigits | Original | 5,31–5,36 |
| Satimage-2 | Pre-subspace | 11,41–11,51 |
| Waveform | Pre-subspace | 15,18–15,25 |
| WBC | Original | Regra direta para 9 features |
| WDBC | Original | 7,70–7,79 |
| WPBC | Original | 7,27–7,76 |
| Wine | Original | 5,10–5,72 |

Em Satimage-2 e Waveform a ativação diminuiu a AP face ao original em
0,0017 e 0,0449. No WDBC, a regra selecionou o original (AP 0,4402) e
perdeu o ganho do pre-subspace (AP 0,6203). Optdigits ilustra a situação
inversa: a escolha do original recuperou cerca de 0,0825 em ROC-AUC face
ao pre-subspace. [Medições por split e seed](real_runs.csv).

## Novas seeds sintéticas

Nos 160 pares seed–dimensão, o CP1 e a regra por número de features
selecionaram o mesmo ramo. Em 2D, 5D e 10D a regra fixa o original por
construção. Em 20D, 50D, 100D, 200D e 500D, a mediana de LID ultrapassou
10 em todas as 20 seeds. A AP média do CP1 foi a seguinte.

| Features | AP CP1 | Δ face ao original | Δ face ao pre-subspace |
|---:|---:|---:|---:|
| 2 | 0,9740 | 0,0000 | −0,0042 |
| 5 | 0,8872 | 0,0000 | +0,0069 |
| 10 | 0,7731 | 0,0000 | +0,0099 |
| 20 | 0,5681 | −0,0177 | 0,0000 |
| 50 | 0,3182 | +0,0599 | 0,0000 |
| 100 | 0,1876 | +0,1030 | 0,0000 |
| 200 | 0,1396 | +0,0891 | 0,0000 |
| 500 | 0,0897 | +0,0397 | 0,0000 |

A 100D o CP1 venceu o original em AP nas 20 seeds. O mesmo ganho já
está disponível com os subespaços fixos. A 20D, a seleção ativou também
os subespaços e manteve a sua perda média. As comparações anteriores ao
CP1 usaram outras seeds sintéticas. Todos os deltas acima foram calculados
nas mesmas seeds 30–49 para as quatro políticas.
[Efeitos, IC e testes](synthetic_effects.csv) · [Medições brutas](synthetic_runs.csv).

## Decisão experimental

Manter o pre-subspace como referência para a próxima alteração. O CP1
preserva o benefício dos subespaços no regime sintético de ruído elevado,
mas este limiar global de LID selecionou mal os dois casos reais em que
ativou o candidato. Estes resultados avaliam especificamente o limiar 10
e a decisão global por treino. Uma política revista deverá ter outro
identificador e validação independente.
