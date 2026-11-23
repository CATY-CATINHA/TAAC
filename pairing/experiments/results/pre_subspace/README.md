# Avaliação do pre-subspace

**Pre-subspace** designa o candidato `random_subspace_10_shuffled` congelado
no commit `e917c79`. Cada um dos 256 estimadores usa 18 âncoras e até dez
features aleatórias. O raio quadrático deriva da distância do par no
subespaço e os raios são redistribuídos pelos pares. O comparador é o
`aderh==1.0.0` original. Ambas as configurações usam os mesmos dados,
split e seed do modelo. O ajuste é não supervisionado e o score é calculado
de uma vez para todo o lote de teste, como exige a semântica de densidade do
ADERH para comparações consistentes.

## Dados e comparabilidade

- **Sintético.** Repetimos a experiência das duas features informativas com
  20 novos datasets (seeds 10–29). Cada um tem 2 000 pontos, 5% de anomalias
  e oito versões aninhadas com 2 a 500 features. Dentro de cada seed todas
  as dimensionalidades partilham os mesmos pontos e o mesmo split 70/30.
  O snapshot anterior usa as seeds 0–9 e está em
  [subspace-full_summary.csv](../current/subspace-full_summary.csv).
- **Real.** Incluímos todos os datasets da Tabela 6 presentes na coleção
  ADBench Classical com menos de 10 000 linhas. O limite de custo foi
  escolhido antes de correr os modelos. Os onze datasets completos vão de
  129 a 6 870 observações e de 9 a 166 features. O manifesto fixa a versão
  dos ficheiros, dimensões, número de anomalias e os valores do ADERH nas
  Tabelas 1 (ROC-AUC) e 8 (AUC-PR). [Artigo](https://proceedings.neurips.cc/paper_files/paper/2025/file/f2b6bc2104f6f4481eef101f6a32c085-Paper-Conference.pdf) · [Ficheiros do ADBench](https://github.com/Minqi824/ADBench/tree/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/datasets/Classical) · [Manifesto](../../data/adbench_manifest.csv).
- **Protocolo real.** Usámos três splits estratificados 70/30 com
  `StratifiedShuffleSplit(random_state=0)` e cinco seeds de modelo por split.
  Seguimos as seeds do artigo (0, 1, 2, 100, 1000). O script público dos
  autores inclui 10000 em lugar de 100 e normaliza o dataset inteiro. No
  nosso protocolo o MinMaxScaler aprende os limites exclusivamente no
  treino. A comparação emparelhada usa o original reexecutado sob este
  mesmo protocolo. As colunas `paper` e `original_minus_paper` permitem
  verificar a distância face à referência publicada, arredondada a três
  casas. [Script dos autores](https://github.com/Walid10010/ADERH/blob/main/experiments/run_experiment.py).

## Resultados sintéticos nas seeds 10–29

Diferenças médias absolutas do pre-subspace face ao original. AP ganhou em
19/20 seeds a 100D e em 20/20 a 200D e 500D. Nestes três regimes ambas as
métricas mantêm significância após Holm. A 50D a AP aumentou em 16/20
seeds. O teste dos sinais corrigido dá p = 0,106. A 5D houve uma perda de AP
em 17/20 seeds (p corrigido = 0,026).

| Features | AP original | AP pre-subspace | Δ AP | IC 95% de Δ AP | Δ ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 2 | 0,9881 | 0,9801 | −0,0080 | [−0,0172, +0,0009] | −0,0002 |
| 5 | 0,8711 | 0,8538 | −0,0174 | [−0,0249, −0,0104] | −0,0004 |
| 10 | 0,7637 | 0,7631 | −0,0006 | [−0,0114, +0,0105] | +0,0009 |
| 20 | 0,5674 | 0,5446 | −0,0228 | [−0,0423, −0,0042] | −0,0023 |
| 50 | 0,2473 | 0,3019 | +0,0546 | [+0,0160, +0,1027] | +0,0093 |
| 100 | 0,0827 | 0,1828 | +0,1001 | [+0,0725, +0,1264] | +0,0837 |
| 200 | 0,0503 | 0,1216 | +0,0712 | [+0,0551, +0,0894] | +0,1963 |
| 500 | 0,0500 | 0,0902 | +0,0402 | [+0,0309, +0,0507] | +0,1211 |

A 100D, 200D e 500D a fração de pontos de teste sem cobertura do original
foi 42,0%, 99,2% e 100%, respetivamente. O pre-subspace cobriu todos nesses
regimes. Uma verificação adicional na seed histórica 0 reproduziu ambas as
métricas do snapshot anterior em 2D, 50D, 100D, 200D e 500D, com tolerância
absoluta de 10⁻¹². [Estatísticas completas](synthetic_effects.csv).

## Resultados nos 11 datasets reais

| Métrica | ADERH original | Pre-subspace | Δ médio | IC 95% do Δ | Vitórias / empates / derrotas | p de Holm |
|---|---:|---:|---:|---:|---:|---:|
| Average Precision | 0,5407 | 0,5503 | +0,0096 | [−0,0147, +0,0470] | 3 / 2 / 6 | 0,508 |
| ROC-AUC | 0,8882 | 0,8773 | −0,0109 | [−0,0270, +0,0004] | 2 / 2 / 7 | 0,359 |

O ganho médio de AP concentra-se no WDBC (+0,1801). Nos restantes dez
datasets o delta médio é −0,0074. No Waveform a AP desce 0,0449 e no
Optdigits a ROC-AUC desce 0,0825. No Musk, o único dataset deste painel com
mais de 100 features (166), ambos os modelos já atingem AP e ROC-AUC de
1,000. O sinal e a magnitude do efeito variam entre datasets reais. Acima
de 166 features, a evidência desta avaliação provém apenas do sintético.
[Comparação por dataset e amplitude por split](real_by_dataset.csv) · [Efeitos globais](real_effects.csv).

### Verificação posterior da referência WDBC

O WDBC tem apenas três anomalias por partição de teste. Com o scaler
ajustado no treino, a AP média do original foi 0,4402 e a do pre-subspace
0,6203. Investigámos a diferença entre o original e a AP publicada de 0,614
variando separadamente a normalização e a lista de seeds apenas no original.

| Ajuste do scaler | AP com seeds do artigo | AP com seeds do script |
|---|---:|---:|
| Treino | 0,4402 | 0,4390 |
| Dataset inteiro, duas transformações como no script | 0,6258 | 0,6141 |

O protocolo do script reproduz os dois valores publicados à precisão da
tabela (AP 0,614 e ROC-AUC 0,981). Neste dataset, a normalização explica a
maior parte da discrepância. O valor de +0,1801 quantifica a vantagem sob o
nosso protocolo de treino. Os dados deste diagnóstico posterior estão num
ficheiro separado e podem ser regenerados com
`python -m pairing.experiments.pre_subspace wdbc-reference`.
[Execuções WDBC](wdbc_reference.csv).

## Como ler a estatística

As duas métricas são Average Precision (AP, `average_precision_score`) e
ROC-AUC. Um delta positivo favorece o pre-subspace. A prevalência é a
referência de AP para interpretar o desequilíbrio de classes e pode ser
consultada por split (`test_anomalies / n_test`).

- No sintético calculamos o delta emparelhado dentro de cada seed e
  dimensão. Cada teste tem 20 unidades. A correção de Holm abrange os 16
  testes (oito dimensões × duas métricas), inclusive quando as dimensões
  partilham observações.
- Nos dados reais calculamos primeiro a média das 15 execuções de cada
  modelo por dataset. Damos depois peso igual aos onze datasets. Assim,
  repetir o treino ou reutilizar pontos em vários splits não aumenta
  artificialmente o número de unidades independentes. A família de Holm
  real contém os dois testes globais, um por métrica.
- Para a **magnitude** reportamos delta médio, mediana e IC percentil de
  95% da média, com 20 000 reamostragens emparelhadas de seeds ou datasets
  e seed de bootstrap 20261123. São intervalos marginais. Nos dados reais
  traduzem a sensibilidade à composição deste painel escolhido por tamanho.
- Para a **consistência da direção** usamos o teste exato bilateral dos
  sinais. A hipótese nula atribui a mesma probabilidade a vitória e derrota
  entre unidades sem empate. Diferenças que arredondam a zero em 12 casas
  decimais contam como empate. São reportados os valores de p brutos e
  corrigidos por Holm (limiar 0,05). Este teste e o IC da média respondem a
  perguntas distintas quando os efeitos têm magnitudes muito diferentes.

## Reprodução

Os comandos de descarga, execução e análise estão no [guia das experiências](../../README.md#pre-subspace-retest).
Os CSV `*_runs` permitem refazer a estatística sem voltar a treinar. Os
metadados guardam os parâmetros e as versões dos pacotes usadas na execução.
