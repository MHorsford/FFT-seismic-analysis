# FFT aplicada à análise sísmica

Projeto da disciplina **Análise de Algoritmos** (UFPA) sobre a técnica de
**Divisão e Conquista**. A implementação usa o algoritmo FFT Radix-2 de
Cooley–Tukey para transformar sinais sísmicos do domínio do tempo para o
domínio da frequência e compara seu custo com uma DFT direta.

O trabalho cobre os cinco itens do enunciado:

1. **Problema:** identificar as frequências presentes em registros de um
   terremoto.
2. **Técnica:** a FFT divide o vetor entre índices pares e ímpares, resolve os
   dois subproblemas e combina os resultados com operações “borboleta”.
3. **Implementação:** DFT direta, FFT própria, aquisição FDSN e geração de
   gráficos estão separadas em módulos.
4. **Complexidade:** DFT em `O(N²)` contra FFT em `O(N log N)`.
5. **Resultados:** benchmark sintético e estudo de caso com formas de onda
   reais, incluindo conferência numérica DFT × FFT × NumPy.

## A distinção mais importante: tempo × frequência

Os dois domínios mostram **os mesmos dados**, mas respondem perguntas
diferentes:

| Representação | Eixo horizontal | O que permite observar |
|---|---|---|
| **Domínio do tempo** | segundos | quando a onda chegou e como a amplitude variou |
| **Domínio da frequência** | hertz (ciclos por segundo) | quais frequências compõem o registro e com que amplitude |

A FFT recebe `N` amostras ordenadas no tempo e calcula `N` coeficientes de
frequência. Ela não produz um novo terremoto e não altera o registro original;
ela reorganiza matematicamente a informação para tornar visíveis as
periodicidades do sinal.

O arquivo `3_tempo_x_frequencia.png`, gerado no estudo real, coloca as duas
representações lado a lado e deve ser o primeiro gráfico usado para explicar
essa passagem.

## Organização do código

- `main.py`: coordena o benchmark, o estudo sísmico e a interface de linha de
  comando.
- `src/dft.py`: definição direta da DFT; referência correta, porém `O(N²)`.
- `src/fft.py`: FFT própria por Divisão e Conquista, `O(N log N)`.
- `src/data.py`: configuração do terremoto, busca de estações e download FDSN.
- `src/benchmark.py`: metodologia das medições e conferência de corretude.
- `src/visualization.py`: gráficos didáticos, sísmicos e de desempenho.
- `testes/correctness_test.py`: testes locais que não dependem da internet.
- `evento_exemplo.json`: modelo editável para analisar outro terremoto.

As docstrings explicam as decisões conceituais; comentários dentro das funções
ficam reservados às partes em que o motivo não é óbvio pelo próprio código.

## Preparação

Na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Os gráficos do benchmark são locais. O estudo sísmico precisa de internet para
consultar serviços FDSN; a indisponibilidade de uma estação não interrompe as
demais.

## Como executar

Executar as duas fases com o evento apresentado:

```powershell
python main.py
```

Executar somente o benchmark de complexidade, sem rede:

```powershell
python main.py --fase benchmark
```

Executar somente o estudo sísmico:

```powershell
python main.py --fase sismo
```

Ver todas as opções:

```powershell
python main.py --help
```

Observação: o maior ponto padrão do benchmark é `N = 16384`. Como a DFT é
quadrática, esse ponto domina o tempo total; isso é comportamento esperado e
faz parte do resultado da análise.

## Como conferir outro terremoto

1. Faça uma cópia de `evento_exemplo.json`.
2. Altere origem UTC, magnitude, profundidade, latitude e longitude.
3. Ajuste `regiao_busca` para um retângulo que contenha estações próximas ao
   novo epicentro.
4. Execute apenas a fase sísmica e use outra pasta para não sobrescrever o caso
   apresentado:

```powershell
python main.py --fase sismo --evento meu_evento.json --saida-sismo resultados_meu_evento
```

Campos do JSON:

- `nome`: rótulo livre usado nos títulos.
- `origem`: instante de origem em ISO-8601 e UTC, por exemplo
  `2026-06-24T22:04:33Z`.
- `magnitude`: valor informativo usado no resumo.
- `profundidade_km`: profundidade usada pelo TauP para estimar a chegada P.
- `latitude`, `longitude`: epicentro usado no mapa e nas distâncias.
- `distancia_min_km`: estações mais próximas que esse limite são ignoradas;
  use `0` para aceitar qualquer distância no teste de outro evento.
- `regiao_busca`: limites `minlatitude`, `maxlatitude`, `minlongitude` e
  `maxlongitude` enviados ao serviço de estações.

O código valida campos, limites geográficos e profundidade antes de acessar a
rede. A lista estática de contingência é usada somente no evento padrão, pois
reaproveitar estações da Venezuela para outro epicentro criaria um resultado
enganoso.

## Arquivos produzidos

`resultados/` (estudo de complexidade):

- `benchmark_avancado.csv` e `benchmark_avancado.png`;
- `divisao_n8.png` e `borboleta_n8.png`.

`resultados_fft_sismo/` (evento real):

- `1_mapa_estacoes.png`;
- `2_dominio_tempo.png`;
- `3_tempo_x_frequencia.png`;
- `4_dominio_frequencia.png`;
- `5_benchmark_dft_fft.png`;
- `resumo_estacoes.csv`.

## Verificação

Os testes comparam a FFT própria tanto com a DFT do projeto quanto com
`numpy.fft.fft`, incluindo vetores não triviais e zero-padding:

```powershell
python -m unittest discover -s testes -p "*_test.py" -v
```

A igualdade dos resultados mostra que a FFT é uma maneira mais eficiente de
calcular a **mesma DFT**; a diferença entre os algoritmos está no número de
operações, não na transformada obtida.
