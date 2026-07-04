"""
Benchmarks de desempenho: DFT força bruta vs FFT (Divisão e Conquista) vs
numpy.fft.fft, em função do tamanho do sinal N.

Existem duas funções de benchmark, com propósitos diferentes:

- `benchmark_comparacao`: uma execução por N, sem repetição. Rápida, boa
  para uma checagem informal enquanto se está desenvolvendo.
- `benchmark_avancado`: a versão rigorosa, com repetições, desvio padrão,
  contagem teórica de operações e conferência de corretude cruzada. É essa
  que deve alimentar a análise de algoritmo "de verdade" (a que vai pra
  apresentação) — ver `_medir_tempo` para os detalhes de metodologia.
"""

import gc
import time
import numpy as np
from scipy.optimize import curve_fit

from src.dft import dft_forca_bruta
from src.fft import fft_divisao_conquista, preparar_sinal_para_fft


def _medir_tempo(func, *args, num_repeticoes=5):
    """
    Executa `func(*args)` repetidas vezes e retorna
    (resultado_da_ultima_execucao, tempo_medio_em_segundos, desvio_padrao_em_segundos).

    Duas decisões de metodologia, tomadas de propósito (e que valem a pena
    citar na análise/arguição oral):

    1. Sem "warm-up" (chamada de descarte antes de cronometrar). Isso ajuda
       bastante em cenários com JIT ou cache de disco, mas aqui as três
       funções medidas (DFT, FFT própria, numpy.fft.fft) são só numpy
       "puro" — não há nada relevante para esquentar. Em compensação, uma
       chamada extra de descarte custaria caro justamente onde N é maior
       (a DFT é O(N²): em N=2¹⁴, uma chamada de descarte custa ~15s
       sozinha). Não compensa o troca-troca aqui.
    2. Coletor de lixo (`gc`) desativado durante a cronometragem. Uma
       pausa do garbage collector no meio de uma medição pequena (N
       baixo, tempos da ordem de microssegundos) pode, sozinha, dominar o
       tempo medido — um ruído que não tem nada a ver com o algoritmo.
       Reativamos o gc no `finally` mesmo se alguma medição levantar
       exceção no meio do caminho.

    O desvio padrão retornado serve para avaliar se uma diferença entre
    dois tempos médios é "real" ou só ruído de medição — importante para
    não superinterpretar pequenas diferenças, especialmente em N pequeno.
    """
    tempos = []
    gc.disable()
    try:
        for _ in range(num_repeticoes):
            t0 = time.perf_counter()
            resultado = func(*args)
            tempos.append(time.perf_counter() - t0)
    finally:
        gc.enable()

    return resultado, float(np.mean(tempos)), float(np.std(tempos))


def benchmark_comparacao(lista_N):
    """
    Para cada N em lista_N, gera um sinal aleatório, aplica DFT e FFT uma
    única vez e mede o tempo. Retorna uma lista de dicionários.

    Versão "rápida"/informal — sem repetições nem desvio padrão. Para a
    análise de algoritmo de verdade, prefira `benchmark_avancado`.
    """
    resultados = []
    for N in lista_N:
        # Sinal aleatório de comprimento N; a FFT exige potência de 2,
        # então preparamos o sinal (zero-padding) antes de usar em ambos
        # os algoritmos — assim DFT e FFT rodam sobre o mesmíssimo vetor.
        sinal_original = np.random.randn(N)
        sinal, N_pad = preparar_sinal_para_fft(sinal_original)

        t0 = time.perf_counter()
        espec_dft = dft_forca_bruta(sinal)
        t_dft = time.perf_counter() - t0

        t0 = time.perf_counter()
        espec_fft = fft_divisao_conquista(sinal)
        t_fft = time.perf_counter() - t0

        resultados.append({
            "N_original": N,
            "N_pad": N_pad,
            "tempo_dft_s": t_dft,
            "tempo_fft_s": t_fft,
            "speedup": t_dft / t_fft if t_fft > 0 else float("nan"),
            "resultados_batem": bool(np.allclose(espec_dft, espec_fft, atol=1e-6)),
        })

    return resultados


def benchmark_avancado(lista_N, num_repeticoes=5):
    """
    Benchmark rigoroso para a análise de algoritmo. Para cada N em
    lista_N, mede DFT força bruta, FFT própria (Divisão e Conquista) e
    numpy.fft.fft (referência externa, implementação em C), com
    `num_repeticoes` repetições cada (ver `_medir_tempo`).

    Além do tempo, cada resultado traz:

    - `tempo_*_std_s`: desvio padrão das repetições — quão ruidosa foi a
      medição.
    - `ops_dft_teorico` / `ops_fft_teorico`: contagem TEÓRICA de operações
      (N² e N·log2(N), respectivamente) — não é tempo medido, é só o
      "número de passos" de cada algoritmo pela definição. Comparar isso
      com o tempo medido é o cerne da análise: a nota teórica estima
      ~186x de ganho contando operações (N=2048: 4.194.304 vs 22.528),
      mas o SPEEDUP MEDIDO tende a ser bem diferente, porque a DFT usa
      multiplicação de matriz vetorizada (BLAS/C por baixo do numpy)
      enquanto a FFT própria é recursão em Python puro, com overhead de
      chamada de função e fatiamento de array a cada nível de recursão.
      Reportar as duas coisas lado a lado é o que permite discutir essa
      diferença entre complexidade assintótica e desempenho medido na
      prática (ótimo ponto para a arguição oral — ver Plano de Execução,
      Passo 6).
    - `resultados_batem`: DFT própria vs FFT própria.
    - `fft_propria_bate_com_numpy`: FFT própria vs numpy.fft.fft. Testamos
      contra os DOIS (não só contra a própria DFT) porque, se as duas
      implementações do projeto tivessem o mesmo bug, uma bateria de
      testes que só compara DFT-própria-vs-FFT-própria não pegaria isso;
      numpy.fft.fft é uma referência externa e independente.

    Parâmetros
    ----------
    lista_N : list[int]
        Tamanhos de sinal a testar. Não precisam ser potência de 2 (o
        padding é feito aqui dentro); tipicamente potências de 2 de
        2⁴ a 2¹⁴, como sugerido no Plano de Execução, Passo 6.
    num_repeticoes : int
        Repetições por N. Mais repetições = média mais estável, mas
        também mais tempo total — a DFT é O(N²), então o N mais alto da
        lista domina o tempo total do benchmark. Se estiver demorando
        demais, reduza `num_repeticoes` ou o maior N da lista antes de
        reduzir a quantidade de pontos intermediários (eles é que dão a
        forma da curva).

    Retorna
    -------
    list[dict], um por N, pronto para virar tabela/CSV ou entrar em
    `visualization.plot_benchmark_avancado`.
    """
    resultados = []
    for N in lista_N:
        sinal_original = np.random.randn(N)
        sinal, N_pad = preparar_sinal_para_fft(sinal_original)

        espec_dft, t_dft, std_dft = _medir_tempo(
            dft_forca_bruta, sinal, num_repeticoes=num_repeticoes)
        espec_fft, t_fft, std_fft = _medir_tempo(
            fft_divisao_conquista, sinal, num_repeticoes=num_repeticoes)
        espec_numpy, t_numpy, std_numpy = _medir_tempo(
            np.fft.fft, sinal, num_repeticoes=num_repeticoes)

        ops_dft_teorico = float(N_pad ** 2)
        ops_fft_teorico = float(N_pad * np.log2(N_pad)) if N_pad > 1 else 1.0

        resultado = {
            "N_original": N,
            "N_pad": N_pad,
            "tempo_dft_s": t_dft,
            "tempo_dft_std_s": std_dft,
            "tempo_fft_propria_s": t_fft,
            "tempo_fft_propria_std_s": std_fft,
            "tempo_numpy_fft_s": t_numpy,
            "tempo_numpy_fft_std_s": std_numpy,
            "ops_dft_teorico": ops_dft_teorico,
            "ops_fft_teorico": ops_fft_teorico,
            "speedup_dft_vs_fft_propria": t_dft / t_fft if t_fft > 0 else float("nan"),
            "speedup_dft_vs_numpy": t_dft / t_numpy if t_numpy > 0 else float("nan"),
            "resultados_batem": bool(np.allclose(espec_dft, espec_fft, atol=1e-6)),
            "fft_propria_bate_com_numpy": bool(np.allclose(espec_fft, espec_numpy, atol=1e-6)),
        }
        resultados.append(resultado)

        print(f"  N={N_pad:>6}: DFT={t_dft*1e3:9.3f} ms (±{std_dft*1e3:.3f})   "
              f"FFT={t_fft*1e3:9.3f} ms (±{std_fft*1e3:.3f})   "
              f"numpy={t_numpy*1e3:9.3f} ms   "
              f"speedup DFT/FFT={resultado['speedup_dft_vs_fft_propria']:6.1f}x   "
              f"bate={resultado['resultados_batem']}")

    return resultados


def ajustar_curvas(N_vals, tempos, modelo):
    """
    Ajusta `modelo` (uma função de N e parâmetros) aos dados
    (N_vals, tempos) por mínimos quadrados (scipy.optimize.curve_fit).

    Como N_vals costuma cobrir várias ordens de grandeza (ex.: 16 até
    16384), o(s) parâmetro(s) do modelo tende a ser um número bem pequeno
    (ex.: o `a` de a*N² fica na casa de 1e-9 para tempos em segundos). O
    chute inicial padrão do curve_fit (1.0 para todo parâmetro) pode não
    convergir bem nessa escala, então estimamos um `p0` a partir do maior
    ponto medido antes de chamar curve_fit.

    Retorna (params, curva_ajustada), onde curva_ajustada = modelo(N_vals,
    *params) — já pronta para sobrepor aos pontos medidos num gráfico.
    """
    N_vals = np.asarray(N_vals, dtype=float)
    tempos = np.asarray(tempos, dtype=float)

    referencia = modelo(N_vals[-1], 1.0)
    p0 = [tempos[-1] / referencia] if referencia not in (0, np.inf) and not np.isnan(referencia) else [1.0]

    params, _ = curve_fit(modelo, N_vals, tempos, p0=p0, maxfev=10000)
    return params, modelo(N_vals, *params)