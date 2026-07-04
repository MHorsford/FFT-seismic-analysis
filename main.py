"""
Script principal — amarra os módulos de `src/` nas duas análises do
projeto (ver `FFT_e_Análise_Sísmica.md` e `Plano_de_Execucao_Tarefa2_FFT.md`
para o contexto teórico e o plano completo):

1. FASE 1 — Benchmark de complexidade (sinal sintético)
   Compara DFT força bruta, FFT própria (Divisão e Conquista) e
   `numpy.fft.fft` para uma faixa de N (potências de 2, 2⁴ a 2¹⁴), com
   repetições e desvio padrão (`benchmark.benchmark_avancado`). Gera
   `resultados/benchmark_avancado.csv` e o gráfico correspondente
   (`visualization.plot_benchmark_avancado`, que já inclui o ajuste das
   curvas teóricas O(N²) e O(N log N) — ver `benchmark.ajustar_curvas`).
   Não depende de rede: só de numpy/scipy.

2. FASE 2 — Estudo de caso real (sismos da Venezuela, 24/06/2026)
   Busca estações sismográficas entre a Venezuela e o Pará
   (`data.buscar_estacoes`), baixa a forma de onda vertical de cada uma
   (`data.baixar_forma_de_onda`), roda DFT e FFT sobre a mesma janela de
   análise e mede o tempo de cada uma (`processar_estacao`, abaixo).
   Gera mapa das estações, seção sísmica, espectros de frequência,
   comparação DFT×FFT por estação, resumo em CSV e o diagrama da
   borboleta para N=8 — tudo em `resultados_fft_sismo/`. Depende dos
   serviços FDSN (rede).

As duas fases são independentes de propósito — uma é síntese/teoria, a
outra é o dado real — por isso vivem em funções separadas
(`rodar_benchmark_complexidade` / `rodar_estudo_de_caso_real`) em vez de
código solto no nível do módulo. Isso permite rodar/depurar uma sem
esperar pela outra (a Fase 2 é a mais lenta e a única que pode falhar por
motivo fora do nosso controle, tipo um serviço FDSN fora do ar).

Uso: `python main.py`, a partir da raiz do projeto (onde está a pasta
`src/`).
"""

import csv
import os
import time

import numpy as np

from src.data import (
    buscar_estacoes,
    baixar_forma_de_onda,
    tempo_chegada_p_teorico,
    ORIGEM_T0,
)
from src.dft import dft_forca_bruta
from src.benchmark import benchmark_avancado
from src.fft import (
    preparar_sinal_para_fft,
    fft_divisao_conquista,
)
from src.visualization import (
    plot_mapa_estacoes,
    plot_secao_sismica,
    plot_espectros,
    plot_benchmark,
    salvar_resumo_csv,
    plot_borboleta_n8,
    plot_benchmark_avancado,
)

# --------------------------------------------------------------------
# Configuração — Fase 1 (benchmark de complexidade)
# --------------------------------------------------------------------
LISTA_N_BENCHMARK = [2 ** k for k in range(4, 15)]  # 16, 32, 64, ..., 16384
NUM_REPETICOES_BENCHMARK = 3
DIR_BENCHMARK = "resultados"

# --------------------------------------------------------------------
# Configuração — Fase 2 (estudo de caso real)
# --------------------------------------------------------------------
N_ANALISE = 2048       # amostras por janela (ver FFT_e_Análise_Sísmica.md)
MARGEM_PRE_P_S = 10    # segundos "antes da onda P" incluídos na janela
RESULTADOS_DIR = "resultados_fft_sismo"


def rodar_benchmark_complexidade(lista_N=LISTA_N_BENCHMARK,
                                  num_repeticoes=NUM_REPETICOES_BENCHMARK,
                                  diretorio_saida=DIR_BENCHMARK):
    """
    Fase 1 — benchmark de complexidade sobre sinal sintético (ruído
    aleatório): roda DFT força bruta, FFT própria e numpy.fft.fft para
    cada N em `lista_N` (via `benchmark.benchmark_avancado`) e salva CSV
    + gráfico em `diretorio_saida`.

    É esse benchmark — não a tabela de ~186x da nota teórica, que conta
    OPERAÇÕES, não tempo — que sustenta a discussão de complexidade
    assintótica vs. desempenho medido (ver Plano de Execução, Passo 6).

    Aviso de tempo: o maior N padrão é 2¹⁴ = 16384; como a DFT força
    bruta é O(N²), esse ponto sozinho domina o tempo total do benchmark
    (a própria `_medir_tempo`, em benchmark.py, cita ~15s por chamada
    nesse N). Para uma checagem rápida em desenvolvimento, chame esta
    função com uma `lista_N` menor.

    Retorna a lista de dicionários produzida por `benchmark_avancado`
    (um por N), a mesma coisa que termina no CSV.
    """
    os.makedirs(diretorio_saida, exist_ok=True)

    print("=" * 60)
    print("FASE 1 — Benchmark de complexidade (sinal sintético)")
    print("=" * 60)
    print(f"N de {lista_N[0]} a {lista_N[-1]} ({len(lista_N)} pontos), "
          f"{num_repeticoes} repetições cada...\n")

    resultados_bench = benchmark_avancado(lista_N, num_repeticoes=num_repeticoes)

    caminho_csv = os.path.join(diretorio_saida, "benchmark_avancado.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=resultados_bench[0].keys())
        escritor.writeheader()
        escritor.writerows(resultados_bench)

    caminho_grafico = os.path.join(diretorio_saida, "benchmark_avancado.png")
    plot_benchmark_avancado(resultados_bench, caminho_grafico)

    print(f"\n  -> CSV salvo em:     {caminho_csv}")
    print(f"  -> Gráfico salvo em: {caminho_grafico}")

    return resultados_bench


def extrair_janela_analise(tr, inicio_janela, n_amostras):
    """
    Recorta `n_amostras` do Trace `tr` (ObsPy) a partir do instante
    `inicio_janela` (um UTCDateTime).

    Se a estação começou a gravar depois de `inicio_janela`, ou a
    transmissão foi cortada antes de completar a janela, o trecho baixado
    sai mais curto que `n_amostras`; em vez de descartar a estação
    inteira nesse caso, completamos com zeros à direita (zero-padding) e
    sinalizamos isso em `incompleta=True` — quem consome o retorno decide
    se isso é aceitável (aqui, ainda dá pra rodar DFT/FFT e ver o
    espectro, só que parte da janela reflete silêncio, não sinal real).
    """
    taxa = tr.stats.sampling_rate
    offset_amostras = int(round((inicio_janela - tr.stats.starttime) * taxa))
    offset_amostras = max(offset_amostras, 0)
    trecho = tr.data[offset_amostras: offset_amostras + n_amostras]
    incompleta = len(trecho) < n_amostras
    if incompleta:
        preenchido = np.zeros(n_amostras)
        preenchido[:len(trecho)] = trecho
        trecho = preenchido
    return trecho, incompleta


def processar_estacao(estacao, tr):
    """
    Aplica o pipeline DFT/FFT completo à forma de onda `tr` de uma
    estação: recorta a janela de análise (`extrair_janela_analise`),
    prepara o sinal (detrend + zero-padding até potência de 2, via
    `fft.preparar_sinal_para_fft`), roda DFT força bruta e FFT própria
    sobre o MESMO sinal preparado e cronometra as duas.

    Retorna um dict combinando os dados originais da estação (`**estacao`
    — rede, estação, lat/lon, dist_km, taxa de amostragem, vindos de
    `data.buscar_estacoes`) com os resultados da análise. Esse é
    exatamente o formato que `visualization.py` espera: cada função de
    plot lá (`plot_mapa_estacoes`, `plot_secao_sismica`, `plot_espectros`,
    `plot_benchmark`, `salvar_resumo_csv`) lê só o subconjunto de chaves
    que precisa, então um único dict "rico" como este alimenta todas.
    """
    t_chegada_p = tempo_chegada_p_teorico(estacao["dist_km"])
    inicio_janela = ORIGEM_T0 + t_chegada_p - MARGEM_PRE_P_S
    janela_bruta, incompleta = extrair_janela_analise(tr, inicio_janela, N_ANALISE)
    sinal, N_pad = preparar_sinal_para_fft(janela_bruta)

    t0 = time.perf_counter()
    espec_dft = dft_forca_bruta(sinal)
    t_dft = time.perf_counter() - t0

    t0 = time.perf_counter()
    espec_fft = fft_divisao_conquista(sinal)
    t_fft = time.perf_counter() - t0

    resultados_batem = bool(np.allclose(espec_dft, espec_fft, atol=1e-6))

    # Espectro de amplitude de um lado só (0 até Nyquist); o fator
    # 2/N_pad normaliza pra amplitude "física" de cada componente, já
    # compensando a energia que a FFT de sinal real espelha na metade
    # negativa do espectro (que aqui simplesmente descartamos com
    # `[:metade]`, já que ela é redundante para sinal real).
    taxa = tr.stats.sampling_rate
    freqs = np.arange(N_pad) * taxa / N_pad
    metade = N_pad // 2
    amplitude = np.abs(espec_fft) * 2.0 / N_pad

    return {
        **estacao,
        "t_chegada_p_s": t_chegada_p,
        "janela_incompleta": incompleta,
        "trace_completo": tr,
        "janela_analise": janela_bruta,
        "N_pad": N_pad,
        "freqs": freqs[:metade],
        "amplitude": amplitude[:metade],
        "tempo_dft_s": t_dft,
        "tempo_fft_s": t_fft,
        "speedup": (t_dft / t_fft) if t_fft > 0 else float("nan"),
        "resultados_batem": resultados_batem,
    }


def rodar_estudo_de_caso_real(diretorio_saida=RESULTADOS_DIR):
    """
    Fase 2 — estudo de caso com dados sísmicos reais dos sismos da
    Venezuela (24/06/2026): busca estações, baixa a forma de onda de cada
    uma, roda `processar_estacao` e gera os 5 artefatos visuais + o CSV
    resumo em `diretorio_saida`.

    Diferente da Fase 1, esta depende de rede (serviços FDSN via ObsPy):
    uma estação cujo download falhar é pulada (log "[FALHOU]") em vez de
    interromper o script inteiro; só desistimos de fato se NENHUMA
    estação for encontrada ou NENHUMA responder com dados (os dois
    `if not ...: return` abaixo). `data.buscar_estacoes` já tem seu
    próprio plano de contingência se os serviços de busca falharem (ver
    docstring/comentários em `data.py`).

    Retorna a lista de dicionários (um por estação bem-sucedida) — a
    mesma coisa que os gráficos e o CSV usam. Lista vazia se nada deu
    certo.
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    print("=" * 60)
    print("FASE 2 — Sismos da Venezuela (24/06/2026): estudo de caso real")
    print("=" * 60)

    print("\n[1/4] Buscando estações entre a Venezuela e o Pará...")
    estacoes = buscar_estacoes()
    print(f"  -> {len(estacoes)} estações selecionadas")

    if not estacoes:
        print("Nenhuma estação encontrada.")
        return []

    print("\n[2/4] Baixando formas de onda e aplicando DFT/FFT puras...")
    resultados = []
    for est in estacoes:
        rotulo = f'{est["rede"]}.{est["estacao"]} ({est["dist_km"]:.0f} km)'
        try:
            tr = baixar_forma_de_onda(est, n_analise=N_ANALISE, margem_pre_p_s=MARGEM_PRE_P_S)
            res = processar_estacao(est, tr)
            resultados.append(res)
            print(f"  [OK] {rotulo:35s} DFT={res['tempo_dft_s']*1000:7.2f}ms "
                  f"FFT={res['tempo_fft_s']*1000:7.2f}ms")
        except Exception as e:
            print(f"  [FALHOU] {rotulo:35s} -> {e}")

    if not resultados:
        print("Nenhum dado de forma de onda obtido.")
        return []

    print(f"\n[3/4] Gerando gráficos em ./{diretorio_saida}/ ...")
    plot_mapa_estacoes(resultados, os.path.join(diretorio_saida, "1_mapa_estacoes.png"))
    plot_secao_sismica(resultados, os.path.join(diretorio_saida, "2_secao_sismica.png"))
    plot_espectros(resultados, os.path.join(diretorio_saida, "3_espectros_fft.png"))
    plot_benchmark(resultados, os.path.join(diretorio_saida, "4_benchmark_dft_fft.png"))
    salvar_resumo_csv(resultados, os.path.join(diretorio_saida, "resumo_estacoes.csv"))
    plot_borboleta_n8(os.path.join(diretorio_saida, "5_borboleta_n8.png"))

    print("\n[4/4] Resumo do benchmark:")
    print(f'  {"Estação":15s} {"Dist (km)":>10s} {"DFT (ms)":>10s} {"FFT (ms)":>10s} {"Ganho":>8s}')
    for r in resultados:
        print(f'  {r["rede"]+"."+r["estacao"]:15s} {r["dist_km"]:10.0f} '
              f'{r["tempo_dft_s"]*1000:10.2f} {r["tempo_fft_s"]*1000:10.2f} '
              f'{r["speedup"]:7.1f}x')

    return resultados


def main():
    """
    Roda as duas fases em sequência: primeiro o benchmark de
    complexidade (rápido de reexecutar, sem rede), depois o estudo de
    caso real (mais lento, depende dos serviços FDSN).

    Para rodar só uma delas — por exemplo, ajustando os gráficos do
    estudo de caso sem recalcular o benchmark inteiro de novo — chame a
    função da fase correspondente diretamente (`rodar_benchmark_complexidade()`
    ou `rodar_estudo_de_caso_real()`) em vez de `main()`.
    """
    rodar_benchmark_complexidade()
    print()
    rodar_estudo_de_caso_real()
    print("\nConcluído.")


if __name__ == "__main__":
    main()