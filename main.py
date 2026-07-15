"""
Script principal — integra os módulos de `src/` nas duas análises do projeto.
O contexto teórico e as instruções de reprodução estão no `README.md`:

1. FASE 1 — Benchmark de complexidade (sinal sintético)
   Compara DFT força bruta, FFT própria (Divisão e Conquista) e
   `numpy.fft.fft` para uma faixa de N (potências de 2, 2⁴ a 2¹⁴), com
   repetições e desvio padrão (`benchmark.benchmark_avancado`). Gera
   `resultados/benchmark_avancado.csv` e o gráfico correspondente
   (`visualization.plot_benchmark_avancado`, que já inclui o ajuste das
   curvas teóricas O(N²) e O(N log N) — ver `benchmark.ajustar_curvas`).
   Também gera aqui os diagramas DIDÁTICOS do algoritmo para N=8
   (`visualization.plot_divisao_fft_n8` e `plot_borboleta_n8`) — eles não
   fazem parte do benchmark em si (não medem tempo nem usam dado real ou
   sintético nenhum, só índices abstratos x[0..7]), mas ficam em
   `resultados/` porque, assim como o benchmark, não dependem de rede
   nem do sismo real — iam soar deslocados dentro de
   `resultados_fft_sismo/`, que é só para artefatos com dado real da
   Venezuela. Não depende de rede: só de numpy/scipy.

2. FASE 2 — Estudo de caso real (por padrão, Venezuela, 24/06/2026)
   Busca estações sismográficas na região definida pelo evento
   (`data.buscar_estacoes`), baixa a forma de onda vertical de cada uma
   (`data.baixar_forma_de_onda`), roda DFT e FFT sobre a mesma janela de
   análise e mede o tempo de cada uma (`processar_estacao`, abaixo).
   Gera mapa das estações, seção sísmica, espectros de frequência,
   comparação DFT×FFT por estação e resumo em CSV — tudo em
   `resultados_fft_sismo/`. Depende dos serviços FDSN (rede). Outro evento
   pode ser fornecido com `--evento arquivo.json`.

As duas fases são independentes de propósito — uma é síntese/teoria, a
outra é o dado real — por isso vivem em funções separadas
(`rodar_benchmark_complexidade` / `rodar_estudo_de_caso_real`) em vez de
código solto no nível do módulo. Isso permite rodar/depurar uma sem
esperar pela outra (a Fase 2 é a mais lenta e a única que pode falhar por
motivo fora do nosso controle, tipo um serviço FDSN fora do ar).

Uso: `python main.py`, a partir da raiz do projeto (onde está a pasta
`src/`).
"""

import argparse
import csv
import os
import time

import numpy as np

from src.data import (
    buscar_estacoes,
    baixar_forma_de_onda,
    carregar_evento,
    tempo_chegada_p_teorico,
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
    plot_tempo_frequencia,
    plot_espectros,
    plot_benchmark,
    salvar_resumo_csv,
    plot_divisao_fft_n8,
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
N_ANALISE = 2048       # 2¹¹ amostras: potência de 2 e custo DFT ainda viável
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
    assintótica vs. desempenho medido.

    Também gera, na mesma pasta, os dois diagramas didáticos do algoritmo
    para N=8 (`visualization.plot_divisao_fft_n8` e `plot_borboleta_n8`).
    Eles não entram no CSV nem têm relação matemática com o benchmark —
    é só conveniência de organização: como nenhum dos dois pipelines
    "genéricos" (este e os diagramas) depende de rede ou do sismo real,
    faz mais sentido morarem juntos aqui do que em `resultados_fft_sismo/`
    (reservada para artefatos com dado real da Venezuela).

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

    caminho_divisao = os.path.join(diretorio_saida, "divisao_n8.png")
    plot_divisao_fft_n8(caminho_divisao)
    caminho_borboleta = os.path.join(diretorio_saida, "borboleta_n8.png")
    plot_borboleta_n8(caminho_borboleta)
    print(f"  -> Diagrama de divisão salvo em:  {caminho_divisao}")
    print(f"  -> Diagrama da borboleta salvo em: {caminho_borboleta}")

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


def processar_estacao(estacao, tr, evento=None):
    """
    Aplica o pipeline DFT/FFT completo à forma de onda `tr` de uma
    estação: recorta a janela de análise (`extrair_janela_analise`),
    prepara o sinal (detrend + zero-padding até potência de 2, via
    `fft.preparar_sinal_para_fft`), roda DFT força bruta e FFT própria
    sobre o MESMO sinal preparado e cronometra as duas.

    `evento` é o dicionário carregado por `data.carregar_evento`. Mantê-lo
    como parâmetro — em vez de consultar constantes globais — é o que permite
    repetir o estudo com outro terremoto.

    Retorna um dict combinando os dados originais da estação (`**estacao`
    — rede, estação, lat/lon, dist_km, taxa de amostragem, vindos de
    `data.buscar_estacoes`) com os resultados da análise. Esse é
    exatamente o formato que `visualization.py` espera: cada função de
    plot lá (`plot_mapa_estacoes`, `plot_secao_sismica`, `plot_espectros`,
    `plot_benchmark`, `salvar_resumo_csv`) lê só o subconjunto de chaves
    que precisa, então um único dict "rico" como este alimenta todas.
    """
    evento = carregar_evento() if evento is None else evento
    t_chegada_p = tempo_chegada_p_teorico(estacao["dist_km"], evento)
    inicio_janela_s = t_chegada_p - MARGEM_PRE_P_S
    inicio_janela = evento["origem"] + inicio_janela_s
    janela_bruta, incompleta = extrair_janela_analise(tr, inicio_janela, N_ANALISE)
    sinal, N_pad = preparar_sinal_para_fft(janela_bruta)

    t0 = time.perf_counter()
    espec_dft = dft_forca_bruta(sinal)
    t_dft = time.perf_counter() - t0

    t0 = time.perf_counter()
    espec_fft = fft_divisao_conquista(sinal)
    t_fft = time.perf_counter() - t0

    resultados_batem = bool(np.allclose(espec_dft, espec_fft, atol=1e-6))

    # DOMÍNIO DA FREQUÊNCIA: `rfftfreq` cria um eixo de 0 Hz até a
    # frequência de Nyquist (taxa/2). Para sinal real, a metade negativa do
    # espectro é o espelho da positiva; por isso guardamos só N/2+1 bins.
    # A normalização começa em |X|/N e dobra apenas os bins internos. DC
    # (0 Hz) e Nyquist não têm par distinto e, portanto, não são dobrados.
    taxa = tr.stats.sampling_rate
    freqs = np.fft.rfftfreq(N_pad, d=1.0 / taxa)

    def espectro_unilateral(espectro):
        amplitude = np.abs(espectro[:len(freqs)]) / N_pad
        if len(amplitude) > 2:
            amplitude[1:-1] *= 2.0
        return amplitude

    amplitude_fft = espectro_unilateral(espec_fft)
    amplitude_dft = espectro_unilateral(espec_dft)

    return {
        **estacao,
        "evento_nome": evento["nome"],
        "taxa_amostragem_hz": taxa,
        "t_chegada_p_s": t_chegada_p,
        "inicio_janela_s": inicio_janela_s,
        "margem_pre_p_s": MARGEM_PRE_P_S,
        "janela_incompleta": incompleta,
        "trace_completo": tr,
        "janela_analise": janela_bruta,
        "N_pad": N_pad,
        # `amplitude` é mantido como alias por compatibilidade com figuras
        # antigas; novos consumidores devem preferir `amplitude_fft`.
        "freqs": freqs,
        "amplitude": amplitude_fft,
        "amplitude_fft": amplitude_fft,
        "amplitude_dft": amplitude_dft,
        "tempo_dft_s": t_dft,
        "tempo_fft_s": t_fft,
        "speedup": (t_dft / t_fft) if t_fft > 0 else float("nan"),
        "resultados_batem": resultados_batem,
    }


def rodar_estudo_de_caso_real(diretorio_saida=RESULTADOS_DIR, evento=None):
    """
    Fase 2 — estudo de caso com dados sísmicos reais do evento configurado:
    busca estações, baixa a forma de onda de cada uma, roda
    `processar_estacao` e gera 5 artefatos visuais + o CSV resumo em
    `diretorio_saida`.

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
    evento = carregar_evento() if evento is None else evento
    os.makedirs(diretorio_saida, exist_ok=True)
    print("=" * 60)
    print(f'FASE 2 — Estudo de caso real: {evento["nome"]}')
    print("=" * 60)
    print(f'Origem UTC: {evento["origem"]}')
    print(f'Epicentro: {evento["latitude"]:.4f}, {evento["longitude"]:.4f} | '
          f'M{evento["magnitude"]:.1f} | {evento["profundidade_km"]:.1f} km')

    print("\n[1/4] Buscando estações na região configurada...")
    estacoes = buscar_estacoes(evento)
    print(f"  -> {len(estacoes)} estações selecionadas")

    if not estacoes:
        print("Nenhuma estação encontrada.")
        return []

    print("\n[2/4] Baixando formas de onda e calculando DFT/FFT na mesma janela...")
    resultados = []
    for est in estacoes:
        rotulo = f'{est["rede"]}.{est["estacao"]} ({est["dist_km"]:.0f} km)'
        try:
            tr = baixar_forma_de_onda(
                est,
                evento=evento,
                n_analise=N_ANALISE,
                margem_pre_p_s=MARGEM_PRE_P_S,
            )
            res = processar_estacao(est, tr, evento)
            resultados.append(res)
            print(f"  [OK] {rotulo:35s} DFT={res['tempo_dft_s']*1000:7.2f}ms "
                  f"FFT={res['tempo_fft_s']*1000:7.2f}ms")
        except Exception as e:
            print(f"  [FALHOU] {rotulo:35s} -> {e}")

    if not resultados:
        print("Nenhum dado de forma de onda obtido.")
        return []

    print(f"\n[3/4] Gerando gráficos em ./{diretorio_saida}/ ...")
    plot_mapa_estacoes(
        resultados, os.path.join(diretorio_saida, "1_mapa_estacoes.png"), evento
    )
    plot_secao_sismica(
        resultados, os.path.join(diretorio_saida, "2_dominio_tempo.png"), evento
    )
    plot_tempo_frequencia(
        resultados, os.path.join(diretorio_saida, "3_tempo_x_frequencia.png")
    )
    plot_espectros(
        resultados, os.path.join(diretorio_saida, "4_dominio_frequencia.png")
    )
    plot_benchmark(
        resultados, os.path.join(diretorio_saida, "5_benchmark_dft_fft.png")
    )
    salvar_resumo_csv(resultados, os.path.join(diretorio_saida, "resumo_estacoes.csv"))

    print("\n[4/4] Resumo do benchmark:")
    print(f'  {"Estação":15s} {"Dist (km)":>10s} {"DFT (ms)":>10s} {"FFT (ms)":>10s} {"Ganho":>8s}')
    for r in resultados:
        print(f'  {r["rede"]+"."+r["estacao"]:15s} {r["dist_km"]:10.0f} '
              f'{r["tempo_dft_s"]*1000:10.2f} {r["tempo_fft_s"]*1000:10.2f} '
              f'{r["speedup"]:7.1f}x')

    return resultados


def criar_parser_argumentos():
    """Monta a interface de linha de comando exibida por ``--help``.

    A separação em uma função facilita testar a interpretação dos argumentos
    sem iniciar benchmark nem downloads. O uso básico continua sendo
    ``python main.py``; as opções existem principalmente para o professor
    reproduzir apenas uma fase ou fornecer outro terremoto.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compara DFT e FFT por Divisão e Conquista e aplica ambas a "
            "formas de onda de um terremoto real."
        )
    )
    parser.add_argument(
        "--fase",
        choices=("todas", "benchmark", "sismo"),
        default="todas",
        help=(
            "'benchmark' roda só a análise O(N²) x O(N log N); 'sismo' "
            "roda só o estudo real; padrão: todas."
        ),
    )
    parser.add_argument(
        "--evento",
        metavar="ARQUIVO.json",
        help=(
            "configuração de outro terremoto; use evento_exemplo.json como "
            "modelo. Sem esta opção, usa o evento apresentado em sala."
        ),
    )
    parser.add_argument(
        "--saida-sismo",
        default=RESULTADOS_DIR,
        metavar="PASTA",
        help=f"pasta dos resultados sísmicos (padrão: {RESULTADOS_DIR}).",
    )
    return parser


def main(argv=None):
    """Executa as fases escolhidas na linha de comando.

    Exemplos
    --------
    Caso apresentado, duas fases::

        python main.py

    Outro terremoto, sem repetir o benchmark sintético::

        python main.py --fase sismo --evento evento_exemplo.json \
            --saida-sismo resultados_outro_evento
    """
    args = criar_parser_argumentos().parse_args(argv)

    if args.fase in ("todas", "benchmark"):
        rodar_benchmark_complexidade()

    if args.fase in ("todas", "sismo"):
        if args.fase == "todas":
            print()
        evento = carregar_evento(args.evento)
        rodar_estudo_de_caso_real(args.saida_sismo, evento)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
