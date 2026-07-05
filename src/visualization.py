"""
Geração dos gráficos e tabelas usados na apresentação.

Duas famílias de funções aqui, alimentadas por pipelines diferentes:

1. Estudo de complexidade com sinal SINTÉTICO (o que embasa a análise de
   algoritmo em si): `plot_benchmark_avancado`, alimentada por
   `benchmark.benchmark_avancado`. Essa é autossuficiente com o que está
   neste projeto.

2. Estudo de caso com o TERREMOTO REAL (Venezuela -> Pará): `plot_mapa_estacoes`,
   `plot_secao_sismica`, `plot_espectros`, `plot_benchmark` e `salvar_resumo_csv`.
   Essas esperam uma lista de dicionários por estação (ver docstring de cada
   uma para as chaves exigidas) que hoje é montada por um script "principal"
   que amarra `data.buscar_estacoes()` + `data.baixar_forma_de_onda()` com
   `dft`/`fft` — ou seja, essas funções já estão prontas para uso, mas
   dependem de esse script de orquestração (fora do escopo desta revisão)
   preencher os dicionários com os campos esperados.

3. Diagramas DIDÁTICOS do algoritmo em si, para N=8: `plot_divisao_fft_n8`
   (a árvore de recursão — pares/ímpares até as folhas) e `plot_borboleta_n8`
   (a combinação — folhas até a saída, com os twiddle factors). Diferente
   dos dois grupos acima, não dependem de nenhum dado (sintético ou real):
   são só a ilustração do próprio algoritmo `fft.fft_divisao_conquista`,
   pensadas para os slides que explicam "Divisão" e "Conquista"
   separadamente. As folhas de um viram a entrada do outro (mesma ordem
   bit-reversed), então funcionam melhor apresentadas em sequência.
"""

import csv

import matplotlib
matplotlib.use("Agg")  # backend não-interativo — ver nota abaixo
import matplotlib.pyplot as plt
import numpy as np

from src.data import EVENTO_LAT, EVENTO_LON, JANELA_PRE_EVENTO_S
from src.benchmark import ajustar_curvas

# Por que forçar o backend "Agg" (e por que ANTES de importar pyplot):
# --------------------------------------------------------------------
# Este módulo só faz fig.savefig(...) + plt.close(fig) — nunca plt.show().
# Sem essa linha, o matplotlib escolhe o backend automaticamente e, com
# Tkinter instalado (comum no Windows/Anaconda), tende a cair no "TkAgg",
# que cria uma janela do Tk de verdade pra cada figura, mesmo sem
# `plt.show()`. Numa execução em lote como esta — várias figuras criadas
# e fechadas em sequência, com chamadas de rede (ObsPy/FDSN) entre elas —
# isso pode terminar em erros do tipo:
#
#   Exception ignored in: <function Image.__del__ ...>
#   RuntimeError: main thread is not in main loop
#   Tcl_AsyncDelete: async handler deleted by the wrong thread
#
# porque o coletor de lixo do Python finaliza um objeto ligado ao Tk (uma
# Image ou Variable) fora da thread principal do Tk — e o Tk não é
# thread-safe. Esses erros aparecem como "Exception ignored in", ou seja,
# não interrompem o script, só poluem o terminal — mas não têm por que
# acontecer aqui, já que nenhuma janela é necessária: só queremos os
# PNGs em disco. "Agg" é um backend só de renderização (sem GUI/Tk), o
# que elimina o problema pela raiz e ainda é mais leve para esse caso de
# uso. Precisa ser definido antes do primeiro `import matplotlib.pyplot`
# do processo — depois disso, o backend já está carregado.


def plot_mapa_estacoes(resultados, caminho):
    """
    Mapa (lat/lon) do epicentro + estações encontradas, com a distância
    ao epicentro anotada ao lado de cada uma.

    `resultados`: lista de dicts, cada um precisando de "lon", "lat",
    "rede", "estacao", "dist_km" — exatamente o formato que
    `data.buscar_estacoes()` já devolve.
    """
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(EVENTO_LON, EVENTO_LAT, marker="*", s=400, color="red",
               zorder=5, label="Epicentro (Venezuela)")
    lons = [r["lon"] for r in resultados]
    lats = [r["lat"] for r in resultados]
    ax.scatter(lons, lats, s=80, color="royalblue", zorder=4,
               edgecolor="black", label="Estações encontradas")
    for r in resultados:
        ax.annotate(f'{r["rede"]}.{r["estacao"]}\n{r["dist_km"]:.0f} km',
                    (r["lon"], r["lat"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Estações sismográficas: Venezuela → Pará")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_secao_sismica(resultados, caminho):
    """
    "Seção sísmica": uma linha do tempo por estação (empilhadas no eixo Y
    por ordem de distância), cada uma mostrando a forma de onda normalizada
    e um marcador na chegada teórica da onda P. É o gráfico clássico para
    visualizar a propagação de um terremoto ao longo de uma rede.

    `resultados`: lista de dicts, cada um precisando de "trace_completo"
    (objeto Trace do ObsPy, o retorno de `data.baixar_forma_de_onda`),
    "rede", "estacao", "dist_km" e "t_chegada_p_s" (tempo da chegada
    teórica da onda P, em segundos desde a origem do Evento 1 — calculado
    a partir de `data.tempo_chegada_p_teorico`, já orquestrado com o T0 do
    evento).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, r in enumerate(resultados):
        tr = r["trace_completo"]
        t = np.arange(tr.stats.npts) / tr.stats.sampling_rate - JANELA_PRE_EVENTO_S
        dados = tr.data.astype(float)
        dados = dados - np.mean(dados)
        amp = dados / (np.max(np.abs(dados)) + 1e-9)  # normaliza p/ caber na faixa da estação
        ax.plot(t, amp * 0.4 + i, color="black", linewidth=0.5)
        ax.text(t[-1], i, f'  {r["rede"]}.{r["estacao"]} ({r["dist_km"]:.0f} km)',
                fontsize=7, va="center")
        ax.plot(r["t_chegada_p_s"], i, "r|", markersize=12)
    ax.axvline(0, color="blue", linestyle="--", linewidth=1,
               label="Origem (Evento 1, M7.2)")
    ax.plot([], [], "r|", label="Chegada teórica da onda P")  # entrada "fantasma" só p/ legenda
    ax.set_xlabel("Tempo desde a origem (s)")
    ax.set_ylabel("Estações (ordenadas pela distância do epicentro)")
    ax.set_yticks([])
    ax.set_title("Seção sísmica — propagação do sismo (Venezuela → Pará)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_espectros(resultados, caminho, freq_max_hz=10):
    """
    Um subplot por estação com o espectro de amplitude (|FFT|) do trecho
    baixado, limitado a `freq_max_hz` (a maior parte da energia sísmica de
    interesse cai bem abaixo disso; cortar o eixo ajuda a leitura).

    `resultados`: lista de dicts, cada um precisando de "freqs" (eixo de
    frequência, ex.: `np.fft.fftfreq`) e "amplitude" (|FFT| já calculada),
    além de "rede", "estacao", "dist_km" para os rótulos.
    """
    n = len(resultados)
    fig, axs = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
    if n == 1:
        axs = [axs]
    for ax, r in zip(axs, resultados):
        ax.plot(r["freqs"], r["amplitude"], color="darkorange")
        ax.set_ylabel(f'{r["rede"]}.{r["estacao"]}\n{r["dist_km"]:.0f} km', fontsize=8)
        ax.set_xlim(0, min(freq_max_hz, r["freqs"][-1]))
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("Frequência (Hz)")
    fig.suptitle("Espectro de amplitude (FFT pura) por estação")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_benchmark(resultados, caminho):
    """
    Comparação DFT x FFT *por estação real* (não confundir com
    `plot_benchmark_avancado`, que é o estudo sintético por N).

    `resultados`: lista de dicts com "rede", "estacao", "tempo_dft_s",
    "tempo_fft_s", "speedup" — o mesmo formato de linha que
    `benchmark.benchmark_comparacao` produz, só que aqui rodado sobre a
    forma de onda real de cada estação em vez de um sinal sintético.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    rotulos = [f'{r["rede"]}.{r["estacao"]}' for r in resultados]
    tempos_dft = [r["tempo_dft_s"] * 1000 for r in resultados]
    tempos_fft = [r["tempo_fft_s"] * 1000 for r in resultados]
    x = np.arange(len(resultados))
    largura = 0.35
    ax1.bar(x - largura / 2, tempos_dft, largura, label="DFT força bruta O(N²)",
            color="firebrick")
    ax1.bar(x + largura / 2, tempos_fft, largura, label="FFT D&C O(N log N)",
            color="seagreen")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rotulos, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Tempo (ms)")
    ax1.set_title("Tempo de execução por estação")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")
    speedups = [r["speedup"] for r in resultados]
    ax2.bar(x, speedups, color="steelblue")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rotulos, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Ganho (DFT / FFT)")
    ax2.set_title("Quantas vezes a FFT é mais rápida que a DFT")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_divisao_fft_n8(caminho_saida, N=8):
    """
    Diagrama da etapa de DIVISÃO da FFT Radix-2 (Cooley-Tukey): a árvore
    de recursão que separa repetidamente o vetor em índices pares
    (x[0::2]) e ímpares (x[1::2]) até sobrar um único elemento por folha
    — exatamente a metade "Divisão" de `fft.fft_divisao_conquista`, sem
    a parte de combinação.

    Este diagrama é o par do `plot_borboleta_n8`: lá se vê a COMBINAÇÃO
    (bottom-up: das folhas até a saída final, com os twiddle factors);
    aqui se vê a DIVISÃO (top-down: da entrada até as folhas). Não é
    coincidência que as folhas deste diagrama, lidas de cima para baixo,
    saiam na mesma ordem "bit-reversed" (0, 4, 2, 6, 1, 5, 3, 7 para N=8)
    que alimenta a coluna "Entrada" da borboleta — dividir repetidamente
    por paridade de índice É o que produz essa ordem. Colocados lado a
    lado (ou em slides consecutivos), os dois diagramas contam a história
    completa de "Divisão e Conquista" sem sobreposição de conteúdo.

    Construção
    ----------
    A árvore é construída recursivamente (`construir`, função aninhada)
    guardando apenas os ÍNDICES em cada nó, não os valores — aqui
    interessa só QUEM vai para cada ramo, não calcular nenhuma
    transformada (isso já é feito por `fft.fft_divisao_conquista`).

    A posição vertical de cada nó é a média das posições de seus filhos
    (estilo dendrograma), calculada bottom-up em `calcular_posicoes`;
    isso distribui as N folhas uniformemente no eixo Y e mantém os nós
    internos centralizados sobre a própria subárvore. A posição
    horizontal é simplesmente a profundidade na árvore (raiz em x=0).

    Parâmetros
    ----------
    caminho_saida : str
        Caminho do PNG a ser salvo.
    N : int
        Tamanho do vetor de entrada. Precisa ser potência de 2 (mesma
        exigência do Radix-2). Padrão 8, para ficar consistente com
        `plot_borboleta_n8` na apresentação.
    """
    if N < 1 or N & (N - 1) != 0:
        raise ValueError("N precisa ser potência de 2.")

    def construir(indices):
        if len(indices) == 1:
            return {"indices": indices, "filhos": []}
        # Mesma divisão de `fft.fft_divisao_conquista`: pares primeiro,
        # ímpares depois — é essa ordem (não uma divisão "primeira
        # metade/segunda metade") que caracteriza o Radix-2 DIT.
        return {
            "indices": indices,
            "filhos": [construir(indices[0::2]), construir(indices[1::2])],
        }

    raiz = construir(list(range(N)))

    proxima_linha = [0]

    def calcular_posicoes(no, profundidade):
        no["x"] = profundidade
        if not no["filhos"]:
            no["y"] = proxima_linha[0]
            proxima_linha[0] += 1
        else:
            for filho in no["filhos"]:
                calcular_posicoes(filho, profundidade + 1)
            no["y"] = float(np.mean([f["y"] for f in no["filhos"]]))

    calcular_posicoes(raiz, 0)

    profundidade_max = int(np.log2(N))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(-0.5, profundidade_max + 0.7)
    ax.set_ylim(-0.7, N - 0.3)
    ax.invert_yaxis()  # folha 0 no topo, mesmo sentido de leitura da borboleta
    ax.set_xticks(range(profundidade_max + 1))

    rotulos_col = []
    for d in range(profundidade_max + 1):
        tamanho = N // (2 ** d)
        if d == 0:
            rotulos_col.append(f"Entrada\n(N={tamanho})")
        elif d == profundidade_max:
            rotulos_col.append("Folhas\n(N=1, bit-reversed)")
        else:
            rotulos_col.append(f"Divisão {d}\n(N={tamanho})")
    ax.set_xticklabels(rotulos_col)
    ax.set_yticks([])
    ax.set_title(f"FFT Radix-2 (Cooley–Tukey) — Etapa de DIVISÃO para N = {N}", fontsize=14)

    cor_par = "#1f77b4"
    cor_impar = "#d62728"

    def desenhar(no):
        eh_folha = not no["filhos"]
        if eh_folha:
            rotulo = f'x[{no["indices"][0]}]'
        else:
            rotulo = "x[" + ",".join(str(i) for i in no["indices"]) + "]"

        ax.plot(no["x"], no["y"], 'o', color='black', markersize=7, zorder=3)

        # Raiz: rótulo à esquerda. Folhas: rótulo à direita (mesmo estilo
        # de `plot_borboleta_n8` para "x[idx]"/"X[idx]"). Nós internos:
        # rótulo acima do nó, para não cruzar com as arestas.
        if no["x"] == 0:
            ax.text(no["x"] - 0.1, no["y"], rotulo, ha='right', va='center', fontsize=9)
        elif eh_folha:
            ax.text(no["x"] + 0.1, no["y"], rotulo, ha='left', va='center', fontsize=9)
        else:
            ax.text(no["x"], no["y"] - 0.32, rotulo, ha='center', va='bottom', fontsize=7.5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

        if eh_folha:
            return
        pares, impares = no["filhos"]
        ax.plot([no["x"], pares["x"]], [no["y"], pares["y"]], '-', color=cor_par, lw=1.6, zorder=1)
        ax.plot([no["x"], impares["x"]], [no["y"], impares["y"]], '-', color=cor_impar, lw=1.6, zorder=1)
        desenhar(pares)
        desenhar(impares)

    desenhar(raiz)

    ax.plot([], [], color=cor_par, lw=1.6, label="Pares — x[0::2]")
    ax.plot([], [], color=cor_impar, lw=1.6, label="Ímpares — x[1::2]")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=200)
    plt.close(fig)
    print(f"Diagrama de divisão salvo em: {caminho_saida}")


def plot_borboleta_n8(caminho_saida):
    """
    Diagrama da borboleta (butterfly) da FFT Radix-2 DIT para N=8: entrada
    em ordem bit-reversed, 3 estágios, saída em ordem natural.

    Cada borboleta combina um par de valores (a, no nó de cima; b, no nó
    de baixo) em dois valores de saída:

        saída_de_cima = a + W_N^k * b
        saída_de_baixo = a - W_N^k * b

    Ou seja, cada borboleta tem exatamente 4 arestas, não 3: a contribui
    (peso 1) para as DUAS saídas, e b contribui multiplicado pelo twiddle
    para as duas saídas também, com sinal invertido entre elas (+W / -W).
    As 4 arestas são desenhadas com dois estilos: as de peso 1 (linhas
    pretas, sem rótulo) e as de peso ±W_N^k (cinza, com o expoente do
    twiddle escrito ao lado — positivo para a saída de cima, negativo
    para a de baixo).
    """
    N = 8
    estagios = int(np.log2(N))
    x_pos = [0, 1, 2, 3, 4]  # 0: entrada, 1..3: após cada estágio, 4: saída final
    rotulos_col = ["Entrada\n(bit-reversed)", "Estágio 1", "Estágio 2", "Estágio 3",
                   "Saída\n(ordem natural)"]

    entrada_idx = [0, 4, 2, 6, 1, 5, 3, 7]   # ordem bit-reversed na entrada
    saida_idx = list(range(N))               # ordem natural na saída

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(-0.5, len(x_pos) - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rotulos_col)
    ax.set_yticks([])
    ax.invert_yaxis()  # x[0]/X[0] no topo, ordem de leitura mais natural
    ax.set_title("FFT Radix-2 (DIT) para N = 8 — Diagrama da Borboleta", fontsize=14)

    # --- Nós (bolinhas) e rótulos de entrada/saída ---
    # As colunas intermediárias (estágios 1..3) não têm um "índice" próprio
    # para rotular — são valores intermediários da recursão — por isso só
    # a primeira e a última coluna recebem texto (x[idx] / X[idx]).
    colunas_de_nos = [entrada_idx] + [list(range(N))] * 3 + [saida_idx]
    for col, idx_list in enumerate(colunas_de_nos):
        for linha, idx in enumerate(idx_list):
            ax.plot(x_pos[col], linha, 'o', color='black', markersize=6, zorder=3)
            if col == 0:
                ax.text(x_pos[col] - 0.12, linha, f'x[{idx}]', ha='right', va='center', fontsize=9)
            elif col == len(x_pos) - 1:
                ax.text(x_pos[col] + 0.12, linha, f'X[{idx}]', ha='left', va='center', fontsize=9)

    # --- Conexões (borboletas), estágio por estágio ---
    # No estágio s (0-indexado), a distância entre o nó "de cima" e o "de
    # baixo" de cada borboleta é dist = 2**s, e os blocos de borboletas
    # têm tamanho 2*dist. Dentro de um bloco, a i-ésima borboleta usa o
    # twiddle W_N^k com k = i * (N / (2*dist)) — é a mesma progressão de
    # expoentes usada implicitamente em `fft.fft_divisao_conquista`.
    for s in range(estagios):
        col_orig, col_dest = x_pos[s + 1], x_pos[s + 2]
        dist = 2 ** s
        for bloco_inicio in range(0, N, 2 * dist):
            for i in range(dist):
                linha_top = bloco_inicio + i
                linha_bot = linha_top + dist
                k = i * (N // (2 * dist))

                # Arestas de peso 1 (contribuição do nó de cima nas duas saídas)
                ax.plot([col_orig, col_dest], [linha_top, linha_top], 'k-', lw=1.3, zorder=1)
                ax.plot([col_orig, col_dest], [linha_top, linha_bot], 'k-', lw=1.3, zorder=1)

                # Arestas com twiddle (contribuição do nó de baixo, com sinal
                # oposto em cada saída): +W_N^k na saída de cima, -W_N^k na de baixo
                ax.plot([col_orig, col_dest], [linha_bot, linha_top], '-', color='gray', lw=1.0, zorder=1)
                ax.plot([col_orig, col_dest], [linha_bot, linha_bot], '--', color='gray', lw=1.0, zorder=1)

                rotulo_pos = f'$W_{{{N}}}^{{{k}}}$' if k != 0 else '$1$'
                rotulo_neg = f'$-W_{{{N}}}^{{{k}}}$' if k != 0 else '$-1$'
                meio_x = (col_orig + col_dest) / 2
                ax.text(meio_x, (linha_bot + linha_top) / 2, rotulo_pos,
                        fontsize=7, ha='center', va='center', color='dimgray',
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.5), zorder=2)
                ax.text(meio_x, linha_bot, rotulo_neg,
                        fontsize=7, ha='center', va='center', color='dimgray',
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.5), zorder=2)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=200)
    plt.close(fig)
    print(f"Diagrama borboleta salvo em: {caminho_saida}")


def plot_benchmark_avancado(resultados_bench, caminho_saida):
    """
    Figura com dois painéis a partir da saída de `benchmark.benchmark_avancado`:

    - Esquerda: tempo medido (DFT, FFT própria, numpy.fft) em escala
      log-log, com curvas a*N² e b*N*log2(N) ajustadas aos dados (ver
      `benchmark.ajustar_curvas`) sobrepostas — mostra visualmente se o
      crescimento medido acompanha a forma prevista pela teoria.
    - Direita: speedup DFT/FFT-própria e DFT/numpy.

    Se o ajuste de curva falhar para algum modelo (dados insuficientes,
    todos os tempos zerados por falta de resolução do timer etc.), o
    respectivo traçado teórico é simplesmente omitido — o gráfico ainda é
    gerado com os pontos medidos.
    """
    N_vals = np.array([r["N_pad"] for r in resultados_bench])
    t_dft = np.array([r["tempo_dft_s"] * 1000 for r in resultados_bench])      # ms
    t_fft_p = np.array([r["tempo_fft_propria_s"] * 1000 for r in resultados_bench])
    t_numpy = np.array([r["tempo_numpy_fft_s"] * 1000 for r in resultados_bench])
    speedup_p = np.array([r["speedup_dft_vs_fft_propria"] for r in resultados_bench])
    speedup_n = np.array([r["speedup_dft_vs_numpy"] for r in resultados_bench])

    def modelo_n2(n, a):
        return a * n ** 2

    def modelo_nlogn(n, b):
        return b * n * np.log2(n)

    # Ajustamos a*N² aos tempos da DFT e b*N*log2(N) aos tempos da FFT
    # própria — são os dois algoritmos cuja complexidade teórica estamos
    # verificando (numpy.fft.fft não entra aqui: é código em C com
    # otimizações internas que fogem do escopo da análise).
    try:
        params_n2, curva_n2 = ajustar_curvas(N_vals, t_dft, modelo_n2)
    except Exception as e:
        print(f"  [!] Não foi possível ajustar a curva O(N²): {e}")
        curva_n2 = None

    try:
        params_nlogn, curva_nlogn = ajustar_curvas(N_vals, t_fft_p, modelo_nlogn)
    except Exception as e:
        print(f"  [!] Não foi possível ajustar a curva O(N log N): {e}")
        curva_nlogn = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Painel 1: tempos medidos + ajustes teóricos ---
    ax1.loglog(N_vals, t_dft, 'o-', color='firebrick', label='DFT força bruta O(N²)', markersize=6)
    ax1.loglog(N_vals, t_fft_p, 's-', color='seagreen', label='FFT própria O(N log N)', markersize=6)
    ax1.loglog(N_vals, t_numpy, '^--', color='orange', label='numpy.fft.fft (implementação C)', markersize=6)

    if curva_n2 is not None:
        ax1.loglog(N_vals, curva_n2, ':', color='firebrick', alpha=0.6, label='Ajuste a·N²')
    if curva_nlogn is not None:
        ax1.loglog(N_vals, curva_nlogn, ':', color='seagreen', alpha=0.6, label='Ajuste b·N·log₂N')

    ax1.set_xlabel("Tamanho do vetor (N, já com zero-padding)")
    ax1.set_ylabel("Tempo (ms)")
    ax1.set_title("Tempo de execução (escala log-log)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    # --- Painel 2: speedup medido ---
    ax2.plot(N_vals, speedup_p, 'o-', color='seagreen', label='DFT / FFT própria')
    ax2.plot(N_vals, speedup_n, 's--', color='orange', label='DFT / numpy.fft')
    ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')  # abaixo disso, a DFT "vence"
    ax2.set_xlabel("Tamanho do vetor (N, já com zero-padding)")
    ax2.set_ylabel("Ganho (vezes mais rápida)")
    ax2.set_title("Speedup da DFT em relação às FFTs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def salvar_resumo_csv(resultados, caminho):
    """
    Salva o resumo por estação (dataset real, um resultado por estação —
    ver `plot_secao_sismica`/`plot_benchmark` para as chaves esperadas de
    cada dict) em CSV, pronto para anexar como tabela na apresentação.
    """
    campos = ["rede", "estacao", "lat", "lon", "dist_km",
              "taxa_amostragem_hz", "t_chegada_p_s", "janela_incompleta",
              "tempo_dft_s", "tempo_fft_s", "speedup", "resultados_batem"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for r in resultados:
            escritor.writerow({c: r.get(c, "") for c in campos})