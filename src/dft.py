"""
DFT (Transformada Discreta de Fourier) por força bruta.

Esta é a implementação "ingênua", usada como referência O(N²) para comparar
com a FFT de Divisão e Conquista (src/fft.py). A ideia é a definição direta
da DFT, sem nenhum truque algorítmico:

    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N),   para cada k em [0, N)

Ou seja, X = M @ x, onde M é a matriz N x N com M[k, n] = exp(-2j*pi*k*n/N).
"""

import numpy as np


def dft_forca_bruta(x, tamanho_bloco=1024):
    """
    Calcula a DFT de `x` pela definição direta (força bruta).

    Complexidade: O(N²) tempo — para cada uma das N frequências de saída,
    somamos N termos. Não há nenhuma otimização algorítmica aqui; é
    exatamente essa a razão de existir desta função: servir de referência
    "lenta, mas obviamente correta" para comparar com a FFT (O(N log N)).

    Detalhe de implementação — por que em blocos?
    ------------------------------------------------
    A forma mais direta de escrever isso em numpy é montar a matriz M
    completa (N x N) de uma vez e fazer `M @ x`. O problema é que essa
    matriz sozinha ocupa N² * 16 bytes (complex128) de memória: para
    N = 2¹⁴ = 16384 (um tamanho perfeitamente razoável de se querer testar
    num benchmark de complexidade), isso são ~4,3 GB só da matriz — o
    suficiente para travar ou derrubar por falta de memória um notebook
    comum, especialmente se o benchmark estiver rodando vários N em
    sequência.

    Para evitar isso, calculamos `tamanho_bloco` linhas de M por vez (um
    "bloco" de frequências k) em vez da matriz inteira. O número de
    operações de ponto flutuante continua exatamente o mesmo — ainda é
    O(N²), ainda é a definição direta da DFT — só o *pico de memória* cai
    de O(N²) para O(N * tamanho_bloco). Isso é puramente uma questão de
    engenharia (permitir testar N maiores sem estourar RAM); não muda em
    nada a complexidade algorítmica que está sendo demonstrada.

    Parâmetros
    ----------
    x : array_like
        Sinal de entrada (real ou complexo), tamanho N.
    tamanho_bloco : int
        Quantas linhas da matriz M são calculadas por vez. Não afeta o
        resultado, só o uso de memória. O padrão (1024) mantém o pico de
        memória em dezenas de MB mesmo para N grande.

    Retorna
    -------
    np.ndarray (complex128) de tamanho N com o espectro X.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    X = np.empty(N, dtype=complex)

    for inicio in range(0, N, tamanho_bloco):
        fim = min(inicio + tamanho_bloco, N)
        k = np.arange(inicio, fim).reshape(-1, 1)   # bloco de frequências, como coluna
        M_bloco = np.exp(-2j * np.pi * k * n / N)   # (fim-inicio) x N
        X[inicio:fim] = M_bloco @ x

    return X