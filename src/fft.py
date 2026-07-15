"""
FFT Radix-2 por Divisão e Conquista (algoritmo de Cooley-Tukey).

Complexidade: O(N log N), contra O(N²) da DFT força bruta (src/dft.py).
Pré-requisito do algoritmo: N precisa ser potência de 2 (por isso existe
`preparar_sinal_para_fft`, que faz zero-padding até a próxima potência de 2
antes de chamar `fft_divisao_conquista`).
"""

import numpy as np


def proxima_potencia_de_2(n):
    """
    Menor potência de 2 que é >= n.

    Truque de bits: `(n-1).bit_length()` dá a quantidade de bits necessários
    para representar (n-1); `1 << essa_quantidade` monta 2 elevado a esse
    expoente. Exemplos: n=8 (já é potência de 2) -> 8; n=9 -> 16; n=1 -> 1.
    """
    if n < 1:
        raise ValueError("n precisa ser um inteiro positivo.")
    return 1 << (n - 1).bit_length()


def preparar_sinal_para_fft(x):
    """
    Deixa o sinal pronto para `fft_divisao_conquista`:

    1. Remove a média (detrend simples) — evita que um nível DC grande
       domine o espectro e atrapalhe a leitura das outras frequências.
    2. Faz zero-padding até a próxima potência de 2, já que o algoritmo
       Radix-2 exige N = 2^m.

    Retorna (sinal_pronto, N_pad), onde N_pad é o novo tamanho (potência de
    2). Guardamos N_pad separadamente porque é ele — não o N original — que
    deve ser usado depois para calcular o eixo de frequências e para
    comparar com a complexidade teórica O(N log N).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("O sinal deve ser um vetor unidimensional não vazio.")
    x = x - np.mean(x)
    N_original = len(x)
    N_pad = proxima_potencia_de_2(N_original)
    sinal = np.zeros(N_pad, dtype=complex)
    sinal[:N_original] = x
    return sinal, N_pad


def fft_divisao_conquista(x):
    """Calcula a FFT Radix-2 recursiva pelo método de Cooley-Tukey.

    A implementação explicita as três etapas de Divisão e Conquista:

    **Divisão**
        Separa o vetor pelos índices pares (``x[0::2]``) e ímpares
        (``x[1::2]``). Não é uma separação entre primeira e segunda metade:
        cada subproblema conserva amostras alternadas do vetor original.

    **Conquista / caso base**
        Resolve recursivamente as duas metades. Quando ``N == 1``, a DFT de
        uma amostra é a própria amostra; esse caso custa O(1) e encerra a
        recursão.

    **Combinação**
        Une as transformadas par ``E[k]`` e ímpar ``O[k]`` pelas borboletas::

            X[k]       = E[k] + W_N^k O[k]
            X[k + N/2] = E[k] - W_N^k O[k]

        onde ``W_N^k = exp(-2πik/N)``. O vetor ``fator`` contém todos os
        expoentes; sua segunda metade já inclui o sinal negativo porque
        ``W_N^(k+N/2) = -W_N^k``.

    Em cada nível são feitas O(N) operações de combinação, e há log2(N)
    níveis, pois o tamanho cai pela metade. Logo,
    ``T(N) = 2T(N/2) + O(N) = O(N log N)``. A DFT direta, em contraste,
    calcula N somas com N termos e custa O(N²).

    Parameters
    ----------
    x : array_like
        Vetor real ou complexo, não vazio, cujo tamanho seja potência de 2.
        Use :func:`preparar_sinal_para_fft` quando essa condição não for
        garantida.

    Returns
    -------
    numpy.ndarray
        N coeficientes complexos, na mesma convenção de ``numpy.fft.fft``.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 0 or N & (N - 1) != 0:
        raise ValueError(
            "O tamanho do vetor precisa ser uma potência de 2 "
            "(use preparar_sinal_para_fft)."
        )
    if N == 1:
        return x

    # DIVISÃO: dois subproblemas independentes de tamanho N/2.
    pares = fft_divisao_conquista(x[0::2])
    impares = fft_divisao_conquista(x[1::2])

    # COMBINAÇÃO: as duas metades da saída são borboletas com +W e -W.
    fator = np.exp(-2j * np.pi * np.arange(N) / N)
    primeira_metade = pares + fator[:N // 2] * impares
    segunda_metade = pares + fator[N // 2:] * impares

    return np.concatenate([primeira_metade, segunda_metade])
