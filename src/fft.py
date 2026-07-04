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
    return 1 << (n - 1).bit_length()


def preparar_sinal_para_fft(x):
    """
    Deixa o sinal pronto para `fft_divisao_conquista`:

    1. Remove a média (detrend simples) — evita que um nível DC grande
       domine o espectro e atrapalhe a leitura das outras frequências.
    2. Faz zero-padding até a próxima potência de 2, já que o algoritmo
       Radix-2 exige N = 2^m (ver aviso no Plano de Execução, Passo 4).

    Retorna (sinal_pronto, N_pad), onde N_pad é o novo tamanho (potência de
    2). Guardamos N_pad separadamente porque é ele — não o N original — que
    deve ser usado depois para calcular o eixo de frequências e para
    comparar com a complexidade teórica O(N log N).
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    N_original = len(x)
    N_pad = proxima_potencia_de_2(N_original)
    sinal = np.zeros(N_pad, dtype=complex)
    sinal[:N_original] = x
    return sinal, N_pad


def fft_divisao_conquista(x):
    """
    FFT Radix-2 recursiva (Cooley-Tukey), estilo Divisão e Conquista.

    Exige len(x) potência de 2 (use `preparar_sinal_para_fft` antes, se o
    sinal original não for).

    -- Divisão --
    O vetor de entrada é dividido pelos índices pares (x[0::2]) e ímpares
    (x[1::2]), não em "primeira metade / segunda metade". Cada metade tem
    tamanho N/2 e é resolvida recursivamente (essa é a parte "Divisão" da
    técnica: mesmo problema, em instâncias menores).

    -- Conquista (caso base) --
    Quando N <= 1, a "transformada" de um único ponto é o próprio ponto:
    custo O(1), e é isso que garante que a recursão termina.

    -- Combinação (borboleta) --
    Sejam `pares` = FFT dos índices pares (equivalente a E[k] na literatura)
    e `impares` = FFT dos índices ímpares (O[k]). A saída completa de
    tamanho N é reconstruída em O(N) com a fórmula do "twiddle factor":

        X[k]       = E[k] + W_N^k * O[k]         para k = 0 .. N/2-1
        X[k + N/2] = E[k] - W_N^k * O[k]

    O código abaixo calcula só um vetor `fator` com todos os N expoentes de
    uma vez (`fator = W_N^0, W_N^1, ..., W_N^{N-1}`) e usa a segunda metade
    dele (`fator[N//2:]`) para a segunda parte da saída. Isso funciona
    porque W_N^{k+N/2} = -W_N^k (identidade clássica da FFT — o fator gira
    meia volta a mais), ou seja, `fator[N//2:]` já é `-fator[:N//2]`, e
    `pares + fator[N//2:]*impares` é exatamente `pares - fator[:N//2]*impares`,
    a fórmula de X[k+N/2] acima, só escrita sem o sinal de menos explícito.

    Cada nível de recursão faz O(N) trabalho de combinação (a soma e a
    multiplicação pelo twiddle, vetorizadas em numpy), e há O(log N) níveis
    (o vetor cai pela metade a cada chamada) — daí a recorrência
    T(N) = 2*T(N/2) + O(N), que pelo Teorema Mestre (caso 2) resolve para
    Θ(N log N) (ver FFT_e_Análise_Sísmica.md, seção 3).
    """
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("N precisa ser potência de 2 (use zero-padding).")

    pares = fft_divisao_conquista(x[0::2])
    impares = fft_divisao_conquista(x[1::2])

    fator = np.exp(-2j * np.pi * np.arange(N) / N)
    primeira_metade = pares + fator[:N // 2] * impares
    segunda_metade = pares + fator[N // 2:] * impares

    return np.concatenate([primeira_metade, segunda_metade])