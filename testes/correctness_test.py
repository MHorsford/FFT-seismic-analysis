"""Testes de corretude que não acessam a internet.

O objetivo não é medir desempenho — tempos variam entre computadores —, mas
demonstrar que a otimização por Divisão e Conquista preserva o resultado
matemático da DFT. A referência NumPy é independente das duas implementações
do projeto e reduz o risco de dois códigos com o mesmo erro parecerem corretos.

Execute na raiz com::

    python -m unittest discover -s testes -p "*_test.py" -v
"""

import json
import os
import tempfile
import unittest

import numpy as np

from src.data import carregar_evento
from src.dft import dft_forca_bruta
from src.fft import (
    fft_divisao_conquista,
    preparar_sinal_para_fft,
    proxima_potencia_de_2,
)
from src.visualization import plot_tempo_frequencia


class TestFFT(unittest.TestCase):
    """Confere casos pequenos, padding e erros de entrada."""

    def test_proxima_potencia_de_dois(self):
        self.assertEqual(proxima_potencia_de_2(1), 1)
        self.assertEqual(proxima_potencia_de_2(8), 8)
        self.assertEqual(proxima_potencia_de_2(9), 16)
        with self.assertRaises(ValueError):
            proxima_potencia_de_2(0)

    def test_preparacao_remove_media_e_completa_com_zeros(self):
        pronto, n_pad = preparar_sinal_para_fft([1.0, 2.0, 6.0])
        self.assertEqual(n_pad, 4)
        self.assertAlmostEqual(float(np.sum(pronto.real)), 0.0)
        self.assertEqual(pronto[-1], 0.0)

    def test_fft_bate_com_dft_e_numpy(self):
        gerador = np.random.default_rng(2026)
        for n in (1, 2, 4, 8, 16, 32):
            with self.subTest(n=n):
                sinal = gerador.normal(size=n)
                obtido = fft_divisao_conquista(sinal)
                np.testing.assert_allclose(obtido, dft_forca_bruta(sinal), atol=1e-9)
                np.testing.assert_allclose(obtido, np.fft.fft(sinal), atol=1e-9)

    def test_fft_rejeita_tamanho_incompativel(self):
        with self.assertRaises(ValueError):
            fft_divisao_conquista([])
        with self.assertRaises(ValueError):
            fft_divisao_conquista([1, 2, 3])


class TestEventoConfiguravel(unittest.TestCase):
    """Garante que outro evento pode ser fornecido sem alterar o padrão."""

    def test_evento_padrao_eh_copiado(self):
        primeiro = carregar_evento()
        primeiro["nome"] = "alterado apenas no teste"
        segundo = carregar_evento()
        self.assertNotEqual(primeiro["nome"], segundo["nome"])

    def test_carrega_json_e_valida_campos(self):
        configuracao = {
            "nome": "Evento de teste",
            "origem": "2024-01-02T03:04:05Z",
            "magnitude": 5.5,
            "profundidade_km": 12.0,
            "latitude": -1.5,
            "longitude": -48.0,
            "regiao_busca": {
                "minlatitude": -3,
                "maxlatitude": 1,
                "minlongitude": -51,
                "maxlongitude": -46,
            },
        }
        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "evento.json")
            with open(caminho, "w", encoding="utf-8") as arquivo:
                json.dump(configuracao, arquivo)
            evento = carregar_evento(caminho)

        self.assertEqual(evento["nome"], "Evento de teste")
        self.assertEqual(float(evento["magnitude"]), 5.5)
        self.assertEqual(str(evento["origem"]), "2024-01-02T03:04:05.000000Z")

    def test_rejeita_json_incompleto(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "evento.json")
            with open(caminho, "w", encoding="utf-8") as arquivo:
                json.dump({"nome": "incompleto"}, arquivo)
            with self.assertRaises(ValueError):
                carregar_evento(caminho)


class TestVisualizacaoDidatica(unittest.TestCase):
    """Verifica que a comparação tempo × frequência gera uma figura."""

    def test_grafico_tempo_frequencia(self):
        taxa = 20.0
        n = 64
        t = np.arange(n) / taxa
        sinal = np.sin(2 * np.pi * 2.0 * t)
        freqs = np.fft.rfftfreq(n, d=1.0 / taxa)
        amplitude = np.abs(np.fft.rfft(sinal)) / n
        resultado = {
            "rede": "XX",
            "estacao": "TESTE",
            "dist_km": 100.0,
            "taxa_amostragem_hz": taxa,
            "janela_analise": sinal,
            "freqs": freqs,
            "amplitude_fft": amplitude,
            # A função só usa `stats.sampling_rate` como fallback; este objeto
            # mínimo mantém o teste independente de uma forma de onda real.
            "trace_completo": type(
                "TraceMinimo", (), {"stats": type("Stats", (), {"sampling_rate": taxa})()}
            )(),
        }
        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "tempo_frequencia.png")
            plot_tempo_frequencia([resultado], caminho)
            self.assertTrue(os.path.exists(caminho))
            self.assertGreater(os.path.getsize(caminho), 0)


if __name__ == "__main__":
    unittest.main()
