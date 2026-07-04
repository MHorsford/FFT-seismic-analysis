"""
Aquisição de dados sísmicos: FDSN, busca de estações e geração sintética.
"""

import warnings
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client, RoutingClient
from obspy.clients.fdsn.header import FDSNException
from obspy.geodetics import gps2dist_azimuth

try:
    from obspy.taup import TauPyModel
    _TAUP_DISPONIVEL = True
except Exception:
    _TAUP_DISPONIVEL = False

warnings.filterwarnings("ignore")

# -----------------------------------------------
# Parâmetros do evento sísmico
# -----------------------------------------------
EVENTOS = [
    {"nome": "Evento 1 (M7.2, foreshock)", "origem": UTCDateTime("2026-06-24T22:04:33"),
     "mag": 7.2, "profundidade_km": 21.9},
    {"nome": "Evento 2 (M7.5, mainshock)", "origem": UTCDateTime("2026-06-24T22:05:11"),
     "mag": 7.5, "profundidade_km": 10.0},
]

EVENTO_LAT = 10.435
EVENTO_LON = -68.472
EVENTO_PROFUNDIDADE_KM = EVENTOS[1]["profundidade_km"]
ORIGEM_T0 = EVENTOS[0]["origem"]

# -----------------------------------------------
# Configuração da busca de estações
# -----------------------------------------------
REGIAO_BUSCA = {
    "minlatitude": -10.0,
    "maxlatitude": 12.5,
    "minlongitude": -71.0,
    "maxlongitude": -46.0,
}

REDES_PERMITIDAS = "IU,CU,G,BR,BL,ON,NB"
REDES_BRASILEIRAS = "BR,BL,ON,NB"
FONTES_BUSCA = [
    ("EarthScope Federator", "earthscope-federator", REDES_PERMITIDAS),
    ("USP (reforço redes brasileiras)", "USP", REDES_BRASILEIRAS),
]

PRIORIDADE_CANAL = ["BHZ", "HHZ", "HNZ", "EHZ", "SHZ", "LHZ", "BHZ_00", "HHZ_00"]
PADRAO_CANAL_BUSCA = "*"
MAX_ESTACOES = 12
JANELA_PRE_EVENTO_S = 60
JANELA_DURACAO_S = 30 * 60

# Cache de clientes FDSN
_CLIENTES_CACHE = {}

_MODELO_TAUP = TauPyModel(model="iasp91") if _TAUP_DISPONIVEL else None

# -----------------------------------------------
# Funções auxiliares
# -----------------------------------------------

def _melhor_canal(estacao_obspy):
    canais_por_codigo = {}
    for cha in estacao_obspy.channels:
        codigo = cha.code.upper()
        if codigo not in canais_por_codigo:
            canais_por_codigo[codigo] = cha

    for prefixo in PRIORIDADE_CANAL:
        if prefixo in canais_por_codigo:
            return canais_por_codigo[prefixo]

    if estacao_obspy.channels:
        return estacao_obspy.channels[0]
    return None


def _obter_cliente(nome_datacenter):
    if nome_datacenter not in _CLIENTES_CACHE:
        if nome_datacenter in ("earthscope-federator", "iris-federator", "eida-routing"):
            _CLIENTES_CACHE[nome_datacenter] = RoutingClient(nome_datacenter)
        else:
            _CLIENTES_CACHE[nome_datacenter] = Client(nome_datacenter)
    return _CLIENTES_CACHE[nome_datacenter]


def tempo_chegada_p_teorico(dist_km):
    """Estima o tempo de chegada da onda P (TauP ou velocidade de 8 km/s)."""
    dist_deg = dist_km / 111.195
    if _MODELO_TAUP is not None:
        try:
            chegadas = _MODELO_TAUP.get_travel_times(
                source_depth_in_km=EVENTO_PROFUNDIDADE_KM,
                distance_in_degree=dist_deg,
                phase_list=["P", "p", "Pn"],
            )
            if chegadas:
                return chegadas[0].time
        except Exception:
            pass
    return dist_km / 8.0


# -----------------------------------------------
# Busca de estações (Venezuela → Pará)
# -----------------------------------------------
def buscar_estacoes():
    encontradas = {}
    for nome_fonte, nome_datacenter, redes in FONTES_BUSCA:
        print(f"  [busca] Consultando {nome_fonte} (redes: {redes})...")
        try:
            cliente = _obter_cliente(nome_datacenter)
            inventory = cliente.get_stations(
                network=redes,
                station="*",
                channel=PADRAO_CANAL_BUSCA,
                minlatitude=REGIAO_BUSCA["minlatitude"],
                maxlatitude=REGIAO_BUSCA["maxlatitude"],
                minlongitude=REGIAO_BUSCA["minlongitude"],
                maxlongitude=REGIAO_BUSCA["maxlongitude"],
                level="channel",
            )
        except Exception as e:
            print(f"    [!] {nome_fonte} não respondeu ({e}); seguindo...")
            continue

        novas = 0
        for network in inventory:
            for station in network:
                dist_km = gps2dist_azimuth(
                    EVENTO_LAT, EVENTO_LON, station.latitude, station.longitude
                )[0] / 1000.0

                if dist_km <= 500.0:
                    continue

                chave = (network.code, station.code)
                if chave in encontradas:
                    continue

                canal_ref = _melhor_canal(station)
                encontradas[chave] = {
                    "rede": network.code,
                    "estacao": station.code,
                    "datacenter": nome_datacenter,
                    "lat": station.latitude,
                    "lon": station.longitude,
                    "dist_km": dist_km,
                    "taxa_amostragem_hz": canal_ref.sample_rate if canal_ref else 20.0,
                }
                novas += 1

        print(f"    -> {novas} estação(ões) nova(s) via {nome_fonte}.")

    if not encontradas:
        print("  [PLANO B] Nenhuma fonte respondeu; ativando lista de contingência estática...")
        return [
            {"rede": "CU", "estacao": "GRGR", "datacenter": "earthscope-federator",
             "dist_km": 768.0, "lat": 12.112, "lon": -61.679, "taxa_amostragem_hz": 20.0},
            {"rede": "G", "estacao": "MPG", "datacenter": "earthscope-federator",
             "dist_km": 1842.0, "lat": 4.114, "lon": -52.021, "taxa_amostragem_hz": 20.0},
        ][:MAX_ESTACOES]

    lista_final = list(encontradas.values())
    lista_final.sort(key=lambda x: x["dist_km"])
    return lista_final[:MAX_ESTACOES]


# -----------------------------------------------
# Download da forma de onda real
# -----------------------------------------------
def baixar_forma_de_onda(estacao, n_analise=2048, margem_pre_p_s=10):
    """
    Baixa um trecho de forma de onda vertical da estação, centrado na
    janela de análise (usada depois pela DFT/FFT).
    """
    cliente = _obter_cliente(estacao["datacenter"])
    t_chegada_p = tempo_chegada_p_teorico(estacao["dist_km"])
    inicio = ORIGEM_T0 + t_chegada_p - margem_pre_p_s
    fim = inicio + (n_analise / estacao.get("taxa_amostragem_hz", 20.0))

    if estacao["rede"] in ("BR", "BL", "ON", "NB"):
        locais_candidatos = ["00", "*"]
    else:
        locais_candidatos = ["*"]

    for canal_teste in PRIORIDADE_CANAL:
        for loc_codigo in locais_candidatos:
            try:
                st = cliente.get_waveforms(
                    network=estacao["rede"],
                    station=estacao["estacao"],
                    location=loc_codigo,
                    channel=canal_teste,
                    starttime=inicio,
                    endtime=fim
                )
                if len(st) > 0:
                    tr = st[0]
                    tr.detrend("demean")
                    print(f"  [SUCESSO] {estacao['rede']}.{estacao['estacao']} "
                          f"canal {canal_teste} loc '{loc_codigo}'")
                    return tr
            except Exception:
                continue

    raise FDSNException("Nenhum canal vertical válido respondeu com dados.")


# -----------------------------------------------
# Sinal sintético (alternativa para testes)
# -----------------------------------------------
def gerar_sinal_sintetico(taxa_amostragem=20.0, duracao_s=120, freq_central=2.0,
                          amplitude=1.0, ruido=0.05):
    """
    Gera um sinal sintético com uma wavelet de Ricker e ruído branco.
    Útil quando a rede FDSN está indisponível.
    """
    t = np.arange(0, duracao_s, 1.0 / taxa_amostragem)
    t0 = duracao_s / 3  # centrada no início
    # wavelet de Ricker
    tau = t - t0
    wavelet = amplitude * (1 - 2 * (np.pi * freq_central * tau) ** 2) * \
              np.exp(-(np.pi * freq_central * tau) ** 2)
    ruido_sinal = ruido * np.random.randn(len(t))
    sinal = wavelet + ruido_sinal
    return t, sinal