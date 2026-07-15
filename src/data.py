"""Aquisição e configuração dos dados sísmicos usados no projeto.

O módulo concentra tudo o que depende do *evento* (origem, epicentro e
profundidade) e da rede FDSN.  Essa separação é importante porque os módulos de
DFT/FFT não precisam saber de onde o vetor numérico veio: para eles, um
sismograma real e um sinal sintético são apenas sequências de amostras.

O evento apresentado em sala continua sendo o padrão, mas não está mais
"preso" ao código.  :func:`carregar_evento` aceita um arquivo JSON com outro
terremoto; as funções públicas recebem esse dicionário e repetem exatamente o
mesmo pipeline.  Assim, o professor pode conferir a generalidade da aplicação
sem editar os algoritmos.
"""

import json
import warnings
from copy import deepcopy

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

# ---------------------------------------------------------------------------
# Evento e região usados na apresentação
# ---------------------------------------------------------------------------
# A região de busca não representa a área afetada pelo terremoto. Ela é apenas
# o retângulo dentro do qual consultamos estações: do epicentro, na Venezuela,
# até o Pará.  Mantê-la junto do evento permite que outro JSON use outro
# recorte geográfico sem alterar nenhuma função.
EVENTO_PADRAO = {
    "nome": "Venezuela — evento M7.2 de 24/06/2026",
    "origem": UTCDateTime("2026-06-24T22:04:33"),
    "magnitude": 7.2,
    "profundidade_km": 21.9,
    "latitude": 10.435,
    "longitude": -68.472,
    "distancia_min_km": 500.0,
    "regiao_busca": {
        "minlatitude": -10.0,
        "maxlatitude": 12.5,
        "minlongitude": -71.0,
        "maxlongitude": -46.0,
    },
}

# Campos obrigatórios no JSON. A validação antecipada produz uma mensagem
# compreensível, em vez de um KeyError aparecer só no meio de um download.
_CAMPOS_EVENTO = {
    "nome",
    "origem",
    "magnitude",
    "profundidade_km",
    "latitude",
    "longitude",
    "regiao_busca",
}
_CAMPOS_REGIAO = {
    "minlatitude",
    "maxlatitude",
    "minlongitude",
    "maxlongitude",
}

REDES_PERMITIDAS = "IU,CU,G,BR,BL,ON,NB"
REDES_BRASILEIRAS = "BR,BL,ON,NB"
FONTES_BUSCA = [
    ("EarthScope Federator", "earthscope-federator", REDES_PERMITIDAS),
    ("USP (reforço redes brasileiras)", "USP", REDES_BRASILEIRAS),
]

PRIORIDADE_CANAL = ["BHZ", "HHZ", "HNZ", "EHZ", "SHZ", "LHZ"]
PADRAO_CANAL_BUSCA = "*"
MAX_ESTACOES = 12

# Cache de clientes FDSN
_CLIENTES_CACHE = {}

_MODELO_TAUP = TauPyModel(model="iasp91") if _TAUP_DISPONIVEL else None

# -----------------------------------------------
# Configuração e funções auxiliares
# -----------------------------------------------


def carregar_evento(caminho_json=None):
    """Carrega e valida a configuração de um terremoto.

    Parameters
    ----------
    caminho_json : str ou None
        Quando ``None``, devolve uma cópia do evento usado na apresentação.
        Quando informado, deve apontar para um JSON com os mesmos campos de
        ``evento_exemplo.json``. ``origem`` usa o padrão ISO-8601 em UTC.

    Returns
    -------
    dict
        Configuração independente, com ``origem`` convertida para
        :class:`obspy.UTCDateTime`, pronta para as demais funções.

    Notes
    -----
    A cópia evita que uma execução altere acidentalmente o evento padrão. A
    validação verifica estrutura e limites físicos básicos; a existência real
    do evento é confirmada indiretamente quando os serviços FDSN devolvem (ou
    não) formas de onda para a data e a região informadas.
    """
    if caminho_json is None:
        evento = deepcopy(EVENTO_PADRAO)
    else:
        with open(caminho_json, "r", encoding="utf-8") as arquivo:
            evento = json.load(arquivo)

    ausentes = _CAMPOS_EVENTO - set(evento)
    if ausentes:
        raise ValueError(
            "Configuração do evento incompleta; faltam: "
            + ", ".join(sorted(ausentes))
        )

    regiao = evento["regiao_busca"]
    ausentes_regiao = _CAMPOS_REGIAO - set(regiao)
    if ausentes_regiao:
        raise ValueError(
            "regiao_busca incompleta; faltam: "
            + ", ".join(sorted(ausentes_regiao))
        )

    # UTCDateTime aceita tanto a string ISO do JSON quanto uma instância já
    # pronta do evento padrão. Os casts seguintes também aceitam números
    # escritos como inteiros no arquivo.
    evento["origem"] = UTCDateTime(evento["origem"])
    for campo in ("magnitude", "profundidade_km", "latitude", "longitude"):
        evento[campo] = float(evento[campo])
    # Distância mínima é opcional para eventos externos. No caso apresentado,
    # 500 km remove estações quase sobre o epicentro e destaca a propagação até
    # o Pará; para um teste local, o JSON pode usar 0 km.
    evento["distancia_min_km"] = float(evento.get("distancia_min_km", 0.0))
    for campo in _CAMPOS_REGIAO:
        regiao[campo] = float(regiao[campo])

    if not -90 <= evento["latitude"] <= 90:
        raise ValueError("latitude deve estar entre -90 e 90 graus.")
    if not -180 <= evento["longitude"] <= 180:
        raise ValueError("longitude deve estar entre -180 e 180 graus.")
    if evento["profundidade_km"] < 0:
        raise ValueError("profundidade_km não pode ser negativa.")
    if evento["distancia_min_km"] < 0:
        raise ValueError("distancia_min_km não pode ser negativa.")
    if regiao["minlatitude"] >= regiao["maxlatitude"]:
        raise ValueError("minlatitude deve ser menor que maxlatitude.")
    if regiao["minlongitude"] >= regiao["maxlongitude"]:
        raise ValueError("minlongitude deve ser menor que maxlongitude.")
    if not -90 <= regiao["minlatitude"] <= regiao["maxlatitude"] <= 90:
        raise ValueError("Os limites de latitude devem ficar entre -90 e 90.")
    if not -180 <= regiao["minlongitude"] <= regiao["maxlongitude"] <= 180:
        raise ValueError("Os limites de longitude devem ficar entre -180 e 180.")

    return evento

def _melhor_canal(estacao_obspy):
    """Escolhe o canal vertical de maior prioridade disponível.

    Uma estação pode publicar vários sensores. Para comparar estações de
    forma consistente, preferimos canais cuja última letra é ``Z``
    (componente vertical), seguindo ``PRIORIDADE_CANAL``. O retorno contém
    metadados; a forma de onda só é obtida em :func:`baixar_forma_de_onda`.
    """
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
    """Cria uma conexão FDSN uma vez e a reutiliza nas demais consultas."""
    if nome_datacenter not in _CLIENTES_CACHE:
        if nome_datacenter in ("earthscope-federator", "iris-federator", "eida-routing"):
            _CLIENTES_CACHE[nome_datacenter] = RoutingClient(nome_datacenter)
        else:
            _CLIENTES_CACHE[nome_datacenter] = Client(nome_datacenter)
    return _CLIENTES_CACHE[nome_datacenter]


def tempo_chegada_p_teorico(dist_km, evento=None):
    """Estima a chegada da onda P, em segundos após a origem do evento.

    O TauP/IASP91 calcula o tempo de viagem em um modelo estratificado da
    Terra. Se ele não estiver disponível ou não encontrar uma fase P, usamos
    ``distância / 8 km/s`` como aproximação explícita. Essa estimativa apenas
    posiciona a janela perto da primeira chegada; não faz parte da FFT.
    """
    evento = carregar_evento() if evento is None else evento
    dist_deg = dist_km / 111.195
    if _MODELO_TAUP is not None:
        try:
            chegadas = _MODELO_TAUP.get_travel_times(
                source_depth_in_km=evento["profundidade_km"],
                distance_in_degree=dist_deg,
                phase_list=["P", "p", "Pn"],
            )
            if chegadas:
                return chegadas[0].time
        except Exception:
            pass
    return dist_km / 8.0


# -----------------------------------------------
# Busca de estações
# -----------------------------------------------
def buscar_estacoes(evento=None):
    """Consulta estações na região do evento e devolve as mais próximas.

    Cada item retornado contém rede, código da estação, datacenter, posição,
    distância ao epicentro e taxa de amostragem do canal preferido. As fontes
    são independentes: a falha de uma não descarta o que outra já forneceu.
    Estações repetidas são removidas e o resultado é ordenado por distância.
    """
    evento = carregar_evento() if evento is None else evento
    regiao = evento["regiao_busca"]
    encontradas = {}
    for nome_fonte, nome_datacenter, redes in FONTES_BUSCA:
        print(f"  [busca] Consultando {nome_fonte} (redes: {redes})...")
        try:
            cliente = _obter_cliente(nome_datacenter)
            inventory = cliente.get_stations(
                network=redes,
                station="*",
                channel=PADRAO_CANAL_BUSCA,
                minlatitude=regiao["minlatitude"],
                maxlatitude=regiao["maxlatitude"],
                minlongitude=regiao["minlongitude"],
                maxlongitude=regiao["maxlongitude"],
                # Filtra canais cujo período de operação intersecta o instante
                # do sismo. Sem isso, uma estação instalada depois do evento
                # poderia aparecer na busca e falhar apenas no download.
                starttime=evento["origem"],
                endtime=evento["origem"] + 1,
                level="channel",
            )
        except Exception as e:
            print(f"    [!] {nome_fonte} não respondeu ({e}); seguindo...")
            continue

        novas = 0
        for network in inventory:
            for station in network:
                dist_km = gps2dist_azimuth(
                    evento["latitude"], evento["longitude"],
                    station.latitude, station.longitude
                )[0] / 1000.0

                if dist_km <= evento["distancia_min_km"]:
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
        # A contingência abaixo contém estações do caso Venezuela -> Pará.
        # Para outro evento, usá-la seria produzir um resultado enganoso; nesse
        # caso devolvemos vazio e o usuário pode revisar a região no JSON.
        evento_padrao = (
            abs(evento["latitude"] - EVENTO_PADRAO["latitude"]) < 1e-9
            and abs(evento["longitude"] - EVENTO_PADRAO["longitude"]) < 1e-9
            and evento["origem"] == EVENTO_PADRAO["origem"]
        )
        if not evento_padrao:
            print("  [!] Nenhuma estação encontrada para a região do evento informado.")
            return []
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
def baixar_forma_de_onda(estacao, evento=None, n_analise=2048,
                         margem_pre_p_s=10):
    """
    Baixa a janela vertical fornecida, sem mudança, à DFT e à FFT.

    O início é ``origem + chegada_P - margem``; portanto, há amostras antes
    da primeira onda P e depois dela. A duração é
    ``n_analise / taxa_amostragem``. Canais verticais e códigos de localização
    são tentados em ordem porque cada rede adota convenções diferentes.

    A primeira resposta válida é centralizada removendo a média (``demean``).
    Se nenhuma combinação responder, a exceção permite que o chamador pule só
    essa estação, em vez de interromper todo o estudo de caso.
    """
    evento = carregar_evento() if evento is None else evento
    cliente = _obter_cliente(estacao["datacenter"])
    t_chegada_p = tempo_chegada_p_teorico(estacao["dist_km"], evento)
    inicio = evento["origem"] + t_chegada_p - margem_pre_p_s
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
