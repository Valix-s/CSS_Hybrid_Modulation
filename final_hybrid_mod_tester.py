from typing import Optional

import numpy as np
import adi
import scipy.signal
from scipy.sparse import coo_matrix, csr_matrix
from ldpc.bp_decoder import BpDecoder
import time
import serial
import csv
from functools import lru_cache


SDR_IP = "ip:192.168.1.10"
BANDWIDTH = 1000000
SAMPLERATE = 2000000
CENTER_FREQ = 2450000000
BUFFER_SIZE = int(SAMPLERATE * 0.01)  # 10ms pro sdr.rx()
SF = 5

# ═══════════════════════════════════════════════════════════════
# TDMA KONFIGURATION
# ═══════════════════════════════════════════════════════════════
GERAET = 'A'  # ← HIER ÄNDERN ('A' oder 'B')

SLOT_DAUER = 0.500
TX_DELAY = 0.020
# Zeitfenster in Sekunden, um den Burst aufzufangen
# (150ms reicht fuer >10 Pakete)
RX_WINDOW = 0.100

# ═══════════════════════════════════════════════════════════════
# KALIBRIERUNG
# ═══════════════════════════════════════════════════════════════
KALIBRIERUNGS_FAKTOR = 0.5
MIN_ABS_PEAK = None

# ═══════════════════════════════════════════════════════════════
# ESP32 ZEITSYNC
# ═══════════════════════════════════════════════════════════════
ESP32_PORT = "COM15"
ESP32_BAUD = 1000000
ESP32_SYNC = True
esp32_serial = None


# ───────────────────────────────────────────────────────────────
# ESP32 ZEITSYNC
# ───────────────────────────────────────────────────────────────
def sync_esp32(
    port: str = ESP32_PORT, baudrate: int = ESP32_BAUD
) -> Optional[serial.Serial]:
    """
    Baut die serielle Verbindung zum ESP32 auf und wartet, bis das
    erste GPS-Zeitsync-Signal empfangen wird.

    Argumente:
        port (str): Serieller Port, an dem der ESP32 angeschlossen ist.
        baudrate (int): Baudrate der seriellen Verbindung.

    Rückgabewerte:
        Optional[serial.Serial]: Die geoeffnete Serial-Verbindung, wenn
        innerhalb von 5 Sekunden ein Sync-Signal empfangen wurde, sonst
        None.
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
    except serial.SerialException as e:
        print(f"[ESP32] Fehler beim Öffnen von {port}: {e}")
        return None

    ser.reset_input_buffer()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if ser.read(1) == b'S':
            print("[ESP32] GPS Signal empfangen.")
            return ser

    print("[ESP32] TIMEOUT – kein GPS-Signal.")
    ser.close()
    return None


# ───────────────────────────────────────────────────────────────
# Input Generierung (für BER-Tests)
# ───────────────────────────────────────────────────────────────
def generiere_test_payload(
    anzahl_pakete: int, bits_pro_paket: int, seed: int = 42
) -> list:
    """
    Erzeugt deterministische Testbits fuer BER-Messungen.

    Es wird ein einzelnes Zufallspaket erzeugt und anschliessend
    mehrfach aneinandergehaengt, damit als Referenz bekannt ist,
    welche Bits jede Runde senden sollte.

    Argumente:
        anzahl_pakete (int): Anzahl der zu erzeugenden Pakete.
        bits_pro_paket (int): Anzahl Bits pro Paket.
        seed (int): Seed des Zufallsgenerators, fuer Reproduzierbarkeit.

    Rückgabewerte:
        list: Bitliste (0/1) der Laenge anzahl_pakete * bits_pro_paket.
    """
    np.random.seed(seed)
    ein_paket_bits = np.random.randint(0, 2, bits_pro_paket).tolist()
    return ein_paket_bits * anzahl_pakete


# ───────────────────────────────────────────────────────────────
# LDPC
# ───────────────────────────────────────────────────────────────
def setup_matrix(Z: int = 27) -> tuple:
    """
    Baut aus der Protograph-Basismatrix H_base die vollstaendige
    LDPC-Pruefmatrix H sowie die dazu passende Generatormatrix G auf.

    Argumente:
        Z (int): Expansionsfaktor (Groesse der zirkulanten Bloecke).

    Rückgabewerte:
        tuple: (G, H) - Generatormatrix und Pruefmatrix als
        numpy-Arrays.
    """
    H_base = np.array(
        [
            [
                0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0,
                1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            ],
            [
                22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1,
                -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            ],
            [
                6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1,
                -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
            ],
            [
                2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1,
                -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1,
            ],
            [
                23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11,
                -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1,
            ],
            [
                24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1,
                -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1,
            ],
            [
                25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1,
                0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1,
            ],
            [
                13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1,
            ],
            [
                7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1,
            ],
            [
                11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1,
            ],
            [
                25, -1, -1, -1, 16, -1, -1, -1, 11, -1, 0, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0,
            ],
            [
                3, -1, -1, -1, 0, -1, -1, -1, 25, -1, -1, -1,
                1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,
            ],
        ]
    )
    rows, cols = H_base.shape
    H_rows, H_cols = rows * Z, cols * Z
    ri, ci = [], []
    for r in range(rows):
        for c in range(cols):
            s = H_base[r, c]
            if s == -1: continue
            for i in range(Z):
                ri.append(r * Z + i)
                ci.append(c * Z + (i + s) % Z)
    H = coo_matrix(
        (np.ones(len(ri), dtype=int), (ri, ci)), shape=(H_rows, H_cols)
    ).toarray()
    H_sys = H.copy()
    m, n = H_sys.shape
    k = n - m
    pivot = 0
    for col in range(n - m, n):
        if pivot >= m: break
        if H_sys[pivot, col] == 0:
            for r in range(pivot + 1, m):
                if H_sys[r, col] == 1:
                    H_sys[[pivot, r]] = H_sys[[r, pivot]]
                    break
        if H_sys[pivot, col] == 1:
            for r in range(m):
                if r != pivot and H_sys[r, col] == 1:
                    H_sys[r] = (H_sys[r] + H_sys[pivot]) % 2
            pivot += 1
    G = np.hstack((np.eye(k, dtype=int), H_sys[:, :k].T))
    return G, H


def ldpc_encode(bits: list, G: np.ndarray) -> Optional[np.ndarray]:
    """
    Codiert Informationsbits mit der Generatormatrix G.

    Argumente:
        bits (list): Informationsbits (Laenge muss G.shape[0]
        entsprechen).
        G (np.ndarray): Generatormatrix.

    Rückgabewerte:
        Optional[np.ndarray]: Codewort (bits @ G mod 2), oder None,
        falls die Bitlaenge nicht zu G passt.
    """
    bits = np.array(bits, dtype=int)
    if len(bits) != G.shape[0]: return None
    return np.dot(bits, G) % 2


_decoder_cache = {}


def _get_decoder(H_sp: csr_matrix) -> BpDecoder:
    """
    Liefert einen (gecachten) Belief-Propagation-Decoder fuer die
    gegebene Pruefmatrix.

    Der Cache wird bei jedem neuen Schluessel geleert, da im Betrieb
    jeweils nur ein Decoder (fuer die aktuelle Matrix H) benoetigt
    wird.

    Argumente:
        H_sp (csr_matrix): Pruefmatrix in Sparse-Darstellung.

    Rückgabewerte:
        BpDecoder: Decoder-Instanz fuer H_sp.
    """
    key = id(H_sp)
    dec = _decoder_cache.get(key)
    if dec is None:
        dec = BpDecoder(H_sp, error_rate=0.1, max_iter=20, bp_method='ps')
        _decoder_cache.clear()
        _decoder_cache[key] = dec
    return dec


def ldpc_decode(
    received_symbols: list, H: np.ndarray, sf: int = SF
) -> np.ndarray:
    """
    Dekodiert empfangene LoRa-Symbole zurueck zu Informationsbits
    mittels Belief-Propagation (LDPC).

    Argumente:
        received_symbols (list): Liste von Tupeln
        (lora_wert, richtung, qpsk_wert), je eines pro Symbol.
        H (np.ndarray): Pruefmatrix.
        sf (int): Spreading Factor.

    Rückgabewerte:
        np.ndarray: Dekodierte Informationsbits (Laenge k = n - m).
    """
    received_bits = []
    lora_format = f'0{sf}b'
    for lora_wert, richtung, qpsk_wert in received_symbols:
        for b in format(lora_wert, lora_format): received_bits.append(int(b))
        received_bits.append(int(richtung))
        for b in format(qpsk_wert, '02b'): received_bits.append(int(b))

    n, m = H.shape[1], H.shape[0]
    k = n - m
    if len(received_bits) > n:
        received_bits = received_bits[:n]
    elif len(received_bits) < n:
        received_bits.extend([0] * (n - len(received_bits)))

    y = np.array(received_bits, dtype=int)
    H_sp = csr_matrix(H)
    syndrome = (H_sp @ y) % 2
    err = _get_decoder(H_sp).decode(syndrome)
    return ((y + err) % 2)[:k]


# ───────────────────────────────────────────────────────────────
# MODULATION / DEMODULATION
# ───────────────────────────────────────────────────────────────
def bits_zu_symbolen(bit_liste: list, sf: int = SF) -> list:
    """
    Wandelt eine flache Bitliste in LoRa-Symbole
    (lora_wert, richtung, qpsk_wert) um.

    Die Bitliste wird bei Bedarf mit Nullen aufgefuellt, bis ihre
    Laenge ein Vielfaches der Blockgroesse (sf + 3 Bit pro Symbol)
    ist.

    Argumente:
        bit_liste (list): Flache Liste von Bits (0/1).
        sf (int): Spreading Factor.

    Rückgabewerte:
        list: Liste von Tupeln (lora_wert, richtung, qpsk_wert).
    """
    if isinstance(bit_liste, np.ndarray): bit_liste = bit_liste.tolist()
    block_size = sf + 3
    while len(bit_liste) % block_size != 0: bit_liste.append(0)
    out = []
    for i in range(0, len(bit_liste), block_size):
        b = bit_liste[i:i + block_size]
        lora_wert = int(''.join(str(x) for x in b[0:sf]), 2)
        qpsk_wert = int(''.join(str(x) for x in b[sf + 1:block_size]), 2)
        out.append((lora_wert, b[sf], qpsk_wert))
    return out


@lru_cache(maxsize=8)
def _chirp_basis(sf: int, bw: float, fs: float) -> tuple:
    """
    Berechnet die Basis-Chirp-Signale (aufwaerts und abwaerts), die
    fuer Modulation und Demodulation gebraucht werden. Ergebnisse
    werden gecacht, da sie nur von (sf, bw, fs) abhaengen.

    Argumente:
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (float): Abtastrate in Hz.

    Rückgabewerte:
        tuple: (num_samples, base_up, base_down, window) - Anzahl
        Samples pro Symbol, Aufwaerts-Chirp, Abwaerts-Chirp und
        Hanning-Fensterfunktion.
    """
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    phase = 2 * np.pi * (0.5 * k * t**2 + (-bw / 2) * t)
    base_up = np.cos(phase) + 1j * np.sin(phase)
    return num_samples, base_up, np.conj(base_up), np.hanning(num_samples)


def generate_signal(
    symbole: list, sf: int, bw: float, fs: float
) -> np.ndarray:
    """
    Erzeugt aus einer Liste von LoRa-Symbolen das komplexe
    IQ-Sendesignal (Chirps mit Frequenz- und Phasenmodulation).

    Argumente:
        symbole (list): Liste von Tupeln
        (lora_wert, richtung, qpsk_wert).
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (float): Abtastrate in Hz.

    Rückgabewerte:
        np.ndarray: Komplexes IQ-Signal (complex64) fuer den TX.
    """
    num_samples, base, _, _ = _chirp_basis(sf, bw, fs)
    chain, aktuelle_phase = [], np.pi / 4

    for lora_wert, richtung, qpsk_wert in symbole:
        chirp = base.copy()
        chirp = np.roll(chirp, -int((lora_wert / 2**sf) * num_samples))
        chirp = chirp * np.exp(-1j * np.angle((chirp * np.conj(base))[0]))
        if richtung == 0: chirp = np.conj(chirp)
        phase_schritt = qpsk_wert * (np.pi / 2)
        aktuelle_phase = (aktuelle_phase + phase_schritt) % (2 * np.pi)
        chirp = chirp * np.exp(1j * aktuelle_phase)
        chain.append(chirp)

    return (np.concatenate(chain) * (2**14 * 0.5)).astype(np.complex64)


def signal_dechirp(
    full_signal: np.ndarray, sf: int, bw: float, fs: float
) -> list:
    """
    Zerlegt ein empfangenes IQ-Signal wieder in LoRa-Symbole
    (lora_wert, richtung, qpsk_wert).

    Pro Symboldauer wird per FFT geprueft, ob ein Aufwaerts- oder
    Abwaerts-Chirp staerker korreliert; aus der Phasendifferenz
    zwischen aufeinanderfolgenden Symbolen wird zusaetzlich der
    QPSK-Wert ermittelt.

    Argumente:
        full_signal (np.ndarray): Komplexes IQ-Empfangssignal.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (float): Abtastrate in Hz.

    Rückgabewerte:
        list: Liste von Tupeln (lora_wert, richtung, qpsk_wert), eines
        pro erkanntem Symbol.
    """
    num_samples, base_up, base_down, window = _chirp_basis(sf, bw, fs)
    decoded, letzte_phase = [], None

    for i in range(len(full_signal) // num_samples):
        chunk = full_signal[i * num_samples:(i + 1) * num_samples]

        cfft_up = np.fft.fft((chunk * base_down) * window)
        fft_up, pos_up = np.abs(cfft_up), np.argmax(np.abs(cfft_up))

        cfft_dn = np.fft.fft((chunk * base_up) * window)
        fft_dn, pos_dn = np.abs(cfft_dn), np.argmax(np.abs(cfft_dn))

        if fft_up[pos_up] > fft_dn[pos_dn]:
            d, idx, pc = 1, pos_up, cfft_up[pos_up]
        else:
            d, idx, pc = 0, pos_dn, cfft_dn[pos_dn]

        if idx > num_samples // 2: idx -= num_samples
        val = idx % (2**sf)
        if d == 0 and val != 0: val = (2**sf) - val

        aktuelle_phase = np.angle(pc)
        if letzte_phase is None:
            q = 0
        else:
            delta_phase = (aktuelle_phase - letzte_phase) % (2 * np.pi)
            if delta_phase < np.pi / 4 or delta_phase >= 7 * np.pi / 4: q = 0
            elif delta_phase < 3 * np.pi / 4: q = 1
            elif delta_phase < 5 * np.pi / 4: q = 2
            else: q = 3
        letzte_phase = aktuelle_phase
        decoded.append((val, d, q))

    return decoded


# ───────────────────────────────────────────────────────────────
# DC-KORREKTUR & KALIBRIERUNG
# ───────────────────────────────────────────────────────────────
def entferne_dc_lokal(rx_data: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Entfernt den DC-Offset lokal, blockweise pro Symboldauer, statt
    nur einmal global fuer den ganzen Puffer.

    Argumente:
        rx_data (np.ndarray): Rohes IQ-Empfangssignal.
        num_samples (int): Anzahl Samples pro Symbol (Blockgroesse).

    Rückgabewerte:
        np.ndarray: Signal ohne lokalen DC-Offset.
    """
    n, out = len(rx_data) // num_samples, rx_data.copy()
    if n > 0:
        haupt = out[:n * num_samples].reshape(n, num_samples)
        haupt -= haupt.mean(axis=1, keepdims=True)
        out[:n * num_samples] = haupt.reshape(-1)
    if n * num_samples < len(rx_data):
        out[n * num_samples:] -= np.mean(rx_data[n * num_samples:])
    return out


def kalibriere_rauschboden(
    sdr: "adi.Pluto",
    sf: int,
    bw: float,
    fs: float,
    preamble_symbole: list,
    n: int = 20,
) -> float:
    """
    Misst den Rauschboden, indem n Empfangspuffer (ohne aktives
    Sendesignal) mit dem ersten Praeambel-Symbol korreliert werden,
    und leitet daraus einen Erkennungsschwellwert ab.

    Argumente:
        sdr (adi.Pluto): Verbundenes SDR-Geraet.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (float): Abtastrate in Hz.
        preamble_symbole (list): Erwartete Praeambel-Symbole.
        n (int): Anzahl Messungen fuer die Rauschbodenschaetzung.

    Rückgabewerte:
        float: Schwellwert (Median der Peaks * 2.0) fuer die spaetere
        Praeambel-Erkennung.
    """
    print("\n" + "═" * 50)
    print(" KALIBRIERUNG – Rauschboden messen")
    print("═" * 50)
    print(" WICHTIG: Kein anderes Gerät darf senden!")
    num_samples = int((2**sf / bw) * fs)
    ref = generate_signal(preamble_symbole[:1], sf, bw, fs)
    peaks = []
    for i in range(n):
        rx = entferne_dc_lokal(sdr.rx(), num_samples)
        korr = scipy.signal.correlate(rx, ref, mode='valid', method='fft')
        peaks.append(float(np.max(np.abs(korr))))
        print(f" Messung {i + 1:2d}/{n}: Peak = {peaks[-1]:>12.0f}", end='\r')
    schwellwert = np.median(peaks) * 2.0
    print(f"\n → Schwellwert = {schwellwert:.0f}\n" + "═" * 50 + "\n")
    return schwellwert


# ───────────────────────────────────────────────────────────────
# PREAMBLE-DETEKTION & CFO
# ───────────────────────────────────────────────────────────────
def bewerte_preamble_match(
    decoded_symbols: list,
    expected_symbols: list,
    sf: int,
    cluster_tolerance: int = 3,
) -> int:
    """
    Bewertet, wie gut dekodierte Symbole zur erwarteten Praeambel
    passen, indem geprueft wird, wie viele Werte (getrennt nach
    Aufwaerts-/Abwaerts-Richtung) nahe beim jeweiligen Median liegen.

    Argumente:
        decoded_symbols (list): Dekodierte Symbole
        (lora_wert, richtung, qpsk_wert).
        expected_symbols (list): Erwartete Praeambel-Symbole.
        sf (int): Spreading Factor.
        cluster_tolerance (int): Maximal erlaubte Abweichung vom
        Median, damit ein Wert noch zum Cluster gezaehlt wird.

    Rückgabewerte:
        int: Match-Score (Anzahl passender Aufwaerts- plus
        Abwaerts-Symbole, maximal N_PREAMBLE_SYMBOLS).
    """
    max_val = 2**sf

    def signed(v: int) -> int:
        return v if v < max_val // 2 else v - max_val

    up_vals, dn_vals = [], []
    for (val, richtung, _), (_, exp_richtung, _) in zip(
        decoded_symbols, expected_symbols
    ):
        if richtung == exp_richtung:
            (up_vals if richtung == 1 else dn_vals).append(signed(val))

    def cluster_count(vals: list) -> int:
        if not vals: return 0
        arr = np.array(vals)
        return int(np.sum(np.abs(arr - np.median(arr)) <= cluster_tolerance))

    return cluster_count(up_vals) + cluster_count(dn_vals)


@lru_cache(maxsize=4)
def _preamble_refs(sf: int, bw: float, fs: float) -> tuple:
    """
    Erzeugt die Referenzsignale eines einzelnen Aufwaerts- bzw.
    Abwaerts-Praeambelsymbols, fuer die Kreuzkorrelation bei der
    Paketerkennung. Ergebnisse werden gecacht.

    Argumente:
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (float): Abtastrate in Hz.

    Rückgabewerte:
        tuple: (ref_up, ref_dn) - Referenzsignale fuer je ein
        Aufwaerts- und ein Abwaerts-Symbol.
    """
    return (
        generate_signal(((0, 1, 0),), sf, bw, fs),
        generate_signal(((0, 0, 0),), sf, bw, fs),
    )


def finde_paket_start_bruteforce(
    rx_data: np.ndarray,
    sf: int,
    bw: float,
    fs: float,
    preamble_symbols: list,
    top_n: int = 10,
    value_tolerance: int = 2,
    verbose: bool = False,
) -> list:
    """
    Sucht per Brute-Force-Korrelation nach moeglichen Paketanfaengen
    im Empfangspuffer und bewertet die besten Kandidaten anhand des
    Praeambel-Match-Scores.

    Argumente:
        rx_data (np.ndarray): Roher Empfangspuffer.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (float): Abtastrate in Hz.
        preamble_symbols (list): Erwartete Praeambel-Symbole.
        top_n (int): Anzahl der zu bewertenden Korrelations-Kandidaten.
        value_tolerance (int): Cluster-Toleranz fuer
        bewerte_preamble_match.
        verbose (bool): Aktuell ungenutzt, fuer spaetere Diagnose
        reserviert.

    Rückgabewerte:
        list: Liste von Tupeln (offset, score, match, syms), nach
        match absteigend sortiert.
    """
    num_samples = int((2**sf / bw) * fs)
    rx_clean = entferne_dc_lokal(rx_data, num_samples)
    pre_len = len(preamble_symbols) * num_samples
    if len(rx_clean) - pre_len < 0: return []

    ref_up, ref_dn = _preamble_refs(sf, bw, fs)
    corr_up = np.abs(scipy.signal.correlate(
        rx_clean, ref_up, mode='valid', method='fft'
    ))
    corr_dn = np.abs(scipy.signal.correlate(
        rx_clean, ref_dn, mode='valid', method='fft'
    ))

    M = len(rx_clean) - pre_len + 1
    score = np.zeros(M)
    for i, (_, richtung, _) in enumerate(preamble_symbols):
        quelle = corr_up if richtung == 1 else corr_dn
        score += quelle[i * num_samples: i * num_samples + M]

    kandidaten_idx = []
    for idx in np.argsort(score)[::-1]:
        idx = int(idx)
        if all(abs(idx - k) > num_samples // 2 for k in kandidaten_idx):
            kandidaten_idx.append(idx)
        if len(kandidaten_idx) >= top_n: break

    ergebnisse = []
    for offset in kandidaten_idx:
        chunk = rx_clean[offset: offset + pre_len]
        syms = signal_dechirp(chunk, sf, bw, fs)
        match = bewerte_preamble_match(
            syms, preamble_symbols, sf, value_tolerance
        )
        ergebnisse.append((offset, float(score[offset]), match, syms))

    ergebnisse.sort(key=lambda r: r[2], reverse=True)
    return ergebnisse


def measure_fine_cfo(
    pre_chunk: np.ndarray, sf: int, bw: float, fs: float
) -> float:
    """
    Schaetzt den feinen Traegerfrequenzversatz (CFO) innerhalb eines
    Sample anhand der Phasendrift ueber die ersten 8
    Praeambel-Symbole.

    Argumente:
        pre_chunk (np.ndarray): IQ-Ausschnitt der Praeambel.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (float): Abtastrate in Hz.

    Rückgabewerte:
        float: Geschaetzter feiner CFO in Hz.
    """
    num_samples, _, base_down, window = _chirp_basis(sf, bw, fs)
    phases = []
    for i in range(8):
        dc = pre_chunk[i * num_samples:(i + 1) * num_samples]
        dc = dc * base_down * window
        cfft = np.fft.fft(dc)
        phases.append(np.angle(cfft[np.argmax(np.abs(cfft))]))

    deltas = [
        ((phases[i] - phases[i - 1]) + np.pi) % (2 * np.pi) - np.pi
        for i in range(1, len(phases))
    ]
    mean_drift_per_symbol = sum(deltas) / len(deltas)
    return mean_drift_per_symbol / (2 * np.pi) / (num_samples / fs)


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────

g_matrix, h_matrix = setup_matrix()
k_bits = g_matrix.shape[0]

try:
    sdr = adi.Pluto(SDR_IP)
    sdr.sample_rate = SAMPLERATE
    sdr.rx_lo = CENTER_FREQ
    sdr.rx_rf_bandwidth = BANDWIDTH
    sdr.rx_buffer_size = BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 20
    sdr.tx_lo = CENTER_FREQ
    sdr.tx_rf_bandwidth = BANDWIDTH
    sdr.tx_cyclic_buffer = False
    sdr.tx_hardwaregain_chan0 = 0
    for _ in range(5): _ = sdr.rx()
    print(f"SDR verbunden! Bereit auf {CENTER_FREQ / 1e6} MHz.")
except Exception as e:
    print(f"Fehler bei der SDR-Verbindung: {e}")
    exit(1)

if ESP32_SYNC:
    esp32_serial = sync_esp32(ESP32_PORT, ESP32_BAUD)
    if esp32_serial is None: ESP32_SYNC = False

N_PREAMBLE_SYMBOLS = 16
preamble_symbole = [(0, 1, 0)] * 8 + [(0, 0, 0)] * 8

print(f"Gerät: {GERAET}  |  ESP32-Sync: {'AKTIV' if ESP32_SYNC else 'AUS'}")
MIN_ABS_PEAK = kalibriere_rauschboden(
    sdr, SF, BANDWIDTH, SAMPLERATE, preamble_symbole, n=20
)

while True:
    print("\n" + "=" * 50)
    state = input(
        "Was möchtest du tun?\n"
        " [t] TDMA-Test (Burst-Modus mit optimiertem Timing)\n"
        " [q] Beenden\n"
        "Deine Wahl: "
    ).strip().lower()

    if state == "q":
        if esp32_serial and esp32_serial.is_open: esp32_serial.close()
        print("Tschüss!")
        break

    elif state == "t":
        try:
            anzahl_input = input(
                "Wie viele Pakete sollen INSGESAMT getestet werden? "
                "(Standard: 100): "
            ).strip()
            anzahl_pakete = int(anzahl_input) if anzahl_input else 100

            burst_input = input(
                "Wie viele Pakete PRO SLOT (Burst-Size)? (Standard: 3): "
            ).strip()
            burst_size = int(burst_input) if burst_input else 3
        except ValueError:
            print("[Fehler] Bitte gültige Zahlen eingeben.")
            continue

        file_name = input(
            "Gib den Namen für das TDMA-Test-Logfile ein (ohne .txt): "
        ).strip()
        if not file_name: file_name = "tdma_test_log"

        expected_total_payload = generiere_test_payload(
            anzahl_pakete, k_bits, seed=42
        )
        tx_bits = expected_total_payload[:k_bits]
        codeword = ldpc_encode(tx_bits, g_matrix)
        symbole_tx = preamble_symbole + bits_zu_symbolen(codeword, SF)
        single_tx_signal = generate_signal(
            symbole_tx, SF, BANDWIDTH, SAMPLERATE
        )

        # Burst generieren
        tx_signal = np.concatenate([single_tx_signal] * burst_size)
        anzahl_runden = int(np.ceil(anzahl_pakete / burst_size))

        print(f"\n--- TDMA TEST MODUS GESTARTET (Gerät {GERAET}) ---")
        print(
            f"Sende-Signal: {burst_size} Pakete pro Slot "
            f"({len(tx_signal)} Samples)."
        )
        print(
            f"Ziel: {anzahl_pakete} Pakete in {anzahl_runden} Slots "
            f"austauschen...\n"
        )

        alle_decodierten_bits = []
        all_rx_buffers = []

        # ─── Tabellen für die spätere LaTeX-Auswertung ───
        # pakete_log:  1 Zeile pro ERFOLGREICH decodiertem Paket
        # runden_log:  1 Zeile pro RUNDE (auch wenn 0 Pakete gefunden wurden!)
        pakete_log = []
        runden_log = []

        gesamt_empfangene_pakete = 0
        gesamt_bit_fehler = 0
        gesamt_korrigiert = 0
        txt_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            if esp32_serial and esp32_serial.is_open:
                esp32_serial.reset_input_buffer()

            for runde in range(1, anzahl_runden + 1):
                # 1. GPS SYNC
                if esp32_serial and esp32_serial.is_open:
                    esp32_serial.reset_input_buffer()
                    while True:
                        if (
                            esp32_serial.in_waiting > 0
                            and esp32_serial.read(1) == b'S'
                        ):
                            t_start = time.time()
                            break
                        try: _ = sdr.rx()
                        except: pass
                else:
                    while time.time() % 1.0 > 0.005: _ = sdr.rx()
                    while time.time() % 1.0 < 0.995: _ = sdr.rx()
                    t_start = time.time()

                rx_buffers = []

                # 2. VERSCHRÄNKTER TX/RX ABLAUF
                if GERAET == 'A':
                    # A sendet in Slot 1
                    while time.time() - t_start < TX_DELAY: pass
                    sdr.tx(tx_signal)

                    # A wartet auf Slot 2
                    while time.time() - t_start < SLOT_DAUER - 0.010:
                        _ = sdr.rx()
                    while time.time() - t_start < SLOT_DAUER: pass

                    # A empfängt Burst von B
                    rx_deadline = t_start + SLOT_DAUER + TX_DELAY + RX_WINDOW
                    while time.time() < rx_deadline:
                        rx_buffers.append(sdr.rx())

                elif GERAET == 'B':
                    # B empfängt Burst von A in Slot 1
                    rx_deadline = t_start + TX_DELAY + RX_WINDOW
                    while time.time() < rx_deadline:
                        rx_buffers.append(sdr.rx())

                # 3. DECODIEREN (Nutzt die verbleibende Slot-Zeit!)
                t_dec = time.time()

                # Zwischenspeicher NUR für die Pakete dieser einen Runde
                # (wird am Rundenende immer in runden_log geschrieben,
                #  unabhängig davon ob 0, 1 oder mehrere Pakete gefunden
                #  wurden)
                runde_pakete = []

                # Bester Präambel-Match-Score dieser Runde
                # (0 bis N_PREAMBLE_SYMBOLS=16).
                # 16 = alle Präambel-Symbole exakt erkannt, 0 = überhaupt
                # kein plausibler Kandidat gefunden (kein Signal / kein
                # Match).
                runde_praeambel_score = 0

                if rx_buffers:
                    full_rx = np.concatenate(rx_buffers)
                    all_rx_buffers.append(full_rx)

                    search_offset = 0

                    while True:
                        kandidaten = finde_paket_start_bruteforce(
                            full_rx[search_offset:], SF, BANDWIDTH,
                            SAMPLERATE, preamble_symbole, top_n=5,
                            value_tolerance=3, verbose=False,
                        )
                        if kandidaten:
                            runde_praeambel_score = max(
                                runde_praeambel_score, kandidaten[0][2]
                            )
                        if not kandidaten or kandidaten[0][2] < 10:
                            break

                        p_start = search_offset + kandidaten[0][0]
                        num_s = int((2**SF / BANDWIDTH) * SAMPLERATE)
                        pre_len = N_PREAMBLE_SYMBOLS * num_s
                        pay_len = int(
                            np.ceil(h_matrix.shape[1] / (SF + 3))
                        ) * num_s

                        if p_start + pre_len + pay_len > len(full_rx):
                            break

                        pre_chunk = full_rx[p_start:p_start + pre_len]
                        pre_syms = signal_dechirp(
                            pre_chunk, SF, BANDWIDTH, SAMPLERATE
                        )

                        up_b, dn_b = [], []
                        for lv, dr, _ in pre_syms:
                            sb = lv if lv < (2**SF) // 2 else lv - (2**SF)
                            (up_b if dr == 1 else dn_b).append(sb)

                        if up_b and dn_b:
                            cfo_bin = (
                                sum(up_b) / len(up_b) - sum(dn_b) / len(dn_b)
                            ) / 2.0
                            t_bin = (
                                sum(up_b) / len(up_b) + sum(dn_b) / len(dn_b)
                            ) / 2.0
                        else:
                            cfo_bin, t_bin = 0.0, 0.0

                        pay_start = p_start + pre_len
                        pay_start += int(t_bin * (SAMPLERATE / BANDWIDTH))
                        pay_end = pay_start + pay_len

                        if pay_end > len(full_rx):
                            break

                        payload = full_rx[pay_start:pay_end]
                        try:
                            fine_cfo_hz = measure_fine_cfo(
                                pre_chunk, SF, BANDWIDTH, SAMPLERATE
                            )
                        except NameError:
                            fine_cfo_hz = 0.0

                        total_cfo_hz = (
                            cfo_bin * (SAMPLERATE / num_s)
                        ) + fine_cfo_hz
                        zeitachse = np.arange(len(payload)) / SAMPLERATE
                        payload_corrected = payload * np.exp(
                            -1j * 2 * np.pi * total_cfo_hz * zeitachse
                        )

                        pay_syms_rx = signal_dechirp(
                            payload_corrected, SF, BANDWIDTH, SAMPLERATE
                        )
                        decoded = ldpc_decode(pay_syms_rx, h_matrix, SF)

                        gesamt_empfangene_pakete += 1
                        alle_decodierten_bits.extend(decoded.tolist())

                        raw_rx_bits = []
                        for lv, dr, qv in pay_syms_rx:
                            raw_rx_bits.extend(
                                [int(b) for b in format(lv, f'0{SF}b')]
                                + [int(dr)]
                                + [int(b) for b in format(qv, '02b')]
                            )

                        ideal_codeword = ldpc_encode(decoded, g_matrix)
                        if ideal_codeword is not None:
                            korrigierte_fehler = sum(
                                1
                                for a, b in zip(
                                    raw_rx_bits[:h_matrix.shape[1]],
                                    ideal_codeword,
                                )
                                if a != b
                            )
                        else:
                            korrigierte_fehler = 0
                        gesamt_korrigiert += korrigierte_fehler

                        chunk_start = (gesamt_empfangene_pakete - 1) * k_bits
                        chunk_end = gesamt_empfangene_pakete * k_bits
                        expected_chunk = expected_total_payload[
                            chunk_start:chunk_end
                        ]
                        bit_fehler = sum(
                            1
                            for a, b in zip(decoded, expected_chunk)
                            if a != b
                        )
                        gesamt_bit_fehler += bit_fehler

                        paket_eintrag = {
                            "Runde": runde,
                            "Paket_global": gesamt_empfangene_pakete,
                            "Paket_in_Runde": len(runde_pakete) + 1,
                            "Praeambel_Score": kandidaten[0][2],
                            "Drift_Hz": round(total_cfo_hz),
                            "LDPC_korrigiert": korrigierte_fehler,
                            "Bitfehler": bit_fehler,
                        }
                        pakete_log.append(paket_eintrag)
                        runde_pakete.append(paket_eintrag)

                        print(
                            f"[{GERAET}] Runde {runde}: "
                            f"✓ Paket {len(runde_pakete)}/{burst_size} "
                            f"decodiert (Präambel: {kandidaten[0][2]}/"
                            f"{N_PREAMBLE_SYMBOLS} | "
                            f"Drift: {total_cfo_hz:.0f}Hz | "
                            f"LDPC: {korrigierte_fehler} | "
                            f"BER: {bit_fehler})"
                        )
                        search_offset = pay_end

                # ─── Rundenzusammenfassung: IMMER genau 1 Zeile pro
                # Runde ───
                erkannt = len(runde_pakete)
                if erkannt == 0:
                    print(
                        f"[{GERAET}] Runde {runde}: Kein Paket decodiert "
                        f"(bester Präambel-Score: "
                        f"{runde_praeambel_score}/{N_PREAMBLE_SYMBOLS})."
                    )

                if erkannt:
                    drift_mittel_hz = round(
                        sum(p["Drift_Hz"] for p in runde_pakete) / erkannt
                    )
                else:
                    drift_mittel_hz = ""

                runden_log.append({
                    "Runde": runde,
                    "Pakete_erwartet": burst_size,
                    "Pakete_erkannt": erkannt,
                    "Praeambel_Score": runde_praeambel_score,
                    "Drift_Mittel_Hz": drift_mittel_hz,
                    "LDPC_Summe": sum(
                        p["LDPC_korrigiert"] for p in runde_pakete
                    ),
                    "Bitfehler_Summe": sum(
                        p["Bitfehler"] for p in runde_pakete
                    ),
                })

                # 4. GERÄT B MUSS NACH DEM DECODIEREN NOCH SENDEN
                if GERAET == 'B':
                    grenze = SLOT_DAUER + TX_DELAY - 0.010
                    while time.time() - t_start < grenze:
                        _ = sdr.rx()
                    while time.time() - t_start < SLOT_DAUER + TX_DELAY:
                        pass
                    sdr.tx(tx_signal)

            print(f"\n[{GERAET}] TDMA-Test beendet. Speichere Daten...")

        except KeyboardInterrupt:
            print(f"\n[{GERAET}] Abbruch. Speichere bisherige Daten...")

        if all_rx_buffers:
            dateiname = f"tdma_iq_dump_{time.strftime('%Y%m%d_%H%M%S')}.npy"
            np.save(dateiname, np.concatenate(all_rx_buffers))

        total_bits = len(alle_decodierten_bits)
        ber_prozent = (
            (gesamt_bit_fehler / total_bits) * 100 if total_bits > 0 else 0
        )
        # runde = letzte tatsächlich gelaufene Runde
        erwartete_pakete_gesamt = runde * burst_size
        if erwartete_pakete_gesamt > 0:
            paketverlust_prozent = (
                (erwartete_pakete_gesamt - gesamt_empfangene_pakete)
                / erwartete_pakete_gesamt
                * 100
            )
        else:
            paketverlust_prozent = 0

        # ─── Tabelle 1: eine Zeile pro RUNDE ───────────────────────────
        runden_csv = f"{file_name}_runden.csv"
        with open(runden_csv, "w", newline="", encoding="utf-8") as f:
            spalten = [
                "Runde", "Pakete_erwartet", "Pakete_erkannt",
                "Praeambel_Score", "Drift_Mittel_Hz", "LDPC_Summe",
                "Bitfehler_Summe",
            ]
            writer = csv.DictWriter(f, fieldnames=spalten)
            writer.writeheader()
            writer.writerows(runden_log)

        # ─── Tabelle 2: eine Zeile pro erfolgreich decodiertem PAKET ───
        pakete_csv = f"{file_name}_pakete.csv"
        with open(pakete_csv, "w", newline="", encoding="utf-8") as f:
            spalten = [
                "Runde", "Paket_global", "Paket_in_Runde",
                "Praeambel_Score", "Drift_Hz", "LDPC_korrigiert",
                "Bitfehler",
            ]
            writer = csv.DictWriter(f, fieldnames=spalten)
            writer.writeheader()
            writer.writerows(pakete_log)

        # ─── Kurze, lesbare Zusammenfassung (wie bisher, als .txt) ─────
        with open(f"{file_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"SDR TDMA-TEST | BER: {ber_prozent:.4f}%\n")
            f.write(
                f"Runden: {runde}/{anzahl_runden} | "
                f"Pakete erwartet: {erwartete_pakete_gesamt} | "
                f"Pakete empfangen: {gesamt_empfangene_pakete} "
                f"({paketverlust_prozent:.1f}% Verlust) | "
                f"LDPC korrigiert: {gesamt_korrigiert} Bit\n"
            )
            n_kein_paket = sum(
                1 for r in runden_log if r["Pakete_erkannt"] == 0
            )
            n_kein_paket_mit_score = sum(
                1
                for r in runden_log
                if r["Pakete_erkannt"] == 0 and r["Praeambel_Score"] > 0
            )
            f.write(
                f"Runden ganz ohne Paket: {n_kein_paket} "
                f"(davon mit Präambel-Score > 0: {n_kein_paket_mit_score})\n"
            )

        print(f"\n💾 Rundentabelle:  {runden_csv}")
        print(f"💾 Pakettabelle:   {pakete_csv}")
        print(f"💾 Zusammenfassung: {file_name}_summary.txt")
        print(
            f"📊 BER: {ber_prozent:.4f}% | "
            f"Paketverlust: {paketverlust_prozent:.1f}% "
            f"({gesamt_empfangene_pakete}/{erwartete_pakete_gesamt})"
        )