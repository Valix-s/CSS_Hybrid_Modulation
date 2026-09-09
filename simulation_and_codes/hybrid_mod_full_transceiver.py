import numpy as np
import adi
import scipy.signal
from scipy.sparse import coo_matrix, csr_matrix
from ldpc.bp_decoder import BpDecoder
import time
import serial
from functools import lru_cache


SDR_IP      = "ip:192.168.1.10"
BANDWIDTH   = 1000000
SAMPLERATE  = 2000000
CENTER_FREQ = 2450000000
BUFFER_SIZE = int(SAMPLERATE * 0.01)   # 10ms pro sdr.rx()
SF = 5

# ═══════════════════════════════════════════════════════════════
# TDMA KONFIGURATION
# ═══════════════════════════════════════════════════════════════
GERAET = 'B'          # ← HIER ÄNDERN

SLOT_DAUER   = 0.500
PUFFER_START = 0.005
TX_DAUER     = 0.200
PUFFER_ENDE  = 0.005

# ═══════════════════════════════════════════════════════════════
# KALIBRIERUNG
# ═══════════════════════════════════════════════════════════════
KALIBRIERUNGS_FAKTOR = 0.5
MIN_ABS_PEAK         = None

# ═══════════════════════════════════════════════════════════════
# ESP32 ZEITSYNC
# ═══════════════════════════════════════════════════════════════
ESP32_PORT   = "COM11"
ESP32_BAUD   = 1000000
ESP32_SYNC   = True
esp32_serial = None


# ───────────────────────────────────────────────────────────────
# ESP32 ZEITSYNC
# ───────────────────────────────────────────────────────────────

def sync_esp32(port: str = ESP32_PORT, baudrate: int = ESP32_BAUD):
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
# LDPC
# ───────────────────────────────────────────────────────────────

def setup_matrix(Z: int = 27):
    H_base = np.array([
        [0,-1,-1,-1,0,0,-1,-1,0,-1,-1,0,1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [22,0,-1,-1,17,-1,0,0,12,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [6,-1,0,-1,10,-1,-1,-1,24,-1,0,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1],
        [2,-1,-1,0,20,-1,-1,-1,25,0,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1],
        [23,-1,-1,-1,3,-1,-1,-1,0,-1,9,11,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1],
        [24,-1,23,1,17,-1,3,-1,10,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1],
        [25,-1,-1,-1,8,-1,-1,-1,7,18,-1,-1,0,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1],
        [13,24,-1,-1,0,-1,8,-1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1,-1],
        [7,20,-1,16,22,10,-1,-1,23,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1],
        [11,-1,-1,-1,19,-1,-1,-1,13,-1,3,17,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1],
        [25,-1,-1,-1,16,-1,-1,-1,11,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0],
        [3,-1,-1,-1,0,-1,-1,-1,25,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0],
    ])
    rows, cols = H_base.shape
    H_rows, H_cols = rows * Z, cols * Z
    ri, ci = [], []
    for r in range(rows):
        for c in range(cols):
            s = H_base[r, c]
            if s == -1:
                continue
            for i in range(Z):
                ri.append(r * Z + i)
                ci.append(c * Z + (i + s) % Z)
    H = coo_matrix((np.ones(len(ri), dtype=int), (ri, ci)),
                   shape=(H_rows, H_cols)).toarray()
    H_sys = H.copy()
    m, n = H_sys.shape
    k = n - m
    pivot = 0
    for col in range(n - m, n):
        if pivot >= m:
            break
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


def ldpc_encode(bits, G):
    bits = np.array(bits, dtype=int)
    if len(bits) != G.shape[0]:
        print(f"ACHTUNG: Falsche Bitlaenge! Habe {len(bits)}, brauche {G.shape[0]}.")
        return None
    return np.dot(bits, G) % 2


# Der BpDecoder-Aufbau (Tanner-Graph etc.) ist teuer und H ändert sich nie
# während der Laufzeit -> einmal cachen statt bei jedem ldpc_decode neu bauen.
_decoder_cache = {}

def _get_decoder(H_sp):
    key = id(H_sp)
    dec = _decoder_cache.get(key)
    if dec is None:
        dec = BpDecoder(H_sp, error_rate=0.1, max_iter=20, bp_method='ps')
        _decoder_cache.clear()  # nur den aktuellen H behalten
        _decoder_cache[key] = dec
    return dec


def ldpc_decode(received_symbols, H, sf=SF):
    received_bits = []
    lora_format = f'0{sf}b'

    for lora_wert, richtung, qpsk_wert in received_symbols:
        for b in format(lora_wert, lora_format):
            received_bits.append(int(b))
        received_bits.append(int(richtung))
        for b in format(qpsk_wert, '02b'):
            received_bits.append(int(b))

    n, m = H.shape[1], H.shape[0]
    k = n - m
    if len(received_bits) > n:
        received_bits = received_bits[:n]
    elif len(received_bits) < n:
        received_bits.extend([0] * (n - len(received_bits)))

    y = np.array(received_bits, dtype=int)
    H_sp = csr_matrix(H)
    syndrome = (H_sp @ y) % 2
    decoder = _get_decoder(H_sp)
    err = decoder.decode(syndrome)
    return ((y + err) % 2)[:k]


# ───────────────────────────────────────────────────────────────
# MODULATION / DEMODULATION
# ───────────────────────────────────────────────────────────────

def bits_zu_symbolen(bit_liste, sf=SF):
    if isinstance(bit_liste, np.ndarray):
        bit_liste = bit_liste.tolist()

    block_size = sf + 3

    while len(bit_liste) % block_size != 0:
        bit_liste.append(0)

    out = []
    for i in range(0, len(bit_liste), block_size):
        b = bit_liste[i:i+block_size]
        out.append((int(''.join(str(x) for x in b[0:sf]), 2),
                    b[sf],
                    int(''.join(str(x) for x in b[sf+1:block_size]), 2)))
    return out


# Basis-Chirp/Fenster hängen nur von (sf, bw, fs) ab, die sich im Skript
# nie ändern -> einmalig berechnen und cachen statt bei jedem Aufruf neu.
@lru_cache(maxsize=8)
def _chirp_basis(sf, bw, fs):
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    phase = 2 * np.pi * (0.5 * k * t**2 + (-bw/2) * t)
    base_up = np.cos(phase) + 1j * np.sin(phase)
    base_down = np.conj(base_up)
    window = np.hanning(num_samples)
    return num_samples, base_up, base_down, window


def generate_signal(symbole, sf, bw, fs):
    num_samples, base, _, _ = _chirp_basis(sf, bw, fs)

    chain = []
    aktuelle_phase = np.pi / 4

    for lora_wert, richtung, qpsk_wert in symbole:
        chirp = base.copy()
        shift = int((lora_wert / 2**sf) * num_samples)
        chirp = np.roll(chirp, -shift)

        chirp = chirp * np.exp(-1j * np.angle((chirp * np.conj(base))[0]))

        if richtung == 0:
            chirp = np.conj(chirp)

        phasen_sprung = qpsk_wert * (np.pi / 2)
        aktuelle_phase = (aktuelle_phase + phasen_sprung) % (2 * np.pi)

        chirp = chirp * np.exp(1j * aktuelle_phase)
        chain.append(chirp)

    return (np.concatenate(chain) * (2**14 * 0.5)).astype(np.complex64)


def signal_dechirp(full_signal, sf, bw, fs):
    num_samples, base_up, base_down, window = _chirp_basis(sf, bw, fs)

    decoded = []
    letzte_phase = None

    for i in range(len(full_signal) // num_samples):
        chunk = full_signal[i*num_samples:(i+1)*num_samples]

        dc_up   = (chunk * base_down) * window
        cfft_up = np.fft.fft(dc_up)
        fft_up  = np.abs(cfft_up)
        pos_up  = np.argmax(fft_up)

        dc_dn   = (chunk * base_up) * window
        cfft_dn = np.fft.fft(dc_dn)
        fft_dn  = np.abs(cfft_dn)
        pos_dn  = np.argmax(fft_dn)

        if fft_up[pos_up] > fft_dn[pos_dn]:
            d, idx, pc = 1, pos_up, cfft_up[pos_up]
        else:
            d, idx, pc = 0, pos_dn, cfft_dn[pos_dn]

        if idx > num_samples // 2:
            idx -= num_samples
        val = idx % (2**sf)
        if d == 0 and val != 0:
            val = (2**sf) - val

        aktuelle_phase = np.angle(pc)

        if letzte_phase is None:
            q = 0
        else:
            delta_phase = (aktuelle_phase - letzte_phase) % (2 * np.pi)

            if delta_phase < np.pi/4 or delta_phase >= 7*np.pi/4:
                q = 0
            elif delta_phase >= np.pi/4 and delta_phase < 3*np.pi/4:
                q = 1
            elif delta_phase >= 3*np.pi/4 and delta_phase < 5*np.pi/4:
                q = 2
            else:
                q = 3

        letzte_phase = aktuelle_phase
        decoded.append((val, d, q))

    return decoded


# ───────────────────────────────────────────────────────────────
# DC-KORREKTUR & KALIBRIERUNG
# ───────────────────────────────────────────────────────────────

def entferne_dc_lokal(rx_data, num_samples):
    """DC pro Chirp-Fenster entfernen (vektorisiert statt Python-Schleife)."""
    n = len(rx_data) // num_samples
    out = rx_data.copy()

    if n > 0:
        haupt = out[:n * num_samples].reshape(n, num_samples)
        haupt -= haupt.mean(axis=1, keepdims=True)
        out[:n * num_samples] = haupt.reshape(-1)

    rest = n * num_samples
    if rest < len(rx_data):
        out[rest:] -= np.mean(rx_data[rest:])
    return out


def kalibriere_rauschboden(sdr, sf, bw, fs, preamble_symbole, n=20):
    print(f"Kalibrierung: {n} Messungen (kein anderes Gerät darf senden)...")
    num_samples = int((2**sf / bw) * fs)
    ref = generate_signal(preamble_symbole[:1], sf, bw, fs)
    peaks = []
    for _ in range(n):
        rx = entferne_dc_lokal(sdr.rx(), num_samples)
        korr = scipy.signal.correlate(rx, ref, mode='valid', method='fft')
        peaks.append(float(np.max(np.abs(korr))))
    schwellwert = np.median(peaks) * 2.0
    print(f"Kalibrierung fertig: Min={min(peaks):.0f} Max={max(peaks):.0f} "
          f"MIN_ABS_PEAK={schwellwert:.0f}")
    return schwellwert


# ───────────────────────────────────────────────────────────────
# PREAMBLE-DETEKTION
# ───────────────────────────────────────────────────────────────

def bewerte_preamble_match(decoded_symbols, expected_symbols, sf, cluster_tolerance=3):
    max_val = 2**sf

    def signed(v):
        return v if v < max_val // 2 else v - max_val

    up_vals, dn_vals = [], []
    for (val, richtung, _q), (_, exp_richtung, _) in zip(decoded_symbols, expected_symbols):
        if richtung != exp_richtung:
            continue
        sv = signed(val)
        (up_vals if richtung == 1 else dn_vals).append(sv)

    def cluster_count(vals):
        if not vals:
            return 0
        arr = np.array(vals)
        med = np.median(arr)
        return int(np.sum(np.abs(arr - med) <= cluster_tolerance))

    return cluster_count(up_vals) + cluster_count(dn_vals)


# Referenz-Chirps für die Präambel-Suche sind immer identisch -> cachen.
@lru_cache(maxsize=4)
def _preamble_refs(sf, bw, fs):
    ref_up = generate_signal(((0, 1, 0),), sf, bw, fs)
    ref_dn = generate_signal(((0, 0, 0),), sf, bw, fs)
    return ref_up, ref_dn


def finde_paket_start_bruteforce(rx_data, sf, bw, fs, preamble_symbols,
                                   top_n=10, value_tolerance=2, verbose=False):
    num_samples = int((2**sf / bw) * fs)
    rx_clean    = entferne_dc_lokal(rx_data, num_samples)
    n_sym       = len(preamble_symbols)
    pre_len     = n_sym * num_samples
    max_offset  = len(rx_clean) - pre_len
    if max_offset < 0:
        return []

    ref_up, ref_dn = _preamble_refs(sf, bw, fs)

    corr_up = np.abs(scipy.signal.correlate(rx_clean, ref_up, mode='valid', method='fft'))
    corr_dn = np.abs(scipy.signal.correlate(rx_clean, ref_dn, mode='valid', method='fft'))

    M = max_offset + 1
    score = np.zeros(M)
    for i, (_, richtung, _) in enumerate(preamble_symbols):
        quelle = corr_up if richtung == 1 else corr_dn
        start  = i * num_samples
        score += quelle[start: start + M]

    # Nicht-Maximum-Unterdrückung
    order = np.argsort(score)[::-1]
    kandidaten_idx = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - k) > num_samples // 2 for k in kandidaten_idx):
            kandidaten_idx.append(idx)
        if len(kandidaten_idx) >= top_n:
            break

    ergebnisse = []
    for offset in kandidaten_idx:
        chunk = rx_clean[offset: offset + pre_len]
        syms  = signal_dechirp(chunk, sf, bw, fs)
        match = bewerte_preamble_match(syms, preamble_symbols, sf, value_tolerance)
        ergebnisse.append((offset, float(score[offset]), match, syms))

    ergebnisse.sort(key=lambda r: r[2], reverse=True)

    if verbose:
        print(f"[BRUTE-FORCE] {M} Positionen, {len(kandidaten_idx)} Kandidaten getestet")

    return ergebnisse


def finde_paket_start(rx_data, sf, bw, fs, preamble_symbols):
    kandidaten = finde_paket_start_bruteforce(
        rx_data, sf, bw, fs, preamble_symbols,
        top_n=10, value_tolerance=2, verbose=False
    )

    if len(kandidaten) == 0:
        return -1

    offset, score, match, syms = kandidaten[0]
    print(f"[BRUTE] Match={match}   Score={score:.0f}")
    return offset


def measure_fine_cfo(pre_chunk, sf, bw, fs):
    """Misst den exakten Frequenzfehler in Hz anhand der Phasenrotation der Präambel."""
    num_samples, _, base_down, window = _chirp_basis(sf, bw, fs)

    phases = []
    for i in range(8):
        chunk = pre_chunk[i*num_samples:(i+1)*num_samples]
        dc = chunk * base_down * window
        cfft = np.fft.fft(dc)
        pos = np.argmax(np.abs(cfft))
        phases.append(np.angle(cfft[pos]))

    deltas = []
    for i in range(1, len(phases)):
        diff = (phases[i] - phases[i-1])
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        deltas.append(diff)

    mean_drift_per_symbol = sum(deltas) / len(deltas)
    symbol_time = num_samples / fs
    fine_cfo_hz = mean_drift_per_symbol / (2 * np.pi) / symbol_time
    return fine_cfo_hz


# ───────────────────────────────────────────────────────────────
# TDMA HILFSFUNKTIONEN
# ───────────────────────────────────────────────────────────────

def get_slot_pos():
    return time.time() % 1.0


def warte_auf_tx_slot(geraet):
    if geraet == 'A':
        while get_slot_pos() < 0.995:
            time.sleep(0.001)
        while get_slot_pos() > 0.010:
            pass
    else:
        while get_slot_pos() < 0.495:
            time.sleep(0.001)
        while get_slot_pos() < 0.500:
            pass
    return time.time()


def warte_auf_rx_slot(geraet):
    geraet_rx = 'B' if geraet == 'A' else 'A'
    return warte_auf_tx_slot(geraet_rx)


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────

g_matrix, h_matrix = setup_matrix()
k_bits = g_matrix.shape[0]

try:
    sdr = adi.Pluto(SDR_IP)
    sdr.sample_rate              = SAMPLERATE
    sdr.rx_lo                    = CENTER_FREQ
    sdr.rx_rf_bandwidth          = BANDWIDTH
    sdr.rx_buffer_size           = BUFFER_SIZE
    sdr.gain_control_mode_chan0  = "manual"
    sdr.rx_hardwaregain_chan0    = 20
    sdr.tx_lo                    = CENTER_FREQ
    sdr.tx_rf_bandwidth          = BANDWIDTH
    sdr.tx_cyclic_buffer         = False
    sdr.tx_hardwaregain_chan0    = 0
    for _ in range(5):
        _ = sdr.rx()
    print(f"SDR verbunden auf {CENTER_FREQ/1e6} MHz.")
except Exception as e:
    print(f"Fehler bei der SDR-Verbindung: {e}")
    exit(1)

if ESP32_SYNC:
    esp32_serial = sync_esp32(ESP32_PORT, ESP32_BAUD)
    if esp32_serial is None:
        print("[ESP32] Sync fehlgeschlagen – fahre ohne ESP32 fort.")
        ESP32_SYNC = False

N_PREAMBLE_SYMBOLS = 16
preamble_symbole   = [(0, 1, 0)] * 8 + [(0, 0, 0)] * 8

print(f"Gerät: {GERAET}  |  ESP32-Sync: {'AKTIV' if ESP32_SYNC else 'AUS'}")

MIN_ABS_PEAK = kalibriere_rauschboden(
    sdr, SF, BANDWIDTH, SAMPLERATE, preamble_symbole, n=20
)

while True:
    state = input(
        "\n[t] TDMA  [s] Einmalig senden  [e] Manuell empfangen  [q] Beenden\n"
        "Deine Wahl: "
    ).strip().lower()

    # ── BEENDEN ───────────────────────────────────────────────────────────
    if state == "q":
        if esp32_serial and esp32_serial.is_open:
            esp32_serial.close()
        print("Tschüss!")
        break

    # ── TDMA ──────────────────────────────────────────────────────────────
    elif state == "t":
        eingabe = input(f"Gib bis zu {k_bits} Bits ein: ")
        try:
            bits = [int(b) for b in eingabe]
        except ValueError:
            print("[Fehler] Nur 0 und 1 erlaubt.")
            continue
        if len(bits) < k_bits:
            bits.extend([0] * (k_bits - len(bits)))
        elif len(bits) > k_bits:
            print(f"[Fehler] Maximal {k_bits} Bits.")
            continue

        codeword   = ldpc_encode(bits, g_matrix)
        symbole_tx = preamble_symbole + bits_zu_symbolen(codeword, SF)
        tx_signal  = generate_signal(symbole_tx, SF, BANDWIDTH, SAMPLERATE)

        print(f"Signal bereit: {len(tx_signal)} Samples. Strg+C zum Stoppen.")

        TX_DELAY = 0.020
        SLOT_DAUER = 0.500

        try:
            runde = 0
            if esp32_serial and esp32_serial.is_open:
                esp32_serial.reset_input_buffer()

            while True:
                runde += 1

                # 1. AUF DEN GPS-SEKUNDENSCHLAG WARTEN
                if esp32_serial and esp32_serial.is_open:
                    esp32_serial.reset_input_buffer()
                    while True:
                        if esp32_serial.in_waiting > 0:
                            if esp32_serial.read(1) == b'S':
                                t_start = time.time()
                                break
                        try:
                            _ = sdr.rx()
                        except:
                            pass
                else:
                    while time.time() % 1.0 > 0.005:
                        _ = sdr.rx()
                    while time.time() % 1.0 < 0.995:
                        _ = sdr.rx()
                    t_start = time.time()

                rx_buffers = []

                # ==============================================================
                # GERÄT A LOGIK
                # ==============================================================
                if GERAET == 'A':
                    while time.time() - t_start < TX_DELAY:
                        pass

                    sdr.tx(tx_signal)

                    while time.time() - t_start < 0.490:
                        _ = sdr.rx()
                    while time.time() - t_start < SLOT_DAUER:
                        pass

                    rx_deadline = t_start + SLOT_DAUER + 0.050
                    while time.time() < rx_deadline:
                        rx_buffers.append(sdr.rx())

                # ==============================================================
                # GERÄT B LOGIK
                # ==============================================================
                elif GERAET == 'B':
                    rx_deadline = t_start + 0.050
                    while time.time() < rx_deadline:
                        rx_buffers.append(sdr.rx())

                    while time.time() - t_start < (SLOT_DAUER + TX_DELAY - 0.010):
                        _ = sdr.rx()
                    while time.time() - t_start < (SLOT_DAUER + TX_DELAY):
                        pass

                    sdr.tx(tx_signal)

                # ==============================================================
                # GEMEINSAMES DECODIEREN
                # ==============================================================
                t_dec = time.time()
                if not rx_buffers:
                    print(f"[{GERAET}] Runde {runde}: Keine Daten empfangen.")
                else:
                    full_rx = np.concatenate(rx_buffers)

                    try:
                        p_start = finde_paket_start(
                            full_rx, SF, BANDWIDTH, SAMPLERATE,
                            preamble_symbole, min_abs_peak=MIN_ABS_PEAK
                        )
                    except TypeError:
                        p_start = finde_paket_start(
                            full_rx, SF, BANDWIDTH, SAMPLERATE,
                            preamble_symbole
                        )

                    if p_start == -1:
                        print(f"[{GERAET}] Runde {runde}: Kein Paket im Slot gefunden.")
                    else:
                        num_s      = int((2**SF / BANDWIDTH) * SAMPLERATE)
                        pre_len    = N_PREAMBLE_SYMBOLS * num_s
                        n_bits     = h_matrix.shape[1]

                        block_size = SF + 3
                        pay_syms   = int(np.ceil(n_bits / float(block_size)))
                        pay_len    = pay_syms * num_s

                        # PRÄAMBEL-FIX
                        while p_start >= num_s:
                            test_chunk = full_rx[p_start - num_s : p_start]
                            sym = signal_dechirp(test_chunk, SF, BANDWIDTH, SAMPLERATE)
                            if len(sym) > 0 and sym[0][1] == 1:
                                val = sym[0][0]
                                if val <= 3 or val >= (2**SF) - 3:
                                    p_start -= num_s
                                else:
                                    break
                            else:
                                break

                        if p_start + pre_len + pay_len > len(full_rx):
                            print(f"[{GERAET}] Runde {runde}: Paket abgeschnitten.")
                        else:
                            pre_chunk = full_rx[p_start : p_start + pre_len]
                            pre_syms  = signal_dechirp(pre_chunk, SF, BANDWIDTH, SAMPLERATE)
                            up_b, dn_b = [], []
                            for lv, dr, _ in pre_syms:
                                sb = lv if lv < (2**SF)//2 else lv - (2**SF)
                                (up_b if dr == 1 else dn_b).append(sb)

                            if up_b and dn_b:
                                mean_up = sum(up_b) / len(up_b)
                                mean_dn = sum(dn_b) / len(dn_b)
                                cfo_bin = (mean_up - mean_dn) / 2.0
                                t_bin   = (mean_up + mean_dn) / 2.0
                            else:
                                cfo_bin = 0.0
                                t_bin   = 0.0

                            t_off = int(t_bin * (SAMPLERATE / BANDWIDTH))
                            pay_start = p_start + pre_len + t_off
                            pay_end   = pay_start + pay_len

                            if pay_end <= len(full_rx):
                                payload = full_rx[pay_start:pay_end]
                                coarse_cfo_hz = cfo_bin * (SAMPLERATE / num_s)

                                try:
                                    fine_cfo_hz = measure_fine_cfo(pre_chunk, SF, BANDWIDTH, SAMPLERATE)
                                except NameError:
                                    fine_cfo_hz = 0.0

                                total_cfo_hz = coarse_cfo_hz + fine_cfo_hz
                                t_arr = np.arange(len(payload)) / SAMPLERATE
                                payload_corrected = payload * np.exp(-1j * 2 * np.pi * total_cfo_hz * t_arr)

                                pay_syms_rx = signal_dechirp(payload_corrected, SF, BANDWIDTH, SAMPLERATE)
                                decoded = ldpc_decode(pay_syms_rx, h_matrix, SF)

                                print(f"[{GERAET}] Runde {runde}: ✓ EMPFANGEN in "
                                      f"{(time.time()-t_dec)*1000:.0f}ms (Drift: {total_cfo_hz:.0f} Hz)")
                                print(f"         Bits: {decoded.tolist()}")
                            else:
                                print(f"[{GERAET}] Runde {runde}: Payload außerhalb Buffer.")

        except KeyboardInterrupt:
            print(f"\n[{GERAET}] TDMA gestoppt.")

    # ── EINMALIG SENDEN ───────────────────────────────────────────────────
    elif state == "s":
        max_bits = 1296  # 4 Pakete à 324 Bits
        eingabe = input(f"Bits eingeben (max {max_bits}) oder 'q': ")
        if eingabe.lower() == 'q':
            continue
        try:
            bits = [int(b) for b in eingabe]
        except ValueError:
            print("[Fehler] Nur 0 und 1 erlaubt.")
            continue

        if len(bits) == 0:
            bits = [0] * k_bits
        while len(bits) % k_bits != 0:
            bits.append(0)

        if len(bits) > max_bits:
            print(f"[Fehler] Max {max_bits} Bits erlaubt.")
            continue

        alle_signale = []
        for i in range(0, len(bits), k_bits):
            chunk = bits[i:i+k_bits]
            codeword = ldpc_encode(chunk, g_matrix)
            if codeword is None:
                continue
            symbole   = preamble_symbole + bits_zu_symbolen(codeword, SF)
            tx_signal = generate_signal(symbole, SF, BANDWIDTH, SAMPLERATE)
            alle_signale.append(tx_signal)

        if alle_signale:
            final_tx_signal = np.concatenate(alle_signale)
            sdr.tx(final_tx_signal)
            print(f"Gesendet: {len(alle_signale)} Paket(e).")

    # ── MANUELL EMPFANGEN (Strg+C → decodieren) ───────────────────────────
    elif state == "e":
        print("Empfange... Strg+C zum Stoppen und Decodieren.")
        rx_buffers = []
        try:
            while True:
                rx_buffers.append(sdr.rx())
        except KeyboardInterrupt:
            pass

        print(f"{len(rx_buffers)} Buffer gesammelt → decodiere...")

        if not rx_buffers:
            print("Keine Daten.")
            continue

        full_rx = np.concatenate(rx_buffers)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dateiname = f"iq_dump_{timestamp}.npy"
        np.save(dateiname, full_rx)

        search_offset = 0
        alle_decodierten_bits = []
        paket_nr = 0

        num_s    = int((2**SF / BANDWIDTH) * SAMPLERATE)
        pre_len  = N_PREAMBLE_SYMBOLS * num_s
        n_bits   = h_matrix.shape[1]

        block_size = SF + 3
        pay_syms = int(np.ceil(n_bits / float(block_size)))
        pay_len  = pay_syms * num_s

        while True:
            kandidaten = finde_paket_start_bruteforce(
                full_rx[search_offset:], SF, BANDWIDTH, SAMPLERATE,
                preamble_symbole, top_n=10, value_tolerance=3, verbose=False
            )

            if not kandidaten:
                if paket_nr == 0:
                    print("Kein Paket gefunden.")
                else:
                    print(f"Ende des Streams. {paket_nr} Pakete gefunden.")
                break

            best_offset, best_score, best_match, pre_syms = kandidaten[0]

            MIN_MATCH_REQUIRED = 10
            if best_match < MIN_MATCH_REQUIRED:
                if paket_nr == 0:
                    print(f"Bester Kandidat zu schwach (Match: {best_match}/{N_PREAMBLE_SYMBOLS}).")
                else:
                    print("Ende des Streams.")
                break

            p_start = search_offset + best_offset
            paket_nr += 1

            if p_start + pre_len + pay_len > len(full_rx):
                print("Paket abgeschnitten – Puffer zu Ende.")
                break

            up_b, dn_b = [], []
            for lv, dr, _ in pre_syms:
                sb = lv if lv < (2**SF)//2 else lv - (2**SF)
                (up_b if dr == 1 else dn_b).append(sb)

            if up_b and dn_b:
                mean_up = sum(up_b) / len(up_b)
                mean_dn = sum(dn_b) / len(dn_b)
                cfo_bin = (mean_up - mean_dn) / 2.0
                t_bin   = (mean_up + mean_dn) / 2.0
            else:
                cfo_bin = 0.0
                t_bin   = 0.0

            t_off = int(t_bin * (SAMPLERATE / BANDWIDTH))

            pay_start = p_start + pre_len + t_off
            pay_end   = pay_start + pay_len

            if pay_end <= len(full_rx):
                payload = full_rx[pay_start:pay_end]

                coarse_cfo_hz = cfo_bin * (SAMPLERATE / num_s)
                pre_chunk = full_rx[p_start : p_start + pre_len]
                fine_cfo_hz = measure_fine_cfo(pre_chunk, SF, BANDWIDTH, SAMPLERATE)
                total_cfo_hz = coarse_cfo_hz + fine_cfo_hz

                t_arr = np.arange(len(payload)) / SAMPLERATE
                payload_corrected = payload * np.exp(-1j * 2 * np.pi * total_cfo_hz * t_arr)

                pay_syms_rx = signal_dechirp(payload_corrected, SF, BANDWIDTH, SAMPLERATE)

                decoded = ldpc_decode(pay_syms_rx, h_matrix, SF)
                alle_decodierten_bits.extend(decoded.tolist())
                print(f"[Paket {paket_nr}] Match={best_match}/{N_PREAMBLE_SYMBOLS} "
                      f"Drift={total_cfo_hz:.0f}Hz → decodiert.")

                search_offset = pay_end
            else:
                print("Payload außerhalb Buffer.")
                break

        if alle_decodierten_bits:
            bit_string = "".join(str(b) for b in alle_decodierten_bits)
            print(f"\nGESAMT-PAYLOAD ({len(alle_decodierten_bits)} Bits): {bit_string}")

    else:
        print("Ungültige Eingabe.")