import numpy as np
import adi
import scipy.signal
from scipy.sparse import coo_matrix, csr_matrix
from ldpc.bp_decoder import BpDecoder
import time
import serial  # NEU: für ESP32-Zeitsync


SDR_IP   = "ip:192.168.1.10"
BANDWIDTH   = 1000000
SAMPLERATE  = 2000000
CENTER_FREQ = 2450000000
BUFFER_SIZE = int(SAMPLERATE * 0.01)   # 10ms pro sdr.rx()
SF = 8

# ═══════════════════════════════════════════════════════════════
# TDMA KONFIGURATION
# ═══════════════════════════════════════════════════════════════
#   'A'  →  sendet in der ersten Hälfte  (0.000s – 0.500s)
#   'B'  →  sendet in der zweiten Hälfte (0.500s – 1.000s)
GERAET = 'A'          # ← HIER ÄNDERN

SLOT_DAUER   = 0.500  # Gesamte Slot-Länge
PUFFER_START = 0.005  # Puffer vor dem Senden
TX_DAUER     = 0.200  # Maximale Sendezeit
PUFFER_ENDE  = 0.005  # Puffer am Ende

# ═══════════════════════════════════════════════════════════════
# KALIBRIERUNG
# ═══════════════════════════════════════════════════════════════
KALIBRIERUNGS_FAKTOR = 0.5
MIN_ABS_PEAK         = None

# ═══════════════════════════════════════════════════════════════
# ESP32 ZEITSYNC
# ═══════════════════════════════════════════════════════════════
ESP32_PORT    = "COM10"       
ESP32_BAUD    = 115200
ESP32_SYNC    = True         # False = kein ESP32 angeschlossen
esp32_serial  = None         # wird beim Start befüllt


# ───────────────────────────────────────────────────────────────
# ESP32 ZEITSYNC FUNKTIONEN
# ───────────────────────────────────────────────────────────────

def sync_esp32(port: str = ESP32_PORT, baudrate: int = ESP32_BAUD):
    """
    Verbindet mit dem ESP32 und sendet die aktuelle Unix-Zeit in ms.
    Wartet auf den Sekundenübergang für maximale Präzision.
    Gibt das offene Serial-Objekt zurück.
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
    except serial.SerialException as e:
        print(f"[ESP32] Fehler beim Öffnen von {port}: {e}")
        return None

    time.sleep(2)  # ESP32 Reset abwarten

    print("[ESP32] Warte auf READY...", end=" ", flush=True)
    deadline = time.time() + 10
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "READY":
            break
    else:
        print("TIMEOUT – ESP32 antwortet nicht!")
        ser.close()
        return None

    # Kurz vor dem Sekundenübergang warten für präziseren Sync
    while time.time() % 1.0 < 0.990:
        time.sleep(0.001)
    while time.time() % 1.0 > 0.001:   # busy-wait über den Übergang
        pass

    ts_ms = int(time.time() * 1000)
    ser.write(f"SYNC:{ts_ms}\n".encode())

    resp = ser.readline().decode(errors="ignore").strip()
    if resp.startswith("SYNCED:"):
        print(f"OK – synced auf {resp.split(':')[1]}s")
    else:
        print(f"Warnung: Unerwartete Antwort: {resp}")

    return ser


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
    decoder = BpDecoder(H_sp, error_rate=0.1, max_iter=20, bp_method='ps')
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


def generate_signal(symbole, sf, bw, fs):
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    phase = 2 * np.pi * (0.5 * k * t**2 + (-bw/2) * t)
    base = np.cos(phase) + 1j * np.sin(phase)
    chain = []
    for lora_wert, richtung, qpsk_wert in symbole:
        chirp = base.copy()
        shift = int((lora_wert / 2**sf) * num_samples)
        chirp = np.roll(chirp, -shift)
        chirp = chirp * np.exp(-1j * np.angle((chirp * np.conj(base))[0]))
        if richtung == 0:
            chirp = np.conj(chirp)
        chirp = chirp * np.exp(1j * [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4][qpsk_wert])
        chain.append(chirp)
    return (np.concatenate(chain) * (2**14 * 0.5)).astype(np.complex64)


def signal_dechirp(full_signal, sf, bw, fs):
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    phase = 2 * np.pi * (0.5 * k * t**2 + (-bw/2) * t)
    base_up   = np.cos(phase) + 1j * np.sin(phase)
    base_down = np.conj(base_up)
    window    = np.hanning(num_samples)
    decoded   = []
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
        rp, ip = pc.real, pc.imag
        if   rp >= 0 and ip >= 0: q = 0
        elif rp <  0 and ip >= 0: q = 1
        elif rp <  0 and ip <  0: q = 2
        else:                      q = 3
        decoded.append((val, d, q))
    return decoded


# ───────────────────────────────────────────────────────────────
# DC-KORREKTUR & KALIBRIERUNG
# ───────────────────────────────────────────────────────────────

def entferne_dc_lokal(rx_data, num_samples):
    """DC pro Chirp-Fenster entfernen – robuster als globales mean()."""
    out = rx_data.copy()
    n   = len(rx_data) // num_samples
    for i in range(n):
        s, e = i * num_samples, (i + 1) * num_samples
        out[s:e] -= np.mean(rx_data[s:e])
    rest = n * num_samples
    if rest < len(rx_data):
        out[rest:] -= np.mean(rx_data[rest:])
    return out


def kalibriere_rauschboden(sdr, sf, bw, fs, preamble_symbole, n=20):
    print("\n" + "═"*50)
    print(" KALIBRIERUNG – Rauschboden messen")
    print("═"*50)
    print(" WICHTIG: Kein anderes Gerät darf senden!")
    print(f" Messe {n}× ...\n")
    num_samples = int((2**sf / bw) * fs)
    ref  = generate_signal(preamble_symbole[:1], sf, bw, fs)
    peaks = []
    for i in range(n):
        rx   = entferne_dc_lokal(sdr.rx(), num_samples)
        korr = scipy.signal.correlate(rx, np.conj(ref), mode='valid', method='fft')
        p    = float(np.max(np.abs(korr)))
        peaks.append(p)
        print(f" Messung {i+1:2d}/{n}: Peak = {p:>12.0f}", end='\r')
    schwellwert = np.median(peaks) * 2.0
    print(f"\n\n Min: {min(peaks):.0f}  Max: {max(peaks):.0f}  "
          f"Avg: {np.mean(peaks):.0f}")
    print(f" → MIN_ABS_PEAK = {max(peaks):.0f} × {KALIBRIERUNGS_FAKTOR} = {schwellwert:.0f}")
    print("═"*50 + "\n")
    return schwellwert


# ───────────────────────────────────────────────────────────────
# PREAMBLE-DETEKTION
# ───────────────────────────────────────────────────────────────

def finde_paket_start(rx_data, sf, bw, fs, preamble_symbols, min_abs_peak=None):
    num_samples = int((2**sf / bw) * fs)
    rx_data     = entferne_dc_lokal(rx_data, num_samples)
    ref         = generate_signal(preamble_symbols[:1], sf, bw, fs)

    corr      = scipy.signal.correlate(rx_data, np.conj(ref), mode='valid', method='fft')
    mag       = np.abs(corr)
    idx       = int(np.argmax(mag))
    max_peak  = mag[idx]
    noise     = np.median(mag)
    if noise == 0:
        noise = 1e-9
    ratio = max_peak / noise

    if min_abs_peak is not None and max_peak < min_abs_peak:
        print(f"[DEBUG] Abgelehnt (Pegel): {max_peak:.0f} < {min_abs_peak:.0f}")
        return -1

    if ratio < 5:
        print(f"[DEBUG] Abgelehnt (Ratio {ratio:.1f}x < 5x): Peak={max_peak:.0f}")
        return -1

    idx2 = idx + num_samples
    if idx2 >= len(mag):
        print(f"[DEBUG] Abgelehnt (kein Platz für P2)")
        return -1
    fenster    = mag[max(0, idx2-3) : idx2+3]
    max_second = float(np.max(fenster)) if len(fenster) > 0 else 0
    if max_second < max_peak * 0.2:
        print(f"[DEBUG] Abgelehnt (P2 zu schwach): P1={max_peak:.0f} P2={max_second:.0f}")
        return -1

    n_up = sum(1 for (_, r, _) in preamble_symbols if r == 1)
    perioden_peaks = []
    for k in range(n_up):
        pos = idx + k * num_samples
        if pos >= len(mag):
            break
        f_s = max(0, pos - 3)
        f_e = min(len(mag), pos + 3)
        perioden_peaks.append(float(np.max(mag[f_s:f_e])))

    if len(perioden_peaks) < max(1, n_up // 2):
        print(f"[DEBUG] Abgelehnt (Periodizität): nur {len(perioden_peaks)}/{n_up}")
        return -1

    median_peri = np.median(perioden_peaks)
    peri_rat    = median_peri / max_peak
    n_stark     = sum(1 for p in perioden_peaks if p > max_peak * 0.15)

    if peri_rat < 0.20 or n_stark < max(1, n_up // 2):
        print(f"[DEBUG] Abgelehnt (Periodizität): median={peri_rat*100:.0f}% "
              f"stark={n_stark}/{n_up}")
        return -1       

    print(f"\n[DEBUG] Preamble OK! P1={max_peak:.0f} P2={max_second:.0f} "
          f"Ratio={ratio:.1f}x Peri={peri_rat*100:.0f}%")
    return idx          


# ───────────────────────────────────────────────────────────────
# TDMA HILFSFUNKTIONEN
# ───────────────────────────────────────────────────────────────

def get_slot_pos():
    """Position innerhalb der aktuellen Sekunde (0.0 – 1.0)."""
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
    print(f"SDR verbunden! Bereit auf {CENTER_FREQ/1e6} MHz.")
except Exception as e:
    print(f"Fehler bei der SDR-Verbindung: {e}")
    exit(1)

# ── ESP32 Zeitsync ────────────────────────────────────
if ESP32_SYNC:
    print("\n" + "═"*50)
    print(" ESP32 ZEITSYNC")
    print("═"*50)
    esp32_serial = sync_esp32(ESP32_PORT, ESP32_BAUD)
    if esp32_serial is None:
        print("[ESP32] Sync fehlgeschlagen – fahre ohne ESP32 fort.")
        ESP32_SYNC = False
    print("═"*50)

N_PREAMBLE_SYMBOLS = 16
preamble_symbole   = [(0, 1, 0)] * 8 + [(0, 0, 0)] * 8

print("\n#################################################################")
print(f" Start des Programmes  –  Gerät: {GERAET}")
print(f" Slot-Schema:")
print(f"   Sender A sendet: 0.000s – 0.500s  (TX dann RX)")
print(f"   Sender B sendet: 0.500s – 1.000s  (RX dann TX)")
print(f" Timing pro Slot:")
print(f"   {PUFFER_START*1000:.0f}ms Puffer → {TX_DAUER*1000:.0f}ms Senden → "
      f"{(SLOT_DAUER-PUFFER_START-TX_DAUER-PUFFER_ENDE)*1000:.0f}ms Decodieren → "
      f"{PUFFER_ENDE*1000:.0f}ms Puffer")
if ESP32_SYNC:
    print(f" ESP32 Zeitsync: AKTIV auf {ESP32_PORT}")
print("#################################################################")

MIN_ABS_PEAK = kalibriere_rauschboden(
    sdr, SF, BANDWIDTH, SAMPLERATE, preamble_symbole, n=20
)

while True:
    print("\n" + "="*40)
    state = input(
        "Was möchtest du tun?\n"
        " [t] TDMA starten\n"
        " [s] Einmalig senden\n"
        " [e] Manuell empfangen (Strg+C = stopp + decodieren)\n"
        " [q] Beenden\n"
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
        print(f"\n--- TDMA MODUS (Gerät {GERAET}) ---")
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
        funk_ms    = len(tx_signal) / SAMPLERATE * 1000

        print(f"Signal bereit: {len(tx_signal)} Samples = {funk_ms:.2f}ms über Funk")
        print("Drücke Strg+C um zu stoppen.\n")

        try:
            runde = 0
            while True:
                runde += 1
                print(f"\n─── Runde {runde} ───────────────────────────────")

                # ── TX ────────────────────────────────────────────────────
                print(f"[{GERAET}] Warte auf TX-Slot...", end='\r')
                t_slot = warte_auf_tx_slot(GERAET)
                ts     = time.strftime('%H:%M:%S')
                ms     = int((t_slot % 1) * 1000)
                print(f"[{GERAET}] TX-Slot @ {ts}.{ms:03d}")

                while time.time() - t_slot < PUFFER_START:
                    pass

                t0 = time.time()
                sdr.tx(tx_signal)
                print(f"[{GERAET}] Gesendet in {(time.time()-t0)*1000:.1f}ms  "
                      f"(Funk: {funk_ms:.2f}ms)")

                verbleibend = SLOT_DAUER - PUFFER_START - PUFFER_ENDE - (time.time() - t_slot)
                if verbleibend > 0:
                    time.sleep(verbleibend)

                # ── RX: sammeln ───────────────────────────────────────────
                print(f"[{GERAET}] Warte auf RX-Slot...", end='\r')
                t_rx  = warte_auf_rx_slot(GERAET)
                ts    = time.strftime('%H:%M:%S')
                ms    = int((t_rx % 1) * 1000)
                print(f"[{GERAET}] RX-Slot  @ {ts}.{ms:03d}")

                while time.time() - t_rx < PUFFER_START:
                    pass

                rx_buffers  = []
                rx_deadline = t_rx + PUFFER_START + TX_DAUER
                while time.time() < rx_deadline:
                    rx_buffers.append(sdr.rx())

                print(f"[{GERAET}] {len(rx_buffers)} Buffer → decodiere...")

                # ── RX: decodieren ────────────────────────────────────────
                t_dec   = time.time()
                full_rx = np.concatenate(rx_buffers)
                p_start = finde_paket_start(
                    full_rx, SF, BANDWIDTH, SAMPLERATE,
                    preamble_symbole, min_abs_peak=MIN_ABS_PEAK
                )

                if p_start == -1:
                    print(f"[{GERAET}] ✗ Kein Paket.")
                else:
                    num_s      = int((2**SF / BANDWIDTH) * SAMPLERATE)
                    pre_len    = N_PREAMBLE_SYMBOLS * num_s
                    n_bits     = h_matrix.shape[1]
                    
                    block_size = SF + 3
                    pay_syms   = int(np.ceil(n_bits / float(block_size)))
                    pay_len    = pay_syms * num_s

                    if p_start + pre_len + pay_len > len(full_rx):
                        print(f"[{GERAET}] ✗ Paket abgeschnitten.")
                    else:
                        pre_chunk = full_rx[p_start : p_start + pre_len]
                        pre_syms  = signal_dechirp(pre_chunk, SF, BANDWIDTH, SAMPLERATE)
                        up_b, dn_b = [], []
                        for lv, dr, _ in pre_syms:
                            sb = lv if lv < (2**SF)//2 else lv - (2**SF)
                            (up_b if dr == 1 else dn_b).append(sb)
                        pay_start = p_start + pre_len
                        pay_end   = pay_start + pay_len
                        if pay_end <= len(full_rx):
                            pay_syms_rx = signal_dechirp(
                                full_rx[pay_start:pay_end], SF, BANDWIDTH, SAMPLERATE
                            )
                            decoded = ldpc_decode(pay_syms_rx, h_matrix, SF)
                            print(f"[{GERAET}] ✓ EMPFANGEN in {(time.time()-t_dec)*1000:.0f}ms")
                            print(f"         Bits: {decoded.tolist()}")
                        else:
                            print(f"[{GERAET}] ✗ Payload außerhalb Buffer.")

                rest = SLOT_DAUER - (time.time() - t_rx)
                if rest > 0:
                    time.sleep(rest)

        except KeyboardInterrupt:
            print(f"\n[{GERAET}] TDMA gestoppt.")

    # ── EINMALIG SENDEN ───────────────────────────────────────────────────
    elif state == "s":
        print("\n--- SENDE PFAD ---")
        max_bits = 1296  # 4 Pakete à 324 Bits
        eingabe = input(f"Bits eingeben (max {max_bits}) oder 'q': ")
        if eingabe.lower() == 'q':
            continue
        try:
            bits = [int(b) for b in eingabe]
        except ValueError:
            print("[Fehler] Nur 0 und 1 erlaubt.")
            continue
            
        # Auffüllen auf das nächste Vielfache von k_bits (324)
        if len(bits) == 0:
            bits = [0] * k_bits
        while len(bits) % k_bits != 0:
            bits.append(0)
            
        if len(bits) > max_bits:
            print(f"[Fehler] Max {max_bits} Bits erlaubt.")
            continue

        print(f"Verarbeite {len(bits)} Bits in {len(bits)//k_bits} Blöcken à {k_bits} Bits...")
        
        alle_signale = []
        for i in range(0, len(bits), k_bits):
            chunk = bits[i:i+k_bits]
            codeword  = ldpc_encode(chunk, g_matrix)
            if codeword is None:
                continue
            symbole   = preamble_symbole + bits_zu_symbolen(codeword, SF)
            tx_signal = generate_signal(symbole, SF, BANDWIDTH, SAMPLERATE)
            alle_signale.append(tx_signal)
            
        if alle_signale:
            final_tx_signal = np.concatenate(alle_signale)
            print(f"Sende {len(alle_signale)} zusammenhängende Pakete in einem Burst...")
            sdr.tx(final_tx_signal)
            print("Gesendet!")

    # ── MANUELL EMPFANGEN (Strg+C → decodieren) ───────────────────────────
    elif state == "e":
        print("\n--- MANUELL EMPFANGEN ---")
        print("Empfange... Drücke Strg+C um zu stoppen und zu decodieren.\n")
        rx_buffers = []
        try:
            while True:
                rx_buffers.append(sdr.rx())
                print(f"  {len(rx_buffers)} Buffer ({len(rx_buffers)*BUFFER_SIZE/SAMPLERATE*1000:.0f}ms)...",
                      end='\r')
        except KeyboardInterrupt:
            pass

        print(f"\n{len(rx_buffers)} Buffer gesammelt → decodiere...")

        if not rx_buffers:
            print("Keine Daten.")
            continue

        full_rx = np.concatenate(rx_buffers)

        # ───────────────────────────────────────────────────────────────
        # I/Q DATEN SPEICHERN
        # ───────────────────────────────────────────────────────────────
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dateiname = f"iq_dump_{timestamp}.npy"
        np.save(dateiname, full_rx)
        print(f"\n[DEBUG] 💾 Rohe I/Q-Daten gespeichert in: {dateiname}")
        # ───────────────────────────────────────────────────────────────

        # NEU: Schleife sucht solange im Puffer weiter, bis keine Pakete mehr da sind
        search_offset = 0
        alle_decodierten_bits = []
        paket_nr = 0

        while True:
            p_start_rel = finde_paket_start(
                full_rx[search_offset:], SF, BANDWIDTH, SAMPLERATE,
                preamble_symbole, min_abs_peak=MIN_ABS_PEAK
            )

            if p_start_rel == -1:
                if paket_nr == 0:
                    print("✗ Kein Paket gefunden.")
                else:
                    print(f"\n--- Ende des Streams. {paket_nr} Pakete gefunden. ---")
                break

            # Absolute Startposition im gesamten Array berechnen
            p_start = search_offset + p_start_rel
            paket_nr += 1
            print(f"\n[Paket {paket_nr}] Decodiere...")

            num_s    = int((2**SF / BANDWIDTH) * SAMPLERATE)
            pre_len  = N_PREAMBLE_SYMBOLS * num_s
            n_bits   = h_matrix.shape[1]
            
            block_size = SF + 3
            pay_syms = int(np.ceil(n_bits / float(block_size)))
            pay_len  = pay_syms * num_s

            if p_start + pre_len + pay_len > len(full_rx):
                print("✗ Paket abgeschnitten – Puffer zu Ende.")
                break

            pre_chunk = full_rx[p_start : p_start + pre_len]
            pre_syms  = signal_dechirp(pre_chunk, SF, BANDWIDTH, SAMPLERATE)
            print(pre_syms)
            up_b, dn_b = [], []
            for lv, dr, _ in pre_syms:
                sb = lv if lv < (2**SF)//2 else lv - (2**SF)
                (up_b if dr == 1 else dn_b).append(sb)
            timing_bin = ((sum(up_b)/len(up_b)) - (sum(dn_b)/len(dn_b))) / 2.0 \
                         if up_b and dn_b else 0.0
            t_off = int(timing_bin * (SAMPLERATE / BANDWIDTH))

            pay_start   = p_start + pre_len + t_off
            pay_end     = pay_start + pay_len
            
            if pay_end <= len(full_rx):
                pay_syms_rx = signal_dechirp(
                    full_rx[pay_start:pay_end], SF, BANDWIDTH, SAMPLERATE
                )
                print(pay_syms_rx)
                decoded = ldpc_decode(pay_syms_rx, h_matrix, SF)
                alle_decodierten_bits.extend(decoded.tolist())
                print(f" ✓ Payload {paket_nr} erfolgreich decodiert!")
                
                # WICHTIG: Den Such-Zeiger hinter das aktuelle Paket schieben!
                search_offset = pay_end
            else:
                print("✗ Payload außerhalb Buffer.")
                break

        # Am Ende die Gesamt-Ausgabe printen
        if alle_decodierten_bits:
            print("\n" + "="*50)
            print(f" ✓ GESAMT-PAYLOAD DECODIERT ({len(alle_decodierten_bits)} Bits)")
            print(f" Bits: {alle_decodierten_bits}")
            print("="*50)

    else:
        print("Ungültige Eingabe.")