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
SF = 5


# ═══════════════════════════════════════════════════════════════
# KALIBRIERUNG
# ═══════════════════════════════════════════════════════════════
KALIBRIERUNGS_FAKTOR = 0.5
MIN_ABS_PEAK         = None


# ───────────────────────────────────────────────────────────────
# Input Generierung
# ───────────────────────────────────────────────────────────────

def generiere_test_payload(anzahl_pakete, bits_pro_paket, seed=42):
    """
    Generiert EINEN deterministischen Pseudo-Zufalls-Block und 
    kopiert ihn für alle Pakete, damit Paketverluste die BER nicht ruinieren.
    """
    np.random.seed(seed) 
    
    # Nur für EIN Paket (z.B. 324 Bits) die Zufallszahlen generieren
    ein_paket_bits = np.random.randint(0, 2, bits_pro_paket).tolist()
    
    # Diesen exakt selben Block X-mal hintereinander hängen
    return ein_paket_bits * anzahl_pakete

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
    
    # DQPSK: Startphase festlegen (z.B. 45 Grad)
    aktuelle_phase = np.pi / 4  
    
    for lora_wert, richtung, qpsk_wert in symbole:
        chirp = base.copy()
        shift = int((lora_wert / 2**sf) * num_samples)
        chirp = np.roll(chirp, -shift)
        
        # Basis-Phase nullen
        chirp = chirp * np.exp(-1j * np.angle((chirp * np.conj(base))[0]))
        
        if richtung == 0:
            chirp = np.conj(chirp)
            
        # DQPSK-Logik: Phase abhängig vom Wert weiterdrehen
        # 0 -> +0°, 1 -> +90°, 2 -> +180°, 3 -> +270°
        phasen_sprung = qpsk_wert * (np.pi / 2)
        aktuelle_phase = (aktuelle_phase + phasen_sprung) % (2 * np.pi)
        
        # Aufaddierte Phase anwenden
        chirp = chirp * np.exp(1j * aktuelle_phase)
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
        korr = scipy.signal.correlate(rx, ref, mode='valid', method='fft')
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
def bewerte_preamble_match(decoded_symbols, expected_symbols, sf, cluster_tolerance=3):
    """
    CFO-tolerante Bewertung: Statt exakt Wert=0 zu verlangen, wird geprüft,
    ob alle Symbole mit korrekt erkannter Richtung EINEN gemeinsamen Wert
    teilen (Cluster). Ein echter Restfrequenzfehler (CFO) verschiebt alle
    Up-Chirps gleich (und alle Down-Chirps gleich, nur gegenläufig) -
    genau das wird hier erkannt, statt es fälschlich als 'kein Preamble'
    zu werten.
    """
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


def finde_paket_start_bruteforce(rx_data, sf, bw, fs, preamble_symbols,
                                   top_n=10, value_tolerance=2, verbose=True):
    num_samples = int((2**sf / bw) * fs)
    rx_clean    = entferne_dc_lokal(rx_data, num_samples)
    n_sym       = len(preamble_symbols)
    pre_len     = n_sym * num_samples
    max_offset  = len(rx_clean) - pre_len
    if max_offset < 0:
        if verbose:
            print("[BRUTE-FORCE] Puffer zu kurz.")
        return []
 
    ref_up = generate_signal([(0, 1, 0)], sf, bw, fs)
    ref_dn = generate_signal([(0, 0, 0)], sf, bw, fs)
 
    t0 = time.time()
    corr_up = np.abs(scipy.signal.correlate(rx_clean, ref_up, mode='valid', method='fft'))
    corr_dn = np.abs(scipy.signal.correlate(rx_clean, ref_dn, mode='valid', method='fft'))
    t1 = time.time()
 
    M = max_offset + 1
    score = np.zeros(M)
    for i, (_, richtung, _) in enumerate(preamble_symbols):
        quelle = corr_up if richtung == 1 else corr_dn
        start  = i * num_samples
        score += quelle[start: start + M]
    t2 = time.time()
 
    # Nicht-Maximum-Unterdrückung: Top-Kandidaten sollen sich nicht alle
    # auf denselben Peak häufen, sondern verschiedene Regionen abdecken.
    order = np.argsort(score)[::-1]
    kandidaten_idx = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - k) > num_samples // 2 for k in kandidaten_idx):
            kandidaten_idx.append(idx)
        if len(kandidaten_idx) >= top_n:
            break
    t3 = time.time()
 
    ergebnisse = []
    for offset in kandidaten_idx:
        chunk = rx_clean[offset: offset + pre_len]
        syms  = signal_dechirp(chunk, sf, bw, fs)
        match = bewerte_preamble_match(syms, preamble_symbols, sf, value_tolerance)
        ergebnisse.append((offset, float(score[offset]), match, syms))
    t4 = time.time()
 
    ergebnisse.sort(key=lambda r: r[2], reverse=True)
 
    if verbose:
        print(f"[BRUTE-FORCE] Stufe1(Korrelation)={1000*(t1-t0):.1f}ms  "
              f"Summierung={1000*(t2-t1):.1f}ms  NMS-Auswahl={1000*(t3-t2):.1f}ms  "
              f"Stufe2(Dechirp x{len(kandidaten_idx)})={1000*(t4-t3):.1f}ms  "
              f"GESAMT={1000*(t4-t0):.1f}ms  ({M} Positionen getestet)")
 
    return ergebnisse


def finde_paket_start(rx_data, sf, bw, fs, preamble_symbols):

    kandidaten = finde_paket_start_bruteforce(
        rx_data,
        sf,
        bw,
        fs,
        preamble_symbols,
        top_n=10,
        value_tolerance=2,
        verbose=True
    )

    if len(kandidaten) == 0:
        return -1

    offset, score, match, syms = kandidaten[0]

    print(f"[BRUTE] Match={match:.1f}%   Score={score:.0f}")

    return offset


def measure_fine_cfo(pre_chunk, sf, bw, fs):

    """Misst den exakten Frequenzfehler in Hz anhand der Phasenrotation der Präambel."""
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    phase = 2 * np.pi * (0.5 * k * t**2 + (-bw/2) * t)
    base_down = np.conj(np.cos(phase) + 1j * np.sin(phase))
    window = np.hanning(num_samples)
    
    phases = []
    # Nur die ersten 8 Symbole anschauen (das sind reine Up-Chirps mit Wert 0)
    for i in range(8):
        chunk = pre_chunk[i*num_samples:(i+1)*num_samples]
        dc = chunk * base_down * window
        cfft = np.fft.fft(dc)
        pos = np.argmax(np.abs(cfft))
        phases.append(np.angle(cfft[pos]))
        
    deltas = []
    for i in range(1, len(phases)):
        # Phasenunterschied zwischen zwei Symbolen berechnen (-pi bis +pi)
        diff = (phases[i] - phases[i-1])
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        deltas.append(diff)
        
    # Durchschnittliche Rotation pro Symbol
    mean_drift_per_symbol = sum(deltas) / len(deltas)
    symbol_time = num_samples / fs
    
    # Umrechnung von Radiant/Symbol in Hertz
    fine_cfo_hz = mean_drift_per_symbol / (2 * np.pi) / symbol_time
    return fine_cfo_hz

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

N_PREAMBLE_SYMBOLS = 16
preamble_symbole   = [(0, 1, 0)] * 8 + [(0, 0, 0)] * 8


while True:
    print("\n" + "="*40)
    state = input(
        "Was möchtest du tun?\n"
        " [s] Einmalig senden\n"
        " [e] Empfangen (Strg+C = stopp + decodieren)\n"
        " [q] Beenden\n"
        "Deine Wahl: "
    ).strip().lower()

    # ── BEENDEN ───────────────────────────────────────────────────────────
    if state == "q":
        print("Tschüss!")
        break


    # ── EINMALIG SENDEN ───────────────────────────────────────────────────
    elif state == "s":
        file_name = input("Gib den gespeicherten Daten einen Namen (ohne .txt): ").strip()
        if not file_name:
            file_name = "tx_dump"
            
        print("\n--- SENDE PFAD ---")
        bits = generiere_test_payload(1, 324)
            
        # Auffüllen auf das nächste Vielfache von k_bits (324)
        if len(bits) == 0:
            bits = [0] * k_bits
        while len(bits) % k_bits != 0:
            bits.append(0)
            
        anzahl_bloecke = len(bits) // k_bits
        print(f"Verarbeite {len(bits)} Bits in {anzahl_bloecke} Blöcken à {k_bits} Bits...")
        
        alle_signale = []
        detailed_log = [] # Hier speichern wir die langen Datensätze zwischen
        
        for i in range(0, len(bits), k_bits):
            chunk = bits[i:i+k_bits]
            codeword  = ldpc_encode(chunk, g_matrix)
            
            if codeword is None:
                continue
                
            symbole   = preamble_symbole + bits_zu_symbolen(codeword, SF)
            
            if isinstance(codeword, np.ndarray):
                codeword_list = codeword.tolist()
            else:
                codeword_list = codeword
                
            block_nr = (i // k_bits) + 1
            
            # Detail-Daten in den Speicher legen
            detail_str = f"--- Block {block_nr} ---\n"
            detail_str += f"Codierte Bits (LDPC):\n{codeword_list}\n\n"
            detail_str += f"Symbole (LoRa-Wert, Richtung, QPSK-Wert):\n{symbole}\n\n"
            detail_str += "-" * 40 + "\n\n"
            detailed_log.append(detail_str)
            
            tx_signal = generate_signal(symbole, SF, BANDWIDTH, SAMPLERATE)
            alle_signale.append(tx_signal)
            
        if alle_signale:
            final_tx_signal = np.concatenate(alle_signale)
            
            # ───────────────────────────────────────────────────────────
            # TXT-Datei schreiben (Übersicht oben, Details unten)
            # ───────────────────────────────────────────────────────────
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(f"{file_name}.txt", "w", encoding="utf-8") as f:
                # 1. KOPFDATEN & ÜBERSICHT
                f.write("="*60 + "\n")
                f.write(" SDR SENDE-PROTOKOLL\n")
                f.write("="*60 + "\n")
                f.write(f"Datum / Zeit : {timestamp}\n")
                f.write(f"Konfiguration: SF={SF}, Bandbreite={BANDWIDTH/1e6} MHz, Samplerate={SAMPLERATE/1e6} MSPS\n")
                f.write(f"Gesamt Bits  : {len(bits)} (Nutzdaten)\n")
                f.write(f"Total Pakete : {anzahl_bloecke}\n")
                f.write("="*60 + "\n\n")
                
                # 2. GENERIERTE PAYLOAD
                f.write("ROHE NUTZDATEN (Raw Payload, uncodiert):\n")
                f.write(f"{bits}\n\n")
                f.write("="*60 + "\n\n")
                
                # 3. DIE LANGEN DATENSÄTZE
                f.write("DETAILLIERTE PAKET-DATEN (LDPC & Symbole):\n\n")
                for log_teil in detailed_log:
                    f.write(log_teil)

            print(f"Sende {len(alle_signale)} zusammenhängende Pakete in einem Burst...")
            print(f"💾 Die Log-Daten (Übersicht zuerst) wurden in '{file_name}.txt' gespeichert!")
            sdr.tx(final_tx_signal)
            print("Gesendet!")


    # ── EMPFANGEN (Strg+C → decodieren) ───────────────────────────
    elif state == "e":
        file_name = input("Gib den Namen für die Empfangs-Logdatei ein (ohne .txt): ").strip()
        if not file_name:
            file_name = "rx_dump"

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

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dateiname = f"iq_dump_{timestamp}.npy"
        np.save(dateiname, full_rx)
        print(f"\n[DEBUG] 💾 Rohe I/Q-Daten gespeichert in: {dateiname}")

        search_offset = 0
        alle_decodierten_bits = []
        paket_nr = 0
        gesamt_bit_fehler = 0
        gesamt_korrigiert = 0
        
        num_s    = int((2**SF / BANDWIDTH) * SAMPLERATE)
        pre_len  = N_PREAMBLE_SYMBOLS * num_s
        n_bits   = h_matrix.shape[1]
        
        block_size = SF + 3
        pay_syms = int(np.ceil(n_bits / float(block_size)))
        pay_len  = pay_syms * num_s

        expected_total_payload = generiere_test_payload(100, k_bits)
        
        detailed_log = [] # Speicher für die langen Paket-Datensätze
        txt_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        while True:
            kandidaten = finde_paket_start_bruteforce(
                full_rx[search_offset:], SF, BANDWIDTH, SAMPLERATE,
                preamble_symbole, top_n=10, value_tolerance=3, verbose=True
            )

            if not kandidaten:
                if paket_nr == 0:
                    print("✗ Kein Paket gefunden.")
                else:
                    print(f"\n--- Ende des Streams. {paket_nr} Pakete gefunden. ---")
                break

            best_offset, best_score, best_match, pre_syms = kandidaten[0]

            MIN_MATCH_REQUIRED = 10
            if best_match < MIN_MATCH_REQUIRED:
                if paket_nr == 0:
                    print(f"✗ Bester Kandidat war zu schwach (Match: {best_match}/{N_PREAMBLE_SYMBOLS}).")
                else:
                    print(f"\n--- Ende des Streams. ---")
                break

            p_start = search_offset + best_offset
            paket_nr += 1
            print(f"\n[Paket {paket_nr}] Präambel bei Index {p_start} (Match: {best_match}/{N_PREAMBLE_SYMBOLS})!")

            if p_start + pre_len + pay_len > len(full_rx):
                print("✗ Paket abgeschnitten – Puffer zu Ende.")
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
            pay_start   = p_start + pre_len + t_off
            pay_end     = pay_start + pay_len
            
            if pay_end <= len(full_rx):
                payload = full_rx[pay_start:pay_end]
                
                coarse_cfo_hz = cfo_bin * (SAMPLERATE / num_s)
                pre_chunk = full_rx[p_start : p_start + pre_len]
                fine_cfo_hz = measure_fine_cfo(pre_chunk, SF, BANDWIDTH, SAMPLERATE)
                total_cfo_hz = coarse_cfo_hz + fine_cfo_hz
                
                t_arr = np.arange(len(payload)) / SAMPLERATE
                payload_corrected = payload * np.exp(-1j * 2 * np.pi * total_cfo_hz * t_arr)
                
                pay_syms_rx = signal_dechirp(
                    payload_corrected, SF, BANDWIDTH, SAMPLERATE
                )
                
                decoded = ldpc_decode(pay_syms_rx, h_matrix, SF)
                alle_decodierten_bits.extend(decoded.tolist())
                
                # STATISTIKEN BERECHNEN
                raw_rx_bits = []
                for lora_val, dir_val, qpsk_val in pay_syms_rx:
                    for b in format(lora_val, f'0{SF}b'): raw_rx_bits.append(int(b))
                    raw_rx_bits.append(int(dir_val))
                    for b in format(qpsk_val, '02b'): raw_rx_bits.append(int(b))
                    
                ideal_codeword = ldpc_encode(decoded, g_matrix)
                korrigierte_fehler = 0
                if ideal_codeword is not None:
                    korrigierte_fehler = sum(1 for a, b in zip(raw_rx_bits[:n_bits], ideal_codeword) if a != b)
                    gesamt_korrigiert += korrigierte_fehler
                    
                expected_chunk = expected_total_payload[(paket_nr - 1) * k_bits : paket_nr * k_bits]
                bit_fehler = sum(1 for a, b in zip(decoded, expected_chunk) if a != b)
                gesamt_bit_fehler += bit_fehler
                
                # DETAILS IN DEN ZWISCHENSPEICHER SCHREIBEN
                detail_str = f"--- Paket {paket_nr} ---\n"
                detail_str += f"Statistik pro Paket: LDPC hat {korrigierte_fehler} Fehler repariert | Unkorrigierbare Bitfehler (BER): {bit_fehler}\n\n"
                detail_str += f"Decodierte Bits:\n{decoded.tolist()}\n\n"
                detail_str += f"Empfangene Symbole:\n{pay_syms_rx}\n\n"
                detail_str += "-" * 40 + "\n\n"
                detailed_log.append(detail_str)

                print(f" ✓ Payload {paket_nr} erfolgreich decodiert! (Bitfehler: {bit_fehler} | Korrigiert: {korrigierte_fehler})")
                search_offset = pay_end
            else:
                print("✗ Payload außerhalb Buffer.")
                break

        # ───────────────────────────────────────────────────────────
        # DATEI SCHREIBEN: ÜBERSICHT OBEN, LANGE DATEN UNTEN
        # ───────────────────────────────────────────────────────────
        if alle_decodierten_bits:
            total_bits = len(alle_decodierten_bits)
            ber_prozent = (gesamt_bit_fehler / total_bits) * 100 if total_bits > 0 else 0
            
            with open(f"{file_name}.txt", "w", encoding="utf-8") as f:
                # 1. KOPFDATEN & ZUSAMMENFASSUNG
                f.write("="*60 + "\n")
                f.write(" SDR EMPFANGS-PROTOKOLL & BER-ANALYSE\n")
                f.write("="*60 + "\n")
                f.write(f"Datum / Zeit : {txt_timestamp}\n")
                # HIER IST DER VERWEIS AUF DIE I/Q-DATEN:
                f.write(f"I/Q-Rohdaten : {dateiname} (als komprimiertes NumPy-Array gespeichert)\n")
                f.write(f"Konfiguration: SF={SF}, Bandbreite={BANDWIDTH/1e6} MHz, Samplerate={SAMPLERATE/1e6} MSPS\n\n")
                
                f.write("GESAMT-STATISTIK:\n")
                f.write(f"Total empfangene Pakete     : {paket_nr}\n")
                f.write(f"Total empfangene Nutzbits   : {total_bits}\n")
                f.write(f"Gesamt Hardware-Fehler      : {gesamt_korrigiert} (durch LDPC erfolgreich repariert)\n")
                f.write(f"Gesamt fehlerhafte Nutzbits : {gesamt_bit_fehler} (unkorrigierbar)\n")
                f.write(f"Finale Bit Error Rate (BER) : {ber_prozent:.4f}%\n")
                f.write("="*60 + "\n\n")
                
                # 2. GENERIERTE (EMPFANGENE) PAYLOAD
                f.write("GESAMTER EMPFANGENER BITSTREAM (Decodiert):\n")
                bit_string = "".join(str(b) for b in alle_decodierten_bits)
                f.write(f"{bit_string}\n\n")
                f.write("="*60 + "\n\n")
                
                # 3. DIE LANGEN PAKET-DATENSÄTZE
                f.write("DETAILLIERTE PAKET-DATEN (Symbole & Array-Dumps):\n\n")
                for log_teil in detailed_log:
                    f.write(log_teil)

            print(f"\n💾 Analyse abgeschlossen. Log gespeichert unter: '{file_name}.txt'")
            print(f"🔗 Die zugehörigen I/Q-Rohdaten liegen in: '{dateiname}'")
            print("\n" + "="*50)
            print(f" ✓ GESAMT-STATISTIK ({total_bits} Bits)")
            print(f"   - Korrigierte Hardware-Fehler : {gesamt_korrigiert}")
            print(f"   - Verbleibende Bitfehler      : {gesamt_bit_fehler}")
            print(f"   - Bit Error Rate (BER)        : {ber_prozent:.4f}%")
            print("="*50)



    else:
        print("Ungültige Eingabe.")
