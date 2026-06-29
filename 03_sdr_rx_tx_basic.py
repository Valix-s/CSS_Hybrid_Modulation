import numpy as np
import adi
import scipy.signal
import time

# ═══════════════════════════════════════════════════════════════
# KONFIGURATION
# ═══════════════════════════════════════════════════════════════
SDR_IP      = "ip:192.168.1.10"
BANDWIDTH   = 100000
SAMPLERATE  = 2000000
CENTER_FREQ = 2450000000
BUFFER_SIZE = int(SAMPLERATE * 0.01)   
SF          = 8
PAKET_BITS  = 324  # Feste Paketgröße

N_PREAMBLE_SYMBOLS = 16
preamble_symbole   = [(0, 1, 0)] * 8 + [(0, 0, 0)] * 8

# ───────────────────────────────────────────────────────────────
# KERNFUNKTIONEN
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

def finde_paket_start(rx_data, sf, bw, fs, preamble_symbols, min_abs_peak=150000000, snr_thresh=100.0, leise=False):
    num_samples = int((2**sf / bw) * fs)
    rx_data = rx_data - np.mean(rx_data)

    # 1. Such-Schlüssel aus Upchirps generieren
    n_up = sum(1 for (_, r, _) in preamble_symbols if r == 1)
    ref = generate_signal(preamble_symbols[:n_up], sf, bw, fs)

    corr = scipy.signal.correlate(rx_data, ref, mode='valid', method='fft')
    mag = np.abs(corr)
    
    idx = int(np.argmax(mag))
    max_peak = mag[idx]
    
    # 2. Check: Absoluter Pegel
    if max_peak < min_abs_peak:
        if not leise: 
            print(f"[DEBUG] Abgelehnt (Pegel zu schwach: {max_peak:.0f} < {min_abs_peak})")
        return -1

    # 3. Check: SNR-Ratio
    noise = np.median(mag)
    if noise == 0: 
        noise = 1e-9
    ratio = max_peak / noise

    if ratio < snr_thresh: 
        if not leise: 
            print(f"[DEBUG] Abgelehnt (Ratio zu schlecht: {ratio:.1f}x < {snr_thresh}x)")
        return -1

    # 4. Check: Struktur-Test (NEU)
    # Ist das laute Signal wirklich ein Chirp oder nur WLAN-Müll?
    if idx + 2 * num_samples <= len(rx_data):
        test_chunk = rx_data[idx : idx + 2 * num_samples]
        test_syms = signal_dechirp(test_chunk, sf, bw, fs)
        
        valid_chirps = 0
        for val, direction, _ in test_syms:
            # Ein echter Preamble-Upchirp hat Richtung=1 und Wert nahe 0.
            # (Wir erlauben +/- 3 Toleranz für minimalen Timing-Offset)
            if direction == 1 and (val <= 3 or val >= (2**sf - 3)):
                valid_chirps += 1
                
        if valid_chirps < 2: # Wir wollen mindestens 2 saubere Chirps sehen
            if not leise:
                print(f"[DEBUG] Abgelehnt (Struktur-Test fehlgeschlagen: Vermutlich WLAN/Bluetooth-Burst)")
            return -1

    if not leise: 
        print(f"[DEBUG] Preamble OK! Peak={max_peak:.0f}, Ratio={ratio:.1f}x")
        
    return idx

# ───────────────────────────────────────────────────────────────
# MAIN SDR SETUP & STEUERUNG
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    while True:
        print("\n" + "="*40)
        state = input(
            "Was möchtest du tun?\n"
            " [s] Einmalig senden (1 Paket, 324 Bits)\n"
            " [e] Manuell empfangen (Strg+C = stopp + decodieren)\n"
            " [q] Beenden\n"
            "Deine Wahl: "
        ).strip().lower()

        # ── BEENDEN ───────────────────────────────────────────────────────────
        if state == "q":
            print("Tschüss!")
            break

        # ── EINMALIG SENDEN (324 Bits) ────────────────────────────────────────
        elif state == "s":
            print(f"\n--- SENDE {PAKET_BITS} BITS ---")
            eingabe = input(f"Bits eingeben (werden auf {PAKET_BITS} Bits aufgefüllt/gekürzt): ")
            
            try:
                bits = [int(b) for b in eingabe if b in '01']
            except ValueError:
                print("[Fehler] Nur 0 und 1 erlaubt.")
                continue
                
            # Exakt auf 324 Bits trimmen oder auffüllen
            if len(bits) > PAKET_BITS:
                bits = bits[:PAKET_BITS]
            elif len(bits) < PAKET_BITS:
                bits.extend([0] * (PAKET_BITS - len(bits)))
                
            symbole = preamble_symbole + bits_zu_symbolen(bits, SF)
            tx_signal = generate_signal(symbole, SF, BANDWIDTH, SAMPLERATE)
            
            print(f"Sende {len(tx_signal)} Samples...")
            sdr.tx(tx_signal)
            print("Gesendet!")

        # ── MANUELL EMPFANGEN ─────────────────────────────────────────────────
        elif state == "e":
            print("\n--- MANUELL EMPFANGEN ---")
            print("Empfange... Drücke Strg+C um zu stoppen und zu decodieren.\n")
            rx_buffers = []
            
            try:
                while True:
                    rx_buffers.append(sdr.rx())
                    print(f"  {len(rx_buffers)} Buffer gesammelt...", end='\r')
            except KeyboardInterrupt:
                pass

            print(f"\n\n{len(rx_buffers)} Buffer gesammelt → decodiere...")

            if not rx_buffers:
                print("Keine Daten.")
                continue

            full_rx = np.concatenate(rx_buffers)
            
            # Ohne Kalibrierung setzen wir den Schwellenwert auf None 
            # (bzw. auf einen kleinen Mindestwert wie 1e5, falls Rauschen triggert)
            p_start = finde_paket_start(full_rx, SF, BANDWIDTH, SAMPLERATE, preamble_symbole, min_abs_peak=5000000, snr_thresh=100.0, leise=False)

            if p_start == -1:
                print("✗ Kein Paket gefunden.")
                continue

            print("\n[Paket] Decodiere Payload...")

            num_s      = int((2**SF / BANDWIDTH) * SAMPLERATE)
            pre_len    = N_PREAMBLE_SYMBOLS * num_s
            
            # Wieviele Symbole entsprechen 324 Bits?
            block_size = SF + 3
            pay_syms   = int(np.ceil(PAKET_BITS / float(block_size)))
            pay_len    = pay_syms * num_s

            pay_start = p_start + pre_len
            pay_end   = pay_start + pay_len
            
            if pay_end <= len(full_rx):
                pay_syms_rx = signal_dechirp(full_rx[pay_start:pay_end], SF, BANDWIDTH, SAMPLERATE)
                
                # Simples Demapping (ersetzt LDPC-Decodierung)
                decoded_bits = []
                for lora_wert, richtung, qpsk_wert in pay_syms_rx:
                    # Lora Wert (SF Bits)
                    decoded_bits.extend([int(b) for b in format(lora_wert, f'0{SF}b')])
                    # Richtung (1 Bit)
                    decoded_bits.append(int(richtung))
                    # QPSK Wert (2 Bits)
                    decoded_bits.extend([int(b) for b in format(qpsk_wert, '02b')])
                    
                # Auf exakt 324 Bits kappen (falls durch Symbolgrenzen etwas Überschuss entstand)
                decoded_bits = decoded_bits[:PAKET_BITS]
                
                print(f" ✓ Payload erfolgreich decodiert ({len(decoded_bits)} Bits)!")
                print("="*50)
                print(f"Bits: {decoded_bits}")
                print("="*50)
            else:
                print("✗ Paket abgeschnitten – Puffer endete während der Payload.")

        else:
            print("Ungültige Eingabe.")
