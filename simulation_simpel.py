import numpy as np

# ───────────────────────────────────────────────────────────────
# KERNFUNKTIONEN (Reines NumPy, kein SciPy mehr!)
# ───────────────────────────────────────────────────────────────

def bits_zu_symbolen(bit_liste, sf=8):
    if isinstance(bit_liste, np.ndarray): bit_liste = bit_liste.tolist()
    block_size = sf + 3 
    while len(bit_liste) % block_size != 0: bit_liste.append(0)
    out = []
    for i in range(0, len(bit_liste), block_size):
        b = bit_liste[i:i+block_size]
        out.append((int(''.join(str(x) for x in b[0:sf]), 2), b[sf], int(''.join(str(x) for x in b[sf+1:block_size]), 2)))
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
        if richtung == 0: chirp = np.conj(chirp)
        chirp = chirp * np.exp(1j * [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4][qpsk_wert])
        chain.append(chirp)
    return (np.concatenate(chain) * (2**14 * 0.5)).astype(np.complex64)

def signal_dechirp(full_signal, sf, bw, fs):
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    phase = 2 * np.pi * (0.5 * k * t**2 + (-bw/2) * t)
    base_up, base_down = np.cos(phase) + 1j * np.sin(phase), np.conj(np.cos(phase) + 1j * np.sin(phase))
    window = np.hanning(num_samples)
    decoded = []
    for i in range(len(full_signal) // num_samples):
        chunk = full_signal[i*num_samples:(i+1)*num_samples]
        dc_up, dc_dn = (chunk * base_down) * window, (chunk * base_up) * window
        fft_up, fft_dn = np.abs(np.fft.fft(dc_up)), np.abs(np.fft.fft(dc_dn))
        pos_up, pos_dn = np.argmax(fft_up), np.argmax(fft_dn)
        
        if fft_up[pos_up] > fft_dn[pos_dn]: d, idx, pc = 1, pos_up, np.fft.fft(dc_up)[pos_up]
        else:                               d, idx, pc = 0, pos_dn, np.fft.fft(dc_dn)[pos_dn]
            
        if idx > num_samples // 2: idx -= num_samples
        val = idx % (2**sf)
        if d == 0 and val != 0: val = (2**sf) - val
        
        rp, ip = pc.real, pc.imag
        if   rp >= 0 and ip >= 0: q = 0
        elif rp <  0 and ip >= 0: q = 1
        elif rp <  0 and ip <  0: q = 2
        else:                     q = 3
        decoded.append((val, d, q))
    return decoded

def finde_paket_start(rx_data, sf, bw, fs, preamble_symbols, snr_thresh=8.0, leise=False):
    """
    Manuelle Synchronisation über Dechirp + FFT (Frequenzbereich).
    Komplett ohne scipy.signal.correlate!
    """
    num_samples = int((2**sf / bw) * fs)
    n_up = sum(1 for (_, r, _) in preamble_symbols if r == 1)

    # 1. Lokalen Base-Downchirp erzeugen
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    phase = 2 * np.pi * (0.5 * k * t**2 + (-bw/2) * t)
    base_down = np.conj(np.cos(phase) + 1j * np.sin(phase))
    window = np.hanning(num_samples)

    # 2. Fenster in halben Symbol-Schritten durch das Signal schieben
    step = num_samples // 2
    
    for i in range(0, len(rx_data) - num_samples, step):
        chunk = rx_data[i : i + num_samples]
        
        # Dechirp & FFT Berechnung manuell via Numpy
        dc = (chunk * base_down) * window
        fft_mag = np.abs(np.fft.fft(dc))

        peak_idx = int(np.argmax(fft_mag))
        peak_val = fft_mag[peak_idx]

        noise = np.median(fft_mag)
        if noise == 0: noise = 1e-9
        ratio = peak_val / noise

        # 3. Struktur-Test: Haben wir eine Kette von sauberen FFT-Peaks?
        if ratio > snr_thresh:
            valid_count = 1
            
            for j in range(1, n_up):
                test_start = i + j * num_samples
                if test_start + num_samples > len(rx_data):
                    break

                test_chunk = rx_data[test_start : test_start + num_samples]
                test_dc = (test_chunk * base_down) * window
                test_fft = np.abs(np.fft.fft(test_dc))

                test_peak_idx = int(np.argmax(test_fft))
                test_ratio = test_fft[test_peak_idx] / (np.median(test_fft) + 1e-9)

                # Peak muss stark sein UND im gleichen FFT-Bin (± 4 Bins Toleranz) liegen
                if test_ratio > snr_thresh and abs(test_peak_idx - peak_idx) <= 4:
                    valid_count += 1
                else:
                    break 

            # 4. Auswertung: Kette lang genug? -> Paketstart berechnen!
            if valid_count >= max(2, n_up - 2):
                if not leise:
                    print(f"[DEBUG] FFT-Sync! {valid_count}/{n_up} Symbole. Ratio: {ratio:.1f}x | Bin: {peak_idx}")

                # Aus dem FFT-Bin den exakten Timing-Offset in Samples berechnen
                if peak_idx > num_samples // 2:
                    offset = peak_idx - num_samples
                else:
                    offset = peak_idx

                exact_start = int(i - (offset * (fs / bw)))
                return max(0, exact_start)

    return -1

# ───────────────────────────────────────────────────────────────
# SIMULATION: TX -> KANAL (Rauschen) -> RX
# ───────────────────────────────────────────────────────────────

def run_simulation(snr_db):
    print(f"\n{'='*40}\nStarte Simulation mit SNR: {snr_db} dB")
    
    # Parameter
    SF = 8
    BW = 1000000
    FS = 2000000
    PAKET_BITS = 324
    N_PREAMBLE = 16
    preamble_symbole = [(0, 1, 0)] * 8 + [(0, 0, 0)] * 8
    
    # 1. ZUFÄLLIGE BITS ERZEUGEN
    tx_bits = np.random.randint(0, 2, PAKET_BITS).tolist()
    
    # 2. MODULATION (Senden)
    symbole_tx = preamble_symbole + bits_zu_symbolen(tx_bits, SF)
    tx_signal = generate_signal(symbole_tx, SF, BW, FS)
    
    # 3. KANAL (AWGN - Rauschen)
    signal_power = np.mean(np.abs(tx_signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    
    puffer = np.zeros(len(tx_signal) * 2, dtype=np.complex64)
    start_idx = np.random.randint(0, len(tx_signal) // 2)
    
    puffer[start_idx : start_idx + len(tx_signal)] = tx_signal
    
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(puffer)) + 1j * np.random.randn(len(puffer)))
    rx_signal = puffer + noise
    
    # 4. EMPFANGEN (Synchronisation via FFT)
    p_start = finde_paket_start(rx_signal, SF, BW, FS, preamble_symbole, snr_thresh=8.0, leise=True)
    
    if p_start == -1:
        print(" ✗ FEHLER: Paket im Rauschen verloren (Preamble nicht gefunden)!")
        return
        
    print(f" ✓ Paket gefunden. (Echter Start: {start_idx}, Geschätzt: {p_start}, Fehler: {p_start - start_idx} Samples)")
    
    # 5. DEMODULATION
    num_s = int((2**SF / BW) * FS)
    pre_len = N_PREAMBLE * num_s
    
    block_size = SF + 3
    pay_syms = int(np.ceil(PAKET_BITS / float(block_size)))
    pay_len = pay_syms * num_s
    
    pay_start = p_start + pre_len
    pay_end = pay_start + pay_len
    
    pay_syms_rx = signal_dechirp(rx_signal[pay_start:pay_end], SF, BW, FS)
    
    rx_bits = []
    for lora_wert, richtung, qpsk_wert in pay_syms_rx:
        rx_bits.extend([int(b) for b in format(lora_wert, f'0{SF}b')])
        rx_bits.append(int(richtung))
        rx_bits.extend([int(b) for b in format(qpsk_wert, '02b')])
        
    rx_bits = rx_bits[:PAKET_BITS] 
    
    # 6. AUSWERTUNG
    errors = sum(1 for a, b in zip(tx_bits, rx_bits) if a != b)
    ber = errors / PAKET_BITS
    
    print(f" ✓ Decodiert: {len(rx_bits)} Bits")
    if errors == 0:
        print(" ✓ 100% Fehlerfrei!")
    else:
        print(f" ⚠ Bitfehler: {errors} (BER: {ber*100:.2f}%)")

# =======================================================
# SIMULATIONS-LAUF
# =======================================================
if __name__ == "__main__":
    test_snrs = [10, 5, 0, -5, -8, -10]
    
    for snr in test_snrs:
        run_simulation(snr)