import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix 
from ldpc import BpDecoder


# --- Konstanten ---
SPREADING_FACTOR = 4
BANDWIDTH = 10000000.0
SAMPLERATE = 20000000.0
T_SYM = (2**SPREADING_FACTOR) / BANDWIDTH
SNR_START = 4
SNR_STOP = -3
SNR_SCHRITT = -1
PAKETE_PRO_SNR = 20

def random_bits_input() -> list:
    """
    Generiert einen zufaelligen Bit-Strom zur Simulation 
    einer dynamischen Datenquelle.

    Rückgabewerte:
        list: Zufaellige Sequenz aus 0 und 1 (Laenge 1 bis 399).
    """
    size = np.random.randint(1, 400)
    return np.random.randint(0, 2, size).tolist()


def bits_to_chunk() -> list: 
    """
    Coroutine (Generator), die asynchron eintreffende Bits sammelt 
    und exakt zugeschnittene Bloecke fuer den LDPC-Encoder ausgibt.

    Rückgabewerte:
        list: Eine Liste, die wiederum Listen der exakten Blockgroesse enthaelt.
    """
    k_size = 324
    bit_history = []
    out_chunk = []

    while True:
        new_bits = yield out_chunk
        out_chunk = []

        if new_bits is None: 
            continue

        bit_history.extend(new_bits)

        # Puffert die Bits, bis ein vollstaendiger Block codiert werden kann
        while len(bit_history) >= k_size:
            chunk = bit_history[:k_size]
            bit_history = bit_history[k_size:] 
            out_chunk.append(chunk)


def setup_matrix(Z: int = 27) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt die Generatormatrix G und Parity-Check-Matrix H 
    fuer den IEEE 802.11n Standard (Rate 1/2).

    Argumente:
        Z (int): Expansionsfaktor fuer die Matrixgroesse.

    Rückgabewerte:
        tuple: (Generatormatrix G, Parity-Check-Matrix H)
    """
    H_base = np.array([
        [0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0, 
         1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1, 
         -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, 
         -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1, 
         -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, 
         -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
        [24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1, 
         -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1, 
         0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, 
         -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, 
         -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
        [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, 
         -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [25, -1, -1, -1, 16, -1, -1, -1, 11, -1, 0, -1, 
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [3, -1, -1, -1, 0, -1, -1, -1, 25, -1, -1, -1, 
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
    ])

    rows, cols = H_base.shape
    H_rows = rows * Z
    H_cols = cols * Z
    row_indices, col_indices = [], []
    
    for r in range(rows):
        for c in range(cols):
            shift = H_base[r, c]
            if shift == -1: 
                continue
            for i in range(Z):
                row_indices.append(r * Z + i)
                col_indices.append(c * Z + (i + shift) % Z)
                
    data = np.ones(len(row_indices), dtype=int)
    H = coo_matrix(
        (data, (row_indices, col_indices)), 
        shape=(H_rows, H_cols)
    ).toarray()

    H_sys = H.copy()

    # Systematische Form herstellen (Gauss-Elimination)
    m, n = H_sys.shape
    k = n - m
    pivot_row = 0
    for col in range(n - m, n):
        if pivot_row >= m: 
            break
            
        if H_sys[pivot_row, col] == 0:
            for r in range(pivot_row + 1, m):
                if H_sys[r, col] == 1:
                    H_sys[[pivot_row, r]] = H_sys[[r, pivot_row]]
                    break
                    
        if H_sys[pivot_row, col] == 1:
            for r in range(m):
                if r != pivot_row and H_sys[r, col] == 1:
                    H_sys[r] = (H_sys[r] + H_sys[pivot_row]) % 2
            pivot_row += 1

    P = (H_sys[:, :k]).T
    I_k = np.eye(k, dtype=int)
    G = np.hstack((I_k, P))
    
    return G, H


def ldpc_encode(bits: list, G: np.ndarray) -> np.ndarray:
    """
    Codiert eine Bitfolge mithilfe der Generatormatrix G.

    Argumente:
        bits (list): Die uncodierten Informationsbits.
        G (np.ndarray): Die Generatormatrix.

    Rückgabewerte:
        np.ndarray: Das fehlerkorrigierbare Codewort.
    """
    bits = np.array(bits, dtype=int)
    k = G.shape[0]
    
    if len(bits) != k:
        print(f"ACHTUNG: Falsche Bitlaenge! Habe {len(bits)}, brauche {k}.")
        return None 
        
    # Matrixmultiplikation modulo 2 erzwingt binaere Werte im Codewort
    codeword = np.dot(bits, G) % 2
    return codeword


def bits_zu_symbolen(bit_liste: list) -> list:
    """
    Mappt einen binaeren Bitstrom auf hybride LoRa/QPSK-Symbole. 
    Jedes Symbol (Gesamt 7 Bits) enthaelt:
    - 4 Bits fuer den LoRa-Frequenzversatz (Werte 0-15)
    - 1 Bit fuer die Chirp-Richtung (z.B. 0=Up-Chirp, 1=Down-Chirp)
    - 2 Bits fuer die QPSK-Phase (Werte 0-3)

    Argumente:
        bit_liste (list): Der eingehende Bitstrom (Liste oder Numpy-Array).

    Rückgabewerte:
        list: Eine Liste von Tupeln im Format (lora_wert, richtung, qpsk_wert).
    """
    symbol_blocks = []

    # Falls ein Numpy-Array übergeben wird, in eine normale Liste umwandeln
    if isinstance(bit_liste, np.ndarray):
        bit_liste = bit_liste.tolist()

    # Padding erzwingt vollstaendige 7-Bit-Bloecke fuer das saubere Mapping
    while len(bit_liste) % 7 != 0:
        bit_liste.append(0)

    for i in range(0, len(bit_liste), 7):
        block = bit_liste[i : i + 7]
        
        # 1. Extrahiere die ersten 4 Bits fuer das LoRa-Symbol
        bits_lora = block[0:4]       
        lora_string = "".join(str(b) for b in bits_lora)
        lora_wert = int(lora_string, 2)
        
        # 2. Extrahiere das 1 Richtungs-Bit (Index 4)
        richtung = block[4]
        
        # 3. Extrahiere die letzten 2 Bits fuer die QPSK-Phase
        bits_qpsk = block[5:7]
        qpsk_string = "".join(str(b) for b in bits_qpsk)
        qpsk_wert = int(qpsk_string, 2)
        
        # Speichere alle drei Eigenschaften als Tupel ab
        symbol_blocks.append((lora_wert, richtung, qpsk_wert))

    return symbol_blocks


def generate_signal(
    symbole: list, 
    sf: int, 
    bw: float, 
    fs: int
) -> np.ndarray:
    """
    Erzeugt das sendebereite I/Q-Signal basierend auf den Modulationssymbolen.

    Argumente:
        symbole (list): Liste der zu modulierenden Symbole.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (int): Samplerate in Hz.

    Rückgabewerte:
        np.ndarray: Komplexe I/Q-Samples fuer den Hardware-Sender.
    """
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    
    k = bw / ((2**sf) / bw) 
    f_start = -bw / 2        
    
    phase = 2 * np.pi * (0.5 * k * t**2 + f_start * t)
    base_chirp = np.cos(phase) + 1j * np.sin(phase)
    
    signal_kette = []
    
    for lora_wert, richtung, qpsk_wert in symbole:
        chirp = base_chirp.copy()
        
        # Ein zirkulaerer Shift im Zeitbereich bewirkt den Frequenzsprung
        shift = int((lora_wert / 2**sf) * num_samples)
        chirp = np.roll(chirp, -shift)
        
        phase_err = np.angle((chirp * np.conj(base_chirp))[0])
        chirp = chirp * np.exp(-1j * phase_err)
        
        # Konjugieren invertiert die Phasenlage fuer den Down-Chirp
        if richtung == 0: 
            chirp = np.conj(chirp)
        
        if qpsk_wert == 0:
            chirp = chirp * np.exp(1j * np.pi / 4)
        elif qpsk_wert == 1:
            chirp = chirp * np.exp(1j * 3 * np.pi / 4)
        elif qpsk_wert == 2:
            chirp = chirp * np.exp(1j * 5 * np.pi / 4)
        elif qpsk_wert == 3:
            chirp = chirp * np.exp(1j * 7 * np.pi / 4)
            
        signal_kette.append(chirp)
        
    full_signal = np.concatenate(signal_kette)
    
    # Skalierung verhindert ein Clipping des DAC am Hardware-Sender
    full_signal = full_signal * (2**14 * 0.5)
    
    return full_signal.astype(np.complex64)


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Fügt dem Signal weisses Gausssches Rauschen (AWGN) hinz.

    Argumente:
        signal (np.ndarray): Das ideale Sendesignal.
        snr_db (float): Gewuenschtes SNR in Dezibel.

    Rückgabewerte:
        np.ndarray: Das verrauschte Empfangssignal.
    """
    signal_power = np.var(signal)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Leistungsaufteilung auf In-Phase- und Quadraturkomponente
    noise_std = np.sqrt(noise_power / 2)
    
    noise = (
        np.random.normal(0, noise_std, len(signal)) 
        + 1j * np.random.normal(0, noise_std, len(signal))
    )
    
    full_signal = signal + noise
    return full_signal


def print_diagramm(signal: np.ndarray, sf: int, fs: int) -> None:
    """
    Visualisiert das Spektrogramm des komplexen Basisbandsignals.

    Argumente:
        signal (np.ndarray): Das zu visualisierende I/Q-Signal.
        sf (int): Der verwendete Spreading Factor.
        fs (int): Samplerate in Hz.
        
    Rückgabewerte:
        None
    """
    plt.figure(figsize=(12, 6))

    plt.specgram(
        signal, 
        NFFT=1024, 
        Fs=fs, 
        noverlap=900, 
        cmap='inferno'
    )

    plt.title(f"LoRa Simulation (SF{sf})")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Frequenz [Hz]")
    plt.ylim(-150000, 150000)
    plt.colorbar(label="Intensitaet [dB]")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.show()


def signal_dechirp(
    full_signal: np.ndarray, 
    sf: int, 
    bw: float, 
    fs: int
) -> list:
    """
    Demoduliert das hybride Basisbandsignal in einzelne Symbole 
    durch Multiplikation mit einem inversen Chirp und anschliessender FFT.

    Argumente:
        full_signal (np.ndarray): Das verrauschte Empfangssignal.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (int): Samplerate in Hz.

    Rückgabewerte:
        list: Liste der erkannten Symbole als Tupel (lora_wert, richtung, qpsk_wert).
    """
    num_samples = int((2**sf / bw) * fs)
    
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    f_start = -bw / 2
    phase = 2 * np.pi * (0.5 * k * t**2 + f_start * t)
    
    base_up_chirp = np.cos(phase) + 1j * np.sin(phase)
    base_down_chirp = np.conj(base_up_chirp)
    
    # Hanning-Fenster reduziert spektrales Leakage an den Blockgrenzen
    window = np.hanning(num_samples)
    
    decoded_symbols = []
    num_chunks = len(full_signal) // num_samples
    
    for i in range(num_chunks):
        start = i * num_samples
        end = start + num_samples
        chunk = full_signal[start:end]
        
        # --- UP-CHIRP PFAD ---
        dechirped_up = (chunk * base_down_chirp) * window
        complex_fft_up = np.fft.fft(dechirped_up)      # 1. Komplexe FFT speichern!
        fft_up = np.abs(complex_fft_up)                # 2. Betrag nur fuer die Suche
        pos_up = np.argmax(fft_up)
        peak_up = fft_up[pos_up]
        
        # --- DOWN-CHIRP PFAD ---
        dechirped_down = (chunk * base_up_chirp) * window
        complex_fft_down = np.fft.fft(dechirped_down)  # 1. Komplexe FFT speichern!
        fft_down = np.abs(complex_fft_down)            # 2. Betrag nur fuer die Suche
        pos_down = np.argmax(fft_down)
        peak_down = fft_down[pos_down]
        
        # --- ENTSCHEIDUNG: WELCHE RICHTUNG GEWINNT? ---
        if peak_up > peak_down:
            detected_dir = 1  
            raw_index = pos_up
            peak_complex = complex_fft_up[pos_up]      # Komplexe Zahl am Peak abgreifen
        else:
            detected_dir = 0  
            raw_index = pos_down
            peak_complex = complex_fft_down[pos_down]  # Komplexe Zahl am Peak abgreifen
            
        # --- LORA WERT BERECHNEN (Wie bisher) ---
        if raw_index > num_samples // 2:
            raw_index -= num_samples
            
        detected_value = raw_index % (2**sf)
        
        if detected_dir == 0 and detected_value != 0:
            detected_value = (2**sf) - detected_value
            
        # --- QPSK WERT BERECHNEN (Der 45-Grad Trick) ---
        real_part = np.real(peak_complex)
        imag_part = np.imag(peak_complex)
        
        # Quadranten-Entscheidung (Ohne teure Winkel-Berechnung!)
        if real_part >= 0 and imag_part >= 0:
            qpsk_wert = 0  # 1. Quadrant (+, +)
        elif real_part < 0 and imag_part >= 0:
            qpsk_wert = 1  # 2. Quadrant (-, +)
        elif real_part < 0 and imag_part < 0:
            qpsk_wert = 2  # 3. Quadrant (-, -)
        else:
            qpsk_wert = 3  # 4. Quadrant (+, -)
            
        # Alles als fertiges Symbol-Tupel speichern
        decoded_symbols.append((detected_value, detected_dir, qpsk_wert))
        
    return decoded_symbols


def ldpc_decode(received_symbols: list, H: np.ndarray) -> np.ndarray:
    """
    Fuehrt eine Syndrom-basierte LDPC-Decodierung durch.

    Argumente:
        received_symbols (list): Empfangene hybride Symbole (lora_wert, richtung, qpsk_wert).
        H (np.ndarray): Parity-Check-Matrix.

    Rückgabewerte:
        np.ndarray: Die korrigierte originale Bitsequenz.
    """
    received_bits = []
    
    for lora_wert, richtung, qpsk_wert in received_symbols:
        # 1. LoRa-Wert wieder in 4 Bits aufsplitten
        lora_str = format(lora_wert, '04b')
        for bit_char in lora_str:
            received_bits.append(int(bit_char))
            
        # 2. Das eine Richtungs-Bit anhaengen
        received_bits.append(int(richtung))
        
        # 3. Den QPSK-Wert in 2 Bits aufsplitten
        qpsk_str = format(qpsk_wert, '02b')
        for bit_char in qpsk_str:
            received_bits.append(int(bit_char))
            
    n = H.shape[1]
    m = H.shape[0]
    k = n - m
    
    current_len = len(received_bits)
    
    # Längen-Korrektur (Padding/Truncating)
    if current_len > n:
        received_bits = received_bits[:n]
    elif current_len < n:
        received_bits.extend([0] * (n - current_len))

    y_raw = np.array(received_bits, dtype=int)

    # Das Syndrom verifiziert die Einhaltung der Parity-Checks
    H_sparse = csr_matrix(H)
    syndrome = (H_sparse @ y_raw) % 2
    
    # LDPC Belief-Propagation Decoder
    decoder = BpDecoder(H_sparse, error_rate=0.1, max_iter=20, bp_method='ps')
    estimated_error = decoder.decode(syndrome)
    
    corrected_codeword = (y_raw + estimated_error) % 2
    
    # Extrahiere die eigentlichen Nutzdaten (die ersten k Bits)
    return corrected_codeword[:k]


# ===================================================================
# HAUPTPROGRAMM 
# ===================================================================

g_matrix, h_matrix = setup_matrix()
instance_bits_to_chunk = bits_to_chunk()
next(instance_bits_to_chunk)

print("Starte Simulation: Teste verschiedene Rausch-Stufen (SNR)")
print("=========================================================")

# in 1er Schritten vom Start-SNR zum End-SNR
for snr_db in range(int(SNR_START), int(SNR_STOP) - 1, int(SNR_SCHRITT)):
    
    # Zaehler fuer die aktuelle Runde auf Null setzen
    pakete_gesendet = 0
    pakete_fehlerfrei = 0
    bits_gesendet = 0
    bits_falsch = 0
    
    print(f"SNR {snr_db:>3} dB | ", end="")
    
    while pakete_gesendet < PAKETE_PRO_SNR:
        input_bits = random_bits_input()
        chunks = instance_bits_to_chunk.send(input_bits)
        
        if chunks:
            for packet in chunks:
                if pakete_gesendet >= PAKETE_PRO_SNR:
                    break
                    
                # --- SENDER ---
                encoded_bits = ldpc_encode(packet, g_matrix)
                sending_symbols = bits_zu_symbolen(encoded_bits)
                sending_iq_array = generate_signal(sending_symbols, SPREADING_FACTOR, BANDWIDTH, SAMPLERATE)
                
                # --- KANAL (Rauschen) ---
                signal = add_noise(sending_iq_array, snr_db)
                
                # --- EMPFAENGER ---
                recieved_symbols = signal_dechirp(signal, SPREADING_FACTOR, BANDWIDTH, SAMPLERATE)
                decoded_bits = ldpc_decode(recieved_symbols, h_matrix)
                
                # --- AUSWERTUNG ---
                paket_laenge = len(packet)
                decoded_list = decoded_bits.astype(int).tolist()
                vergleichs_laenge = min(paket_laenge, len(decoded_list))
                
                fehler_in_diesem_paket = 0
                for i in range(vergleichs_laenge):
                    if packet[i] != decoded_list[i]:
                        fehler_in_diesem_paket += 1
                
                # Zaehler aktualisieren
                bits_gesendet += paket_laenge
                bits_falsch += fehler_in_diesem_paket
                
                if fehler_in_diesem_paket == 0:
                    pakete_fehlerfrei += 1
                    print(".", end="", flush=True) # Punkt = Richtig
                else:
                    print("X", end="", flush=True) # X = Falsch
                    
                pakete_gesendet += 1
        else:
            # Rauschen abfangen, wenn gerade keine Daten generiert wurden
            dummy_len = int(T_SYM * SAMPLERATE) * 93
            leeres_signal = np.zeros(dummy_len, dtype=np.complex64)
            signal = add_noise(leeres_signal, snr_db)

    # --- ZUSAMMENFASSUNG DRUCKEN ---
    if bits_gesendet > 0:
        ber = (bits_falsch / bits_gesendet) * 100
    else:
        ber = 0
        
    erfolgsquote = (pakete_fehlerfrei / pakete_gesendet) * 100
    
    print(f" | Erfolg: {pakete_fehlerfrei:>2}/{pakete_gesendet} ({erfolgsquote:>3.0f}%) | Bit-Fehler (BER): {ber:>5.2f}%")

print("=========================================================")
print("Simulation beendet!")
