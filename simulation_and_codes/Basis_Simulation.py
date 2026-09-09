import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix 
from ldpc import BpDecoder

# Testdaten
bits = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]

# System-Konstanten
SPREADING_FACTOR = 12
BANDWIDTH = 125000.0
SAMPLERATE = 250000

# Dauer eines Chirps ermitteln, um Puffergrössen korrekt zu dimensionieren
T_SYM = (2**SPREADING_FACTOR) / BANDWIDTH


def setup_matrix(Z: int = 27) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt die Generatormatrix G und Parity-Check-Matrix H 
    fuer den IEEE 802.11n Standard (Rate 1/2).

    Argumente:
        Z (int): Expansionsfaktor. Bestimmt die finale Matrixgroesse.

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

    # Systematische Form herstellen (Gauß-Elimination)
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
    Codiert eine Bitfolge mithilfe der Generatormatrix G (LDPC).

    Argumente:
        bits (list): Die uncodierten Informationsbits.
        G (np.ndarray): Die Generatormatrix.

    Rückgabewerte:
        np.ndarray: Das fehlerkorrigierbare Codewort.
    """
    bits = np.array(bits, dtype=int)
    k = G.shape[0]
    
    if len(bits) != k:
        print(f"ACHTUNG: Falsche Bitlänge! Habe {len(bits)}, brauche {k}.")
        return None 
        
    # Matrixmultiplikation modulo 2 (GF(2) Arithmetik) erzwingt binaere Werte
    codeword = np.dot(bits, G) % 2
    return codeword


def bits_zu_symbolen(bit_liste: list) -> list:
    """
    Mappt einen binären Bitstrom auf LoRa-Symbole. 
    Jedes Symbol enthaelt 1 Richtungs-Bit und 7 Wert-Bits (SF7).

    Argumente:
        bit_liste (list): Der eingehende Bitstrom.

    Rückgabewerte:
        list: Eine Liste von Tupeln im Format (Wert, Richtung).
    """
    symbol_blocks = []

    if isinstance(bit_liste, np.ndarray):
        bit_liste = bit_liste.tolist()

    # Padding erzwingt vollstaendige 8-Bit-Bloecke fuer sauberes Mapping
    while len(bit_liste) % 8 != 0:
        bit_liste.append(0)

    for i in range(0, len(bit_liste), 8):
        block = bit_liste[i : i + 8]
        
        bit_richtung = block[0]       
        bits_wert = block[1:8]       
        
        bit_string = "".join(str(b) for b in bits_wert)
        symbol_wert = int(bit_string, 2)
        
        symbol_blocks.append((symbol_wert, bit_richtung))

    return symbol_blocks


def generate_signal(
    symbole: list, 
    sf: int, 
    bw: float, 
    fs: int
) -> np.ndarray:
    """
    Erzeugt das sendebereite I/Q-Signal basierend auf den zu sendenden Symbolen.

    Argumente:
        symbole (list): Liste der zu modulierenden Symbole.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (int): Samplerate in Hz.

    Rückgabewerte:
        np.ndarray: Komplexe I/Q-Samples fuer das SDR (Pluto).
    """
    num_samples = int((2**sf / bw) * fs)
    t = np.arange(num_samples) / fs
    
    k = bw / ((2**sf) / bw) 
    f_start = -bw / 2        
    
    phase = 2 * np.pi * (0.5 * k * t**2 + f_start * t)
    base_chirp = np.cos(phase) + 1j * np.sin(phase)
    
    signal_kette = []
    
    for wert, richtung in symbole:
        chirp = base_chirp.copy()
        
        # Ein zirkulaerer Shift im Zeitbereich bewirkt den LoRa-Frequenzsprung
        shift = int((wert / 2**sf) * num_samples)
        chirp = np.roll(chirp, -shift)
        
        # Konjugieren dreht die Frequenzrichtung um (Down-Chirp)
        if richtung == 0: 
            chirp = np.conj(chirp)
            
        signal_kette.append(chirp)
        
    full_signal = np.concatenate(signal_kette)
    
    # 50% Skalierung schuetzt den DAC des SDR vor Clipping
    full_signal = full_signal * (2**14 * 0.5)
    
    return full_signal.astype(np.complex64)


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Fügt dem Signal weisses Gausssches Rauschen (AWGN) gemäss 
    dem gewünschten Signal-Rausch-Verhältnis hinzu.

    Argumente:
        signal (np.ndarray): Das ideale, rauscharme Sendesignal.
        snr_db (float): Gewuenschtes SNR in Dezibel.

    Rückgabewerte:
        np.ndarray: Das verrauschte Empfangssignal.
    """
    signal_power = np.var(signal)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Leistungsaufteilung auf Real- (I) und Imaginaerteil (Q)
    noise_std = np.sqrt(noise_power / 2)
    
    noise = (
        np.random.normal(0, noise_std, len(signal)) 
        + 1j * np.random.normal(0, noise_std, len(signal))
    )
    
    full_signal = signal + noise
    return full_signal


def print_diagramm(signal: np.ndarray, sf: int, bw: float, fs: int, num_chirps: int = 7) -> None:
    """
    Visualisiert das Spektrogramm fuer eine bestimmte Anzahl an Chirps.

    Argumente:
        signal (np.ndarray): Das zu visualisierende I/Q-Signal.
        sf (int): Der verwendete Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (int): Samplerate in Hz.
        num_chirps (int): Anzahl der Chirps, die angezeigt werden sollen.
        
    Rückgabewerte:
        None
    """
    # 1. Signal auf die gewuenschte Anzahl Chirps kuerzen
    samples_per_chirp = int((2**sf / bw) * fs)
    samples_to_plot = samples_per_chirp * num_chirps
    
    if len(signal) > samples_to_plot:
        signal_cut = signal[:samples_to_plot]
    else:
        signal_cut = signal

    plt.figure(figsize=(12, 6))

    # 2. NFFT dynamisch anpassen! 
    # Ein guter Wert ist die Haelfte der Samples eines einzelnen Chirps.
    nfft_size = max(64, samples_per_chirp // 2)
    overlap = nfft_size * 7 // 8  # Ca. 87% Ueberlappung fuer ein weiches, hochaufloesendes Bild

    plt.specgram(
        signal_cut, 
        NFFT=nfft_size, 
        Fs=fs, 
        noverlap=overlap, 
        cmap='inferno'
    )

    plt.title(f"LoRa Simulation (SF{sf}) - Erste {num_chirps} Chirps")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Frequenz [Hz]")
    
    # Die Y-Achse dynamisch knapp ueber die Bandbreite setzen
    plt.ylim(-bw/2 - 20000, bw/2 + 20000) 
    
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
    Demoduliert das Basisbandsignal in einzelne LoRa-Symbole 
    durch Multiplikation mit einem inversen Chirp und anschliessender FFT.

    Argumente:
        full_signal (np.ndarray): Das verrauschte Empfangssignal.
        sf (int): Spreading Factor.
        bw (float): Bandbreite in Hz.
        fs (int): Samplerate in Hz.

    Rückgabewerte:
        list: Liste der erkannten Symbole als Tupel (Wert, Richtung).
    """
    num_samples = int((2**sf / bw) * fs)
    
    t = np.arange(num_samples) / fs
    k = bw / ((2**sf) / bw)
    f_start = -bw / 2
    phase = 2 * np.pi * (0.5 * k * t**2 + f_start * t)
    
    base_up_chirp = np.cos(phase) + 1j * np.sin(phase)
    base_down_chirp = np.conj(base_up_chirp)
    
    # Hanning-Fenster reduziert spektrales Leakage an den Chunk-Grenzen
    window = np.hanning(num_samples)
    
    decoded_symbols = []
    num_chunks = len(full_signal) // num_samples
    
    for i in range(num_chunks):
        start = i * num_samples
        end = start + num_samples
        chunk = full_signal[start:end]
        
        dechirped_up = (chunk * base_down_chirp) * window
        fft_up = np.abs(np.fft.fft(dechirped_up))
        peak_up = np.max(fft_up)
        pos_up = np.argmax(fft_up)
        
        dechirped_down = (chunk * base_up_chirp) * window
        fft_down = np.abs(np.fft.fft(dechirped_down))
        peak_down = np.max(fft_down)
        pos_down = np.argmax(fft_down)
        
        if peak_up > peak_down:
            detected_dir = 1  
            raw_index = pos_up
        else:
            detected_dir = 0 
            raw_index = pos_down
            
        # FFT-Indizes ueber der Haelfte repraesentieren negative Frequenzen
        if raw_index > num_samples // 2:
            raw_index -= num_samples
            
        detected_value = raw_index % (2**sf)
        
        # Roll- und Conj-Operationen im Sender erfordern eine 
        # invertierte Wert-Berechnung fuer Down-Chirps
        if detected_dir == 0 and detected_value != 0:
            detected_value = (2**sf) - detected_value
        
        decoded_symbols.append((detected_value, detected_dir))
        
    return decoded_symbols


def ldpc_decode(received_symbols: list, H: np.ndarray) -> np.ndarray:
    """
    Fuehrt eine Syndrom-basierte LDPC-Decodierung auf den empfangenen 
    Symbolen durch, um Uebertragungsfehler zu korrigieren.

    Argumente:
        received_symbols (list): Empfangene Symbole.
        H (np.ndarray): Parity-Check-Matrix.

    Rückgabewerte:
        np.ndarray: Die korrigierte originale Bitsequenz.
    """
    received_bits = []
    
    for wert, richtung in received_symbols:
        received_bits.append(richtung)
        bin_str = format(wert, '07b')
        for bit_char in bin_str:
            received_bits.append(int(bit_char))
            
    n = H.shape[1]
    m = H.shape[0]
    k = n - m
    
    current_len = len(received_bits)
    if current_len > n:
        received_bits = received_bits[:n]
    elif current_len < n:
        received_bits.extend([0] * (n - current_len))

    y_raw = np.array(received_bits, dtype=int)

    # Syndrom s = H * y_raw zeigt an, welche Parity-Checks verletzt sind
    H_sparse = csr_matrix(H)
    syndrome = (H_sparse @ y_raw) % 2
    
    decoder = BpDecoder(H_sparse, error_rate=0.1, max_iter=20, bp_method='ps')
    estimated_error = decoder.decode(syndrome)
    
    corrected_codeword = (y_raw + estimated_error) % 2
    
    return corrected_codeword[:k]


# --- HAUPTPROGRAMM ---

g_matrix, h_matrix = setup_matrix()
k_size = g_matrix.shape[0] 

# Padding der Nutzdaten auf Blockgroesse k des LDPC Encoders
bits_to_send_padded = bits.copy()
while len(bits_to_send_padded) < k_size:
    bits_to_send_padded.append(0)

sending_bits = ldpc_encode(bits_to_send_padded, g_matrix)   
sending_symbols = bits_zu_symbolen(sending_bits)
print(sending_symbols)

pluto_iq = generate_signal(
    sending_symbols, 
    SPREADING_FACTOR, 
    BANDWIDTH, 
    SAMPLERATE
)

signal = add_noise(pluto_iq, -11.8)

received_symbols = signal_dechirp(
    signal, 
    SPREADING_FACTOR, 
    BANDWIDTH, 
    SAMPLERATE
)

encrypted_bits = ldpc_decode(received_symbols, h_matrix)


print(f"Dauer eines Chirps: {T_SYM} s")
print(sending_symbols)
print(pluto_iq)
print(signal)

print_diagramm(pluto_iq, SPREADING_FACTOR, BANDWIDTH, SAMPLERATE, num_chirps=7)
print_diagramm(signal, SPREADING_FACTOR, BANDWIDTH, SAMPLERATE, num_chirps=7)

print(signal_dechirp(signal, SPREADING_FACTOR, BANDWIDTH, SAMPLERATE))

print("\n--- ERGEBNIS ---")
print(f"Original (Start): {bits[:17]}")
print(f"Decoded  (Start): {encrypted_bits[:17].tolist()}")
fehler = np.sum(np.array(bits[:17]) != encrypted_bits[:17])
print(f"Bitfehler in den ersten 17 Bits: {fehler}")
