import adi
import numpy as np
import time

SDR_IP = "ip:192.168.1.10"  # ← HIER IP VON LAPTOP 2 ANPASSEN
SAMPLERATE = 2000000        # 2 MHz
CENTER_FREQ = int(2450e6)   # 2.45 GHz
EXPECTED_TONE = 500000      # Wir erwarten das Signal bei exakt +500 kHz
BUFFER_SIZE = 200000        # Grosser Puffer = Feine FFT-Auflösung (10 Hz pro Bin)

print("Verbinde mit Empfänger-Pluto...")
sdr = adi.Pluto(SDR_IP)
sdr.sample_rate = SAMPLERATE
sdr.rx_lo = CENTER_FREQ
sdr.rx_rf_bandwidth = SAMPLERATE
sdr.rx_buffer_size = BUFFER_SIZE
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 20

# Puffer freiräumen
for _ in range(5):
    sdr.rx()

print("\n" + "="*50)
print(f" FREQUENZDRIFT-ANALYSE (CFO)")
print(f" Erwartete Frequenz: {EXPECTED_TONE/1000} kHz Offset")
print(f" Auflösung: {SAMPLERATE/BUFFER_SIZE:.1f} Hz pro Messung")
print("="*50 + "\n")

try:
    while True:
        # 1. Daten empfangen
        rx_data = sdr.rx()
        
        # 2. Windowing (Blackman-Fenster unterdrückt "Verschmieren" der FFT)
        window = np.blackman(len(rx_data))
        windowed_data = rx_data * window
        
        # 3. FFT berechnen
        fft_data = np.fft.fftshift(np.fft.fft(windowed_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(rx_data), 1/SAMPLERATE))
        
        # 4. Suchfenster einschränken! 
        # Wir suchen den Peak nur zwischen +400 kHz und +600 kHz.
        # Das verhindert, dass das Skript versehentlich den DC-Spike (0 Hz)
        # oder ein starkes WLAN-Signal findet.
        search_min = 400000
        search_min = 400000
        search_max = 600000
        
        # Indizes für das Suchfenster finden
        valid_indices = np.where((freqs >= search_min) & (freqs <= search_max))[0]
        
        # Nur in diesem Bereich den stärksten Peak suchen
        fft_search_area = np.abs(fft_data[valid_indices])
        local_peak_idx = np.argmax(fft_search_area)
        
        # Den tatsächlichen Index im grossen Array zurückrechnen
        global_peak_idx = valid_indices[local_peak_idx]
        
        # 5. Frequenz auslesen und Drift berechnen
        measured_freq = freqs[global_peak_idx]
        drift = measured_freq - EXPECTED_TONE
        
        # Ausgabe
        if drift > 0:
            richtung = "zu HOCH"
        else:
            richtung = "zu TIEF"
            
        print(f"Peak gefunden bei: {measured_freq/1000:>7.3f} kHz  |  Drift: {abs(drift):>6.1f} Hz {richtung}")
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nMessung beendet.")