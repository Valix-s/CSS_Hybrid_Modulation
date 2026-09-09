import numpy as np
import adi
import matplotlib.pyplot as plt
import warnings

# Warnungen unterdrücken
warnings.filterwarnings('ignore')

# ===================================================================
# 1. EINSTELLUNGEN
# ===================================================================
SDR_IP = "ip:192.168.1.10"    
CENTER_FREQ = int(2450000000) # 2.4 GHz
SAMPLERATE = int(10000000)    # 10 MHz
BANDWIDTH = int(10000000)     

FFT_SIZE = 1024               
# NEU: Wir holen viel mehr Daten auf einmal (z.B. 50 FFTs am Stück)
# Bei 10 MSps entsprechen 51.200 Samples ca. 5,1 Millisekunden lückenloser Aufnahme
FFT_CHUNKS = 40
BUFFER_SIZE = FFT_SIZE * FFT_CHUNKS 
WATERFALL_ROWS = 100

# ===================================================================
# 2. SDR SETUP
# ===================================================================
print(f"Verbinde mit SDR auf {SDR_IP}...")
try:
    sdr = adi.Pluto(SDR_IP)
    sdr.sample_rate = SAMPLERATE
    sdr.rx_lo = CENTER_FREQ
    sdr.rx_rf_bandwidth = BANDWIDTH
    sdr.rx_buffer_size = BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 50  
    
    sdr.rx() 
    print(f"SDR verbunden! Analysiere {SAMPLERATE/1e6} MHz Bandbreite...")
except Exception as e:
    print(f"Fehler bei der SDR-Verbindung: {e}")
    exit(1)

# ===================================================================
# 3. PLOT SETUP
# ===================================================================
plt.ion()
fig, (ax_spec, ax_water) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [1, 2]})
fig.patch.set_facecolor('#1e1e1e')
fig.canvas.manager.set_window_title('2.4 GHz Kombi-Analysator (Burst-Catcher)')

freqs_mhz = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/SAMPLERATE)) / 1e6

# --- OBERER PLOT: SPEKTRUMANALYSATOR ---
ax_spec.set_facecolor('#1e1e1e')
# NEU: Zwei Linien - Eine für Live-Daten, eine für Max-Hold
line, = ax_spec.plot(freqs_mhz, np.full(FFT_SIZE, -100), color='#00ffcc', linewidth=1.0, alpha=0.7, label='Live')
line_max, = ax_spec.plot(freqs_mhz, np.full(FFT_SIZE, -100), color='#ff0044', linewidth=1.5, label='Max Hold')

ax_spec.set_title(f"Live Spektrum (Mitte: {CENTER_FREQ/1e9} GHz)", color='white', fontweight='bold')
ax_spec.set_ylabel("dBFS", color='white')
ax_spec.set_ylim(-80, 0)
ax_spec.set_xlim(freqs_mhz[0], freqs_mhz[-1])
ax_spec.grid(color='gray', linestyle='--', alpha=0.5)
ax_spec.tick_params(colors='white')
ax_spec.legend(loc='upper right', facecolor='#1e1e1e', labelcolor='white')

# --- UNTERER PLOT: WASSERFALL ---
waterfall_data = np.full((WATERFALL_ROWS, FFT_SIZE), -90.0) 
ax_water.set_facecolor('#1e1e1e')
img = ax_water.imshow(waterfall_data, aspect='auto', cmap='inferno', 
                      vmin=-80, vmax=-20, # Grenzen leicht angepasst für besseren Kontrast
                      extent=[freqs_mhz[0], freqs_mhz[-1], WATERFALL_ROWS, 0])
ax_water.set_xlabel("Frequenz-Offset [MHz]", color='white', fontsize=12)
ax_water.set_ylabel("Zeitverlauf", color='white')
ax_water.tick_params(colors='white')

plt.tight_layout()
window = np.hanning(FFT_SIZE)

# Speichervariable für Max Hold
max_hold_data = np.full(FFT_SIZE, -120.0)

# ===================================================================
# 4. LIVE-SCHLEIFE
# ===================================================================
print("\n[INFO] Fenster geöffnet. Schließe das Plot-Fenster zum Beenden.")

try:
    while plt.fignum_exists(fig.number):
        # 1. Großen Puffer holen
        rx_data = sdr.rx()
        
        # 2. Daten umformen: Mache aus dem 1D-Array eine Matrix (FFT_CHUNKS x FFT_SIZE)
        # Dadurch verwerfen wir keine Daten mehr!
        chunks = rx_data[:FFT_SIZE * FFT_CHUNKS].reshape((FFT_CHUNKS, FFT_SIZE)) / 2048.0
        
        # 3. FFT über alle Chunks gleichzeitig berechnen (Vektorisierung ist super schnell)
        chunks_windowed = chunks * window
        fft_complex = np.fft.fft(chunks_windowed, axis=1) / FFT_SIZE
        fft_shifted = np.fft.fftshift(fft_complex, axes=1)
        
        # 4. In dBFS umrechnen (Matrix mit allen FFTs)
        psd_db_matrix = 10 * np.log10(np.abs(fft_shifted)**2 + 1e-12)
        
        # 5. DEN STÄRKSTEN PEAK FINDEN (Max über die Zeitachse dieses Puffers)
        psd_db_live = np.max(psd_db_matrix, axis=0)
        
        # 6. Globalen Max-Hold aktualisieren
        max_hold_data = np.maximum(max_hold_data, psd_db_live)
        
        # 7. Plots aktualisieren
        line.set_ydata(psd_db_live)
        line_max.set_ydata(max_hold_data)
        
        waterfall_data = np.roll(waterfall_data, -1, axis=0)
        waterfall_data[-1, :] = psd_db_live
        img.set_data(waterfall_data)
        
        # 8. Neu zeichnen
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\nAbgebrochen.")
finally:
    if 'sdr' in locals():
        del sdr
    plt.close('all')
    print("SDR getrennt.")