import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# HIER DEINEN DATEINAMEN EINTRAGEN
# =================================================================
DATEINAME = "iq_dump_20260730_133830.npy"  
SAMPLERATE = 2000000  # 25 MHz

print(f"Lade Rohdaten aus: {DATEINAME} ...")
try:
    data = np.load(DATEINAME)
except FileNotFoundError:
    print("Fehler: Datei nicht gefunden. Hast du den Namen richtig kopiert?")
    exit()

dauer_ms = (len(data) / SAMPLERATE) * 1000
print(f"Erfolgreich geladen: {len(data)} Samples ({dauer_ms:.1f} Millisekunden)")

# 1. Berechne die absolute Magnitude (Amplitude)
mag = np.abs(data)

# 2. Erstelle das Diagramm-Fenster
plt.figure(figsize=(12, 8))

# --- PLOT 1: Zeitbereich (Amplitude) ---
plt.subplot(2, 1, 1)
t = np.arange(len(mag)) / SAMPLERATE * 1000  # Zeitachse in Millisekunden
plt.plot(t, mag, color='blue', linewidth=0.5)
plt.title("Zeitbereich: Signalamplitude (Wo sind die Bursts?)")
plt.xlabel("Zeit (ms)")
plt.ylabel("Magnitude (Rohwerte)")
plt.grid(True)

# --- PLOT 2: Spektrogramm (Frequenzverlauf) ---
plt.subplot(2, 1, 2)
plt.title("Spektrogramm (Sind die LoRa-Chirps sichtbar?)")
# NFFT bestimmt die Auflösung. Bei 28 MHz Samplerate brauchen wir ein grosses Fenster
plt.specgram(data, NFFT=2048, Fs=SAMPLERATE/1e6, cmap='inferno')
plt.xlabel("Zeit (ms)")
plt.ylabel("Frequenz (MHz relativ zur Mitte)")

plt.tight_layout()
plt.show()