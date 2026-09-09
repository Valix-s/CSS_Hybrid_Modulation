import numpy as np
import sys
import os

def npy_to_csv(npy_filepath):
    print(f"Lade Datei: {npy_filepath}...")
    
    try:
        # 1. .npy Datei in den Arbeitsspeicher laden
        iq_data = np.load(npy_filepath)
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{npy_filepath}' wurde nicht gefunden.")
        return
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        return

    # 2. Die komplexen Zahlen in Realteil (I) und Imaginärteil (Q) aufteilen
    i_part = np.real(iq_data)
    q_part = np.imag(iq_data)

    # 3. Zu einer 2D-Matrix zusammenfügen (Spalte 1: I, Spalte 2: Q)
    csv_data = np.column_stack((i_part, q_part))

    # 4. Dateinamen für die CSV generieren (gleicher Name, andere Endung)
    base_name = os.path.splitext(npy_filepath)[0]
    csv_filepath = f"{base_name}.csv"

    print(f"Konvertiere {len(iq_data)} Samples...")
    
    # 5. Als CSV abspeichern
    # fmt='%.6f' begrenzt die Nachkommastellen, um die Dateigröße im Rahmen zu halten
    # comments='' entfernt das Standard-Raute-Zeichen (#) vor dem Header in numpy
    np.savetxt(csv_filepath, csv_data, delimiter=',', header='I,Q', comments='', fmt='%.6f')
    
    print(f"Erfolgreich gespeichert unter: {csv_filepath}")

if __name__ == "__main__":
    # Standard-Dateiname, falls das Skript ohne Argumente ausgeführt wird.
    # Passe diesen Namen an deine aktuelle Datei an.
    zieldatei = "low_noise_gain50.npy" 
    
    # Erlaubt auch die Ausführung über die Kommandozeile: 
    # python konverter.py mein_dump.npy
    if len(sys.argv) > 1:
        zieldatei = sys.argv[1]
        
    npy_to_csv(zieldatei)