import serial
import time

# ─────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────
ESP32_PORT = "COM15"  # <--- HIER deinen ermittelten Port eintragen!
ESP32_BAUD = 1000000

print("=" * 50)
print(f" GPS PPS TESTER")
print(f" Öffne {ESP32_PORT} mit {ESP32_BAUD} Baud...")
print("=" * 50)

try:
    # Serielle Verbindung aufbauen
    ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=2)
    print("\n✓ Erfolgreich verbunden!")
    print("Warte auf GPS-Signal... (Die LED am ATGM336H muss blinken!)")
    print("-" * 50)
    
    letzter_zeitpunkt = 0
    puls_counter = 0
    
    while True:
        # ser.read(1) blockiert das Skript, bis exakt 1 Byte vom ESP32 kommt
        daten = ser.read(1) 
        
        # Sobald Daten da sind, nehmen wir sofort die exakte Zeit
        jetzt = time.time()
        
        if daten == b'S':
            puls_counter += 1
            if letzter_zeitpunkt != 0:
                # Zeitabstand zum letzten Puls berechnen
                delta = jetzt - letzter_zeitpunkt
                
                # Jitter-Analyse für die Konsole
                abweichung_ms = (delta - 1.0) * 1000
                
                print(f"[{puls_counter}] PPS empfangen! Abstand: {delta:.4f} Sekunden | USB-Jitter: {abweichung_ms:+.1f} ms")
            else:
                print(f"[{puls_counter}] Erster PPS Puls empfangen! Synchronisiere...")
                
            letzter_zeitpunkt = jetzt

except serial.SerialException as e:
    print(f"\n✗ FEHLER: Konnte {ESP32_PORT} nicht öffnen.")
    print("  -> Stimmt die COM-Nummer?")
    print("  -> Ist der Serielle Monitor in der Arduino IDE wirklich geschlossen?")
    print(f"  -> Details: {e}")
except KeyboardInterrupt:
    print("\n\nTest durch Benutzer abgebrochen (Strg+C).")
finally:
    # Port am Ende sauber schließen, damit er für andere Skripte frei ist
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serieller Port wurde wieder freigegeben.")