# Custom Hybrid LoRa/CSS + SSK + DQPSK SDR Modem

> **Maturaarbeit (Schweizer Gymnasium)** von Valentin Seeholzer, Kantonsschule Reussbühl, Klasse K23a, betreut durch Herrn Daniel Zurmühle.

Dieses Repository enthält die vollständige Maturaarbeit inklusive Python-Implementierung eines eigenen physikalischen Übertragungsverfahrens (PHY) für Software Defined Radios (SDR). Untersucht wird, ob sich die Robustheit von **Chirp Spread Spectrum (CSS)** mit einer höheren Informationsdichte pro Symbol kombinieren lässt, indem zusätzlich in **Chirp-Richtung (Slope-Shift Keying, SSK)** und **differenzieller Phase (DQPSK)** codiert wird. Das System wurde auf zwei AD9363/Zynq-7020-basierten SDR-Plattformen entwickelt und unter realen Funkbedingungen im Freifeld getestet.

Die vollständige schriftliche Arbeit befindet sich in **[`Maturarbeit.pdf`](./Maturarbeit.pdf)**. Dieses README beschreibt nur den Aufbau des Repositoriums und die Software.

Finale Konfiguration:

- **2.45 GHz** Trägerfrequenz
- **1 MHz** Kanalbandbreite
- **Spreading Factor SF5**
- **CSS + SSK + DQPSK** Hybridmodulation
- **LDPC-Codierung, Rate 1/2, n = 648**
- GPS-synchronisierte **TDMA**-Burst-Übertragung

Theoretischer Netto-Datendurchsatz: ca. **104.4 kbit/s**. Der gemessene End-to-End-Durchsatz des Python-Prototyps liegt aufgrund von Software-Verarbeitung und TDMA-Overhead deutlich darunter (Details siehe Kapitel 5 der Arbeit).

---

## Repository-Struktur

```
CSS_Hybrid_Modulation/
├── Maturarbeit.pdf                  ← Die eigentliche Maturaarbeit (Hauptdokument)
├── final_hybrid_mod_tester.py       ← Finales Skript zur Messdatenerhebung (siehe unten)
├── README.md
│
├── documents/                       ← Begleitdokumente des Arbeitsprozesses
│   ├── Disposition_Maturaarbeit.pdf     Ursprüngliches Konzept/Projektskizze
│   ├── Anforderungsanalyse.pdf          Anforderungsanalyse zu Projektbeginn
│   ├── Zwischenbericht.pdf              Zwischenbericht während der Bearbeitung
│   └── Arbeitsjournal.pdf               Laufendes Arbeitsjournal
│
├── simulation_and_codes/            ← Gesamter Entwicklungsprozess der Software
│   ├── Basis_Simulation.py              Erste Simulation: LDPC + reines CSS (ohne Hardware)
│   ├── Final_SNR_Simulation.py          Simulation der Hybridmodulation über AWGN-Kanal (BER vs. SNR)
│   ├── Live_Simulation.py               Simulation mit "live" generiertem Bitstrom
│   ├── hybrid_mod_full_transceiver.py   Einfaches Sende-/Empfangssystem für den manuellen Betrieb auf echter Hardware
│   └── tools/                           Hilfsskripte für Debugging/Analyse (siehe unten)
│
└── results/                         ← Messdaten
    ├── final/                           Für die Arbeit verwendete, finale Messreihen (Reichweitentests)
    └── dump/                            Rohdaten/Protokolle aus der Testphase (nicht in der Arbeit ausgewertet)
```

---

## Entwicklungsprozess der Software

Die Skripte in `simulation_and_codes/` entstanden nicht parallel, sondern als **aufeinander aufbauender Entwicklungsprozess**. Jedes Skript war ein Zwischenschritt, um Schritt für Schritt vom reinen Simulationsmodell zum realen, hardwarebasierten System zu kommen:

1. **`Basis_Simulation.py`** – Erster Prototyp rein in Software: LDPC-Codierung/Decodierung und einfaches CSS-Symbol-Mapping, ohne Kanalmodell und ohne Hardware. Diente dazu, den Signalverarbeitungskern (Encoder, Decoder, Chirp-Erzeugung) grundsätzlich zum Laufen zu bringen.
2. **`Final_SNR_Simulation.py`** – Erweiterung um das vollständige Hybrid-Mapping (CSS + SSK + DQPSK) und ein AWGN-Kanalmodell, um die theoretische BER-Performance über verschiedene SNR-Werte zu untersuchen, bevor überhaupt reale Hardware eingesetzt wurde.
3. **`Live_Simulation.py`** – Simulation mit dynamisch/"live" erzeugtem Bitstrom anstelle fester Testdaten, als Vorbereitung auf den Betrieb mit tatsächlich kontinuierlich anfallenden Nutzdaten.
4. **`hybrid_mod_full_transceiver.py`** – Erste vollständige Implementierung auf echter SDR-Hardware (AD9363/Pluto-kompatibel). Enthält das komplette Sende-/Empfangssystem mit CFO-Korrektur, Paketerkennung und TDMA-Logik.
5. **`final_hybrid_mod_tester.py`** – Letzte, für die Messkampagne verwendete Version.

**Nur die letzten zwei Skripte sind für die Reproduktion der Arbeit relevant, alle vorangehenden Simulationen waren reine Entwicklungsschritte:**

- **`simulation_and_codes/hybrid_mod_full_transceiver.py`**
  Einfaches, interaktives Sende-/Empfangssystem. Damit lässt sich das System manuell bedienen: eigene Bits eingeben, senden und auf der Gegenstelle empfangen. Gedacht zum Ausprobieren und Verifizieren der Übertragungskette, nicht zur systematischen Datenerhebung.

- **`final_hybrid_mod_tester.py`** *(Root-Verzeichnis)*
  Das eigentliche Testskript, mit dem sämtliche in der Maturaarbeit ausgewerteten Messreihen erzeugt wurden (Reichweitentests, BER-/Paketverlust-Statistiken). Es sendet/empfängt automatisiert eine definierte Anzahl Pakete pro Runde und schreibt die Resultate direkt als CSV in `results/final/`. **Dieses Skript ist für die Nachvollziehbarkeit der Resultate in Kapitel 4 der Arbeit massgebend.**

### Tools (`simulation_and_codes/tools/`)

Die Skripte `A1_iq_analyzer.py` bis `A7_gps_test.py` sind **KI-generierte Hilfswerkzeuge** und **nicht Teil des eigentlichen Modem-Systems**. Sie wurden gezielt eingesetzt, um während der Entwicklung einzelne Teilprobleme schneller zu verstehen und zu debuggen, z. B.:

- Visualisierung aufgezeichneter I/Q-Rohdaten (`A1_iq_analyzer.py`)
- Live-Spektrumsanzeige des SDR (`A2_spectrum_live.py`)
- Konvertierung von `.npy`-IQ-Dumps nach CSV (`A3_npy_to_csv.py`)
- Berechnung des CFO anhand eines Dauertons (`A4_cw_cfo_calcu.py`)
- Überprüfung auf Buffer-Overruns (`A6_overrun.py`)
- Test der GPS/PPS-Zeitsynchronisation über ESP32 (`A7_gps_test.py`)

Diese Tools waren reine Mittel zum Zweck während der Fehlersuche und flossen nicht direkt in die finalen Messresultate ein.

---

## Ergebnisse (`results/`)

- **`results/final/`** – Die tatsächlich in der Arbeit verwendeten und ausgewerteten Messdaten (Pakete, Runden, Zusammenfassungen) je Distanz, Teststrecke und Gerät.
- **`results/dump/`** – Rohe Log-Dateien und Zwischenstände aus der Testphase (u. a. Sende-/Empfangsprotokolle, Site-Survey-Screenshots). Diese Daten wurden **nicht** in die Auswertung der Arbeit übernommen und dienen nur der Nachvollziehbarkeit des Entwicklungsprozesses.

---

## Key Features

- **Hybride Modulation:** Kombiniert CSS-Symbolverschiebung mit Chirp-Richtung (SSK) und DQPSK-Phaseninformation – **8 Bit pro Symbol bei SF5**.
- **LDPC-Fehlerkorrektur:** Rate-1/2-LDPC-Encoder und Belief-Propagation-Decoder nach **IEEE-802.11n**-Struktur.
- **Eigene SDR-PHY:** Vollständige Sende-/Empfangskette inkl. Modulation, Demodulation, Paketerkennung, CFO-Schätzung/-Korrektur und LDPC-Decodierung.
- **CFO-Kompensation:** Schätzt den Trägerfrequenzversatz aus der Paket-Präambel und korrigiert ihn digital.
- **TDMA-Burst-Betrieb:** GPS/PPS-basierte Zeitsynchronisation zur Trennung von Sende- und Verarbeitungsphase.
- **MIMO-Ready:** Gehäuse und Antennenlayout sind für spätere 2×2-MIMO-Untersuchungen vorbereitet; die finalen Messungen nutzen eine SISO-Verbindung.
- **Hardware-in-the-Loop:** Erzeugt und verarbeitet komplexe I/Q-Basisbandsignale für AD9363/Zynq-basierte SDR-Hardware.

---

## Experimentelle Ergebnisse

Der Prototyp wurde auf mehreren Aussen-Teststrecken zwischen **20 m und 200 m** untersucht. Unter dem definierten Kriterium **≤5 % BER und ≤5 % Paketverlust** blieb die beste gemessene Aussenverbindung auf einer der Teststrecken bis **100 m** praktisch nutzbar.

Die ursprünglich angestrebte Kombination aus mehreren hundert Metern Reichweite und über 10 kbit/s End-to-End-Durchsatz wurde nicht erreicht. Die Versuche halfen jedoch, praktische Limitierungen zu identifizieren:

- Latenz durch Python-Verarbeitung
- Fehler bei der Paketerkennung
- CFO-Schätzfehler bei niedrigem SNR
- Kanalasymmetrie
- reale Interferenzen und Mehrwegeausbreitung

---

## Hardware-Anforderungen

Für einen physischen Nachbau des Testaufbaus wurde folgende Hardware verwendet:

- **2× AD9363/Zynq-7020-basierte SDRs** (z. B. OpenSourceSDRLab 7020-SDR o. ä.)
- **2× 4-Antennen-Konfigurationen** für die SDRs
- **2.4-GHz-Antennen**, ca. **3 dBi**
- SMA-/Koaxialverbindungen passend zur gewählten SDR-Hardware
- Optional: **GPS/PPS-Synchronisationshardware** für den TDMA-Testmodus

---

## Software-Abhängigkeiten

Die Modem- und DSP-Pipeline ist in Python implementiert. Hauptabhängigkeiten installieren mit:

```
pip install numpy scipy matplotlib ldpc
```

Für den Betrieb mit echter SDR-Hardware zusätzlich:

```
pip install pyadi-iio pyserial
```

Verwendete Bibliotheken:

- **NumPy** für numerische/vektorisierte Verarbeitung
- **SciPy** für Signalverarbeitungsfunktionen
- **Matplotlib** für Analyse und Visualisierung
- **ldpc** für Kanalcodierung/-decodierung
- **pyadi-iio** zur Ansteuerung der AD9363/Pluto-Hardware
- **pyserial** für die serielle ESP32/GPS-Kommunikation

---

## Status

**Forschungsprototyp / experimentelles Projekt**

Die aktuelle Implementierung ist für Experimente, Ausbildung und Weiterentwicklung gedacht, nicht für den kommerziellen Einsatz. Mögliche Weiterentwicklungen: FPGA-basierte Beschleunigung, verbesserte Synchronisation/Paketerkennung, adaptive Modulation sowie ein kontrollierter Vergleich mit etablierten Systemen wie LoRa und WLAN.

---

## Lizenz

Dieses Projekt ist Open Source. Der Code darf verwendet, untersucht und verändert werden.
