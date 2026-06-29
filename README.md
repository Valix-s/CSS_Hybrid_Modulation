# High-Speed SDR-Modem: Lückenschluss zwischen WLAN und LoRa

Dieses Repository enthält den Python-Quellcode, die begleitenden Simulationen sowie die Entwicklungsdokumentation zu meiner Maturaarbeit an der Kantonsschule Reussbühl (Luzern).

**Thema der Arbeit:** Entwicklung eines lizenzfreien Funkprotokolls zur Schliessung der Bandbreiten-Reichweiten-Lücke zwischen WLAN und LoRa: Implementierung eines High-Speed SDR-Modems unter Berücksichtigung der BAKOM RIR1010-01 Regulierung.

---

##  1. Ziel der Arbeit und Modulationskonzept

Das primäre Ziel dieses Projekts ist die Entwicklung eines hybriden Kommunikationssystems, das die physikalischen Vorteile der robusten LoRa-Modulation (Chirp Spread Spectrum, CSS) mit wesentlich höheren Datenübertragungsraten kombiniert. Aktuelle Funktechnologien erzwingen meist einen Kompromiss aus hoher Datenrate bei kurzer Reichweite (WLAN) oder hoher Reichweite bei minimaler Datenrate (LoRa). 

Um diese Lücke zu schließen, operiert das entwickelte Protokoll im 2.4 GHz ISM-Band mit einer sehr hohen **Bandbreite von 10 MHz**. Das System ist auf einen niedrigen **Spreading Factor von 5 (SF5)** ausgelegt und bricht die klassischen Datenraten-Limits von LoRa durch eine Erweiterung der Modulationsebene:

1. **Slope-Shift-Keying (SSK):** Parallele Nutzung von Up- und Down-Chirps. Die Erkennung der Chirp-Richtung liefert einen Datengewinn von exakt +1 Bit pro Symbol.
2. **Phasenumtastung (QPSK):** Einbettung von zusätzlichen 2 Bits in die Phasenlage (45°-Offsets: $\pi/4, 3\pi/4, 5\pi/4, 7\pi/4$) jedes einzelnen Chirps.
3. **Kanalcodierung:** Implementierung eines Low-Density Parity-Check (LDPC) Decoders basierend auf der IEEE 802.11n Matrix (Rate 1/2) für eine hocheffiziente Vorwärtsfehlerkorrektur.

Zukünftig soll das System durch ein $2\times2$ MIMO-Setup (Spatial Multiplexing) räumlich erweitert werden.

---

##  2. Regulatorien und exakte Sendeleistung (Link-Budget)

Das System wurde strikt nach dem Schweizerischen Frequenzweisungsplan des BAKOM (Richtlinie RIR1010-01 für Wideband Data Transmission im 2.4 GHz Band) entwickelt. Diese Richtlinie erlaubt eine maximale äquivalente isotrope Strahlungsleistung (EIRP) von 100 mW (entspricht 20 dBm).

Die Sendeleistung der in diesem Projekt verwendeten Hardware setzt sich wie folgt zusammen:
* **Transceiver-Ausgang (AD9361):** 7.9 dBm (bei 2.4 GHz)
* **Zusätzlicher Leistungsverstärker (PGA-102+):** +10.0 dB
* **Verluste durch Platinen-Leiterbahnen:** -0.3 dB
* **Verluste durch Balun (TCM1-63AX+):** -1.4 dB

Die effektive Ausgangsleistung an der SMA-Buchse des SDRs beträgt somit exakt **16.2 dBm**. 
Zusammen mit den verwendeten Rundstrahlantennen (3 dBi Gewinn) und einem minimalen Kabelverlust (0.2 dB) resultiert dies in einer abgestrahlten Gesamtsendeleistung (**EIRP**) von **~19.0 dBm**. Das System reizt das gesetzliche Limit von 20 dBm somit optimal aus, ohne es zu überschreiten.

---

##  3. Verwendete Hardware und Gehäuse

* **SDR-Plattform:** Modifizierte *Professional Edition* des ADALM-Pluto.
  * Transceiver: AD9363 (per Firmware-Mod auf AD9361 aufgewertet für 56 MHz Bandbreite).
  * System-on-Chip: Xilinx Zynq-7020 FPGA mit 1 GB RAM.
  * Oszillator: TCXO mit 0.5 ppm Genauigkeit zur Minimierung des Frequenzdrifts.
  * Datenanbindung: Gigabit-Ethernet für maximalen Durchsatz.
* **Antennen:** GEPRC 2.4 GHz Tri-Band Stick Antennen (Omnidirektional, 3 dBi, VSWR < 1.5).
* **Gehäuse & Abschirmung:** Eigens in Fusion 360 konstruiertes und 3D-gedrucktes Gehäuse. Um die empfindliche SDR-Elektronik vor lokalen 2.4 GHz Interferenzen abzuschirmen, ist der Innenraum vollständig mit Kupferfolie ausgekleidet (Faradayscher Käfig), welche durch Kapton-Tape elektrisch isoliert wurde. Die Kühlung erfolgt aktiv über einen 5V-Lüfter (4020).
* **Zeitsynchronisation (Work-in-Progress):** u-blox NEO-6M und ATGM336H GPS-Module, verbunden über einen ESP32-Mikrocontroller.

---

##  4. Funktionsweise der Skripte

Die Software wurde schrittweise entwickelt. Jedes Skript in diesem Repository erfüllt einen spezifischen Zweck bei der Evaluation des Funkprotokolls:

### `End_Simulation.py`
Die vollständige Basis-Simulation des Protokolls im idealen mathematischen Raum.
* **Ablauf:** Generiert zufällige Bits $\rightarrow$ LDPC-Codierung $\rightarrow$ Symbol-Mapping (Hybrid: LoRa-Wert, Up/Down-Richtung, QPSK-Phase) $\rightarrow$ I/Q-Signal-Synthese $\rightarrow$ AWGN-Kanal (künstliches Rauschen) $\rightarrow$ Empfänger-Dechirping $\rightarrow$ LDPC-Decodierung.
* **Prämisse:** Dieses Skript geht von einer **absolut perfekten Synchronisation** aus. Es existiert kein Carrier Frequency Offset (CFO) und kein Timing-Fehler. Es dient ausschließlich dazu, die theoretische Bitfehlerrate (BER) der Modulation unter starken Rauschbedingungen (SNR-Sweep) zu verifizieren.

### `simulation_simpel.py`
Simulation zur isolierten Entwicklung der Präambel-Erkennung und Paket-Synchronisation.
* **Besonderheit:** Die rechenintensive LDPC-Kanalcodierung wurde hier vollständig entfernt, um Fehlerquellen bei der Erprobung der reinen Frequenz-Synchronisation auszuschließen.
* **Synchronisations-Logik:** Die herkömmliche, extrem fehleranfällige Kreuzkorrelation im Zeitbereich wurde durch ein Schiebe-Fenster-Verfahren im Frequenzbereich ersetzt. Das Skript multipliziert das Empfangssignal mit einem lokalen Downchirp und sucht über eine Fast Fourier Transformation (FFT) nach dem charakteristischen Energie-Peak. Wiederholt sich dieser Peak bei den 16 Präambel-Symbolen im selben Frequenz-Bin, wird aus dem Bin-Index der exakte mathematische Startpunkt des Pakets (Timing-Offset) berechnet.

### `Pluto_test_simpel.py`
Die direkte Übertragung der simplen Simulation auf die reale SDR-Hardware (ADALM-Pluto).
* **Besonderheit:** Verzichtet auf LDPC und TDMA-Logiken. Nutzt die `adi`-Bibliothek für die Hardware-Ansteuerung über Ethernet.
* **Funktion:** Ermöglicht das manuelle Senden und kontinuierliche Empfangen von Basis-Paketen (324 Bits) über die Luft. Es dient dem direkten Test der FFT-Synchronisation unter realen Hochfrequenz-Bedingungen.

### `Pluto_test.py`
Das vollständige und komplexeste Hardware-Skript.
* **Inhalt:** Führt alle Komponenten zusammen: SDR-Steuerung, dynamische Rauschboden-Kalibrierung (das System misst beim Start 20-mal den Hintergrundpegel, um Schwellenwerte zu setzen), LDPC-Fehlerkorrektur und ein Time Division Multiple Access (TDMA) Verfahren.
* **Aktueller Status:** Das TDMA-Verfahren definiert feste Zeitschlitze zum Senden und Empfangen, um Puffer-Unterläufe zu verhindern. Aufgrund des extrem ungenauen Timings (Jitter) von Windows-Betriebssystemen verpasst der Code aktuell jedoch häufig die Slot-Grenzen. Daher wird in der Praxis vorerst das einmalige Senden und das kontinuierliche Empfangen genutzt.

---

##  5. Das Kernproblem: Die fehlgeschlagene Paket-Synchronisation

Trotz des erfolgreichen mathematischen Proof-of-Concepts in den Simulationen, schlägt der Empfang auf der echten SDR-Hardware momentan fehl. **Das einzige, eindeutig identifizierte Kernproblem ist die Paket-Synchronisation über die Luft.**

Das System ist im realen Funkkanal nicht in der Lage, den exakten Startpunkt eines Pakets zu finden und den Puffer korrekt zu zerschneiden. Schlägt der FFT-Trigger nicht exakt an der ersten Sample-Position der Präambel an, verschieben sich alle nachfolgenden Symbolgrenzen. Ein Dechirping der Payload ist unter diesen Umständen mathematisch unmöglich.

### Potenzielle Ursachen für das Versagen der Synchronisation
Die genauen Gründe für das Scheitern des Synchronisations-Algorithmus auf dem SDR sind noch nicht isoliert, jedoch kommen folgende Faktoren als Auslöser in Betracht:

1. **Rechenlatenz und Pufferverlust:** Die Implementierung in reinem Python (Ausführung der komplexen FFTs, Faltungen und Dechirp-Schleifen) beansprucht massiv CPU-Zeit. Es ist extrem wahrscheinlich, dass das Skript während der Berechnung eines Puffer-Blocks (Chunks) neue eintreffende Daten des SDRs verpasst und das gesendete Paket dadurch im Speicher zerschnitten oder überschrieben wird.
2. **Carrier Frequency Offset (CFO) und Phasenrotation:** Trotz des präzisen TCXO-Oszillators weisen die Plutos bei 2.45 GHz immer einen gewissen Frequenzversatz auf. Dies führt dazu, dass sich die Phase des Funksignals während der Übertragung kontinuierlich dreht. Da 2 Bits pro Symbol in der absoluten QPSK-Phasenlage codiert sind, führt ein unkorrigierter Drift unweigerlich zur Zerstörung dieser Datenpunkte.
3. **Massive Störsignale (Interferenzen):** Das 2.4 GHz ISM-Band ist stark ausgelastet. Hochenergetische WLAN-Bursts erzeugen Pegelspitzen im SDR, welche die Signalerkennung übersteuern oder die dynamischen Schwellenwerte (SNR-Ratio) fälschlicherweise triggern. Die Hardware-Verstärkungsregelung (AGC) kann diese Impulse nicht schnell genug glätten, ohne das eigene Signal zu verfälschen.

### Lösungsansatz für die Weiterentwicklung
Um die Software-Latenzen und das mangelhafte Betriebssystem-Timing als Fehlerquelle zu eliminieren, wird das TDMA-Verfahren in Zukunft auf eine reine Hardware-Uhr ausgelagert. Über die bestellten **u-blox NEO-6M GPS-Module** wird ein hochpräzises PPS-Signal (Pulse Per Second) an einen ESP32 geleitet, welcher die Sende- und Empfangsslots der SDRs auf die Mikrosekunde genau synchronisiert.
