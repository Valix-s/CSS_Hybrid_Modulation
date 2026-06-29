# High-Speed SDR Modem: Bridging the Gap Between Wi-Fi and LoRa

This repository contains the Python source code, accompanying simulations, and development documentation for my Matura project (high school graduation thesis) at Kantonsschule Reussbühl (Lucerne, Switzerland).

**Thesis Topic:** Development of a license-free wireless protocol to close the bandwidth-range gap between Wi-Fi and LoRa: Implementation of a high-speed SDR modem in compliance with the BAKOM RIR1010-01 regulation.

---

##  1. Project Goal and Modulation Concept

The primary goal of this project is to develop a hybrid communication system that combines the physical advantages of the robust LoRa modulation (Chirp Spread Spectrum, CSS) with significantly higher data transfer rates. Current wireless technologies usually force a compromise between high data rates at a short range (Wi-Fi) or long range at minimal data rates (LoRa). 

To close this gap, the developed protocol operates in the 2.4 GHz ISM band with a very high **bandwidth of 10 MHz**. The system is designed for a low **Spreading Factor of 5 (SF5)** and breaks the classic data rate limits of LoRa by expanding the modulation layer:

1. **Slope-Shift Keying (SSK):** Parallel use of up- and down-chirps. Detecting the chirp direction provides an exact data gain of +1 bit per symbol.
2. **Quadrature Phase-Shift Keying (QPSK):** Embedding an additional 2 bits into the phase angle (45° offsets: $\pi/4, 3\pi/4, 5\pi/4, 7\pi/4$) of each individual chirp.
3. **Channel Coding:** Implementation of a Low-Density Parity-Check (LDPC) decoder based on the IEEE 802.11n matrix (rate 1/2) for highly efficient forward error correction.

In the future, the system will be spatially expanded via a $2\times2$ MIMO (Spatial Multiplexing) setup.

---

##  2. Regulations and Exact Transmission Power (Link Budget)

The system was developed strictly according to the Swiss frequency allocation plan of BAKOM (Federal Office of Communications, guideline RIR1010-01 for Wideband Data Transmission in the 2.4 GHz band). This guideline permits a maximum Equivalent Isotropically Radiated Power (EIRP) of 100 mW (equivalent to 20 dBm).

The transmission power of the hardware used in this project is calculated as follows:
* **Transceiver Output (AD9361):** 7.9 dBm (at 2.4 GHz)
* **Additional Power Amplifier (PGA-102+):** +10.0 dB
* **PCB Trace Losses:** -0.3 dB
* **Balun Losses (TCM1-63AX+):** -1.4 dB

The effective output power at the SMA connector of the SDR is therefore exactly **16.2 dBm**. 
Combined with the omnidirectional antennas used (3 dBi gain) and minimal cable loss (0.2 dB), this results in a total radiated power (**EIRP**) of **~19.0 dBm**. Thus, the system optimally utilizes the legal limit of 20 dBm without exceeding it.

---

##  3. Hardware and Enclosure Used

* **SDR Platform:** Modified *Professional Edition* of the ADALM-Pluto.
  * Transceiver: AD9363 (upgraded via firmware mod to AD9361 for 56 MHz bandwidth).
  * System-on-Chip: Xilinx Zynq-7020 FPGA with 1 GB RAM.
  * Oscillator: TCXO with 0.5 ppm accuracy to minimize frequency drift.
  * Data Connection: Gigabit Ethernet for maximum throughput.
* **Antennas:** GEPRC 2.4 GHz Tri-Band Stick Antennas (Omnidirectional, 3 dBi, VSWR < 1.5).
* **Enclosure & Shielding:** Custom 3D-printed enclosure designed in Fusion 360. To shield the sensitive SDR electronics from local 2.4 GHz interference, the interior is completely lined with copper foil (Faraday cage), which is electrically isolated using Kapton tape. Active cooling is provided by a 5V fan (4020).
* **Time Synchronization (Work-in-Progress):** u-blox NEO-6M and ATGM336H GPS modules, connected via an ESP32 microcontroller.

---

##  4. Script Functionality

The software was developed incrementally. Each script in this repository fulfills a specific purpose in evaluating the wireless protocol:

### `01_simulation_ideal_channel.py`
The complete basic simulation of the protocol in an ideal mathematical space.
* **Process:** Generates random bits $\rightarrow$ LDPC encoding $\rightarrow$ Symbol mapping (Hybrid: LoRa value, up/down direction, QPSK phase) $\rightarrow$ I/Q signal synthesis $\rightarrow$ AWGN channel (artificial noise) $\rightarrow$ Receiver dechirping $\rightarrow$ LDPC decoding.
* **Premise:** This script assumes **absolutely perfect synchronization**. There is no Carrier Frequency Offset (CFO) and no timing error. It serves exclusively to verify the theoretical Bit Error Rate (BER) of the modulation under heavy noise conditions (SNR sweep).

### `02_simulation_fft_sync.py`
Simulation for the isolated development of preamble detection and packet synchronization.
* **Special Feature:** The computationally intensive LDPC channel coding was completely removed here to rule out sources of error when testing pure frequency synchronization.
* **Synchronization Logic:** The conventional, highly error-prone cross-correlation in the time domain was replaced by a sliding-window method in the frequency domain. The script multiplies the received signal with a local down-chirp and searches for the characteristic energy peak using a Fast Fourier Transform (FFT). If this peak repeats in the same frequency bin for the 16 preamble symbols, the exact mathematical starting point of the packet (timing offset) is calculated from the bin index.

### `03_sdr_rx_tx_basic.py`
The direct porting of the simple simulation to the real SDR hardware (ADALM-Pluto).
* **Special Feature:** Omits LDPC and TDMA logic. Uses the `adi` library for hardware control via Ethernet.
* **Function:** Enables the manual transmission and continuous reception of basic packets (324 bits) over the air. It serves to directly test FFT synchronization under real RF conditions.

### `04_sdr_transceiver_full.py`
The complete and most complex hardware script.
* **Contents:** Brings all components together: SDR control, dynamic noise floor calibration (the system measures the background level 20 times at startup to set thresholds), LDPC error correction, and a Time Division Multiple Access (TDMA) method.
* **Current Status:** The TDMA method defines fixed time slots for transmitting and receiving to prevent buffer underruns. However, due to the extremely imprecise timing (jitter) of Windows operating systems, the code currently frequently misses the slot boundaries. Therefore, for practical testing, one-time transmission and continuous reception are primarily used for now.

---

##  5. The Core Problem: Failed Packet Synchronization

Despite the successful mathematical proof-of-concept in the simulations, reception on the real SDR hardware is currently failing. **The only clearly identified core problem is packet synchronization over the air.**

In the real radio channel, the system is unable to find the exact starting point of a packet and slice the buffer correctly. If the FFT trigger does not hit the exact first sample position of the preamble, all subsequent symbol boundaries shift. Dechirping the payload is mathematically impossible under these conditions.

### Potential Causes for Synchronization Failure
The exact reasons for the failure of the synchronization algorithm on the SDR have not yet been isolated; however, the following factors are considered as potential triggers:

1. **Computing Latency and Buffer Loss:** The implementation in pure Python (execution of complex FFTs, convolutions, and dechirp loops) demands massive CPU time. It is highly likely that while computing a buffer block (chunk), the script misses newly arriving data from the SDR, causing the transmitted packet to be sliced or overwritten in memory.
2. **Carrier Frequency Offset (CFO) and Phase Rotation:** Despite the precise TCXO oscillator, the Plutos always exhibit a certain frequency offset at 2.45 GHz. This causes the phase of the radio signal to rotate continuously during transmission. Since 2 bits per symbol are coded in the absolute QPSK phase, an uncorrected drift inevitably leads to the destruction of these data points.
3. **Massive Interference Signals:** The 2.4 GHz ISM band is heavily utilized. High-energy Wi-Fi bursts generate peak levels in the SDR, overloading signal detection or falsely triggering the dynamic thresholds (SNR ratio). The hardware Automatic Gain Control (AGC) cannot smooth out these impulses quickly enough without distorting the actual signal.

### Proposed Solution for Further Development
To eliminate software latencies and poor OS timing as error sources, the TDMA method will be offloaded to a pure hardware clock in the future. Via the ordered **u-blox NEO-6M GPS modules**, a high-precision PPS (Pulse Per Second) signal will be routed to an ESP32, which will synchronize the transmission and reception slots of the SDRs down to the microsecond.
