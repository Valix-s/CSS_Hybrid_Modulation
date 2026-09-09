# Custom Hybrid LoRa/CSS + SSK + DQPSK SDR Modem

> **A Swiss High School Graduation Project (Maturaarbeit)** by Valentin Seeholzer.

---

## About the Project

This repository contains the Python-based implementation of a custom physical layer (PHY) for Software Defined Radios (SDR).

The project investigates whether a hybrid modulation scheme can combine the robustness of **Chirp Spread Spectrum (CSS)** with a higher information density per symbol by additionally encoding information in **chirp direction (Slope-Shift Keying, SSK)** and **differential phase (DQPSK)**.

The system was developed as part of a Swiss high school graduation project and tested experimentally on two AD9363/Zynq-7020-based SDR platforms under real-world radio conditions.

The final experimental configuration uses:

* **2.45 GHz** carrier frequency
* **1 MHz** channel bandwidth
* **Spreading Factor SF5**
* **CSS + SSK + DQPSK** hybrid modulation
* **LDPC coding, rate 1/2, n = 648**
* GPS-synchronised **TDMA** burst transmission

The theoretical net data rate of the final modulation and packet structure is approximately **104.4 kbit/s**. The measured end-to-end throughput of the current Python-based prototype is considerably lower due to software processing and TDMA timing overhead.

---

## Key Features

* **Hybrid Modulation:** Combines CSS symbol shifts with chirp direction (SSK) and DQPSK phase information, transmitting **8 bits per symbol at SF5**.
* **LDPC Error Correction:** Implements a rate-1/2 LDPC encoder and Belief Propagation decoder based on the **IEEE 802.11n** LDPC structure.
* **Custom SDR PHY:** End-to-end transmission and reception pipeline including modulation, demodulation, packet detection, CFO estimation/correction and LDPC decoding.
* **CFO Compensation:** Estimates carrier frequency offset from the packet preamble and compensates it digitally.
* **TDMA Burst Mode:** Uses GPS/PPS-based time synchronisation to separate transmission and processing phases and reduce buffer overruns.
* **MIMO Ready:** The hardware enclosure and antenna layout are prepared for future 2x2 MIMO investigations; the final measurements primarily use an SISO link.
* **Hardware-in-the-Loop:** Generates and processes complex I/Q baseband signals for AD9363/Zynq-based SDR hardware.

---

## Experimental Results

The prototype was evaluated on several outdoor test paths between **20 m and 200 m**.

Under the project's defined criterion of **≤5% BER and ≤5% packet loss**, the best measured outdoor link remained practically usable up to **100 m** on one of the test paths.

The project did **not** demonstrate the originally targeted combination of several-hundred-metre range and more than 10 kbit/s end-to-end throughput. The experiments instead helped identify practical limitations such as:

* Python processing latency
* packet detection errors
* CFO estimation errors at low SNR
* channel asymmetry
* real-world interference and multipath effects

---

## Hardware Requirements

For a physical reproduction of the test setup, the following hardware was used:

* **2x AD9363/Zynq-7020-based SDRs** (e.g. OpenSourceSDRLab 7020-SDR or similar)
* **2x 4-antenna configurations** for the SDRs
* **2.4 GHz antennas**, approximately **3 dBi**
* SMA/coaxial connections suitable for the chosen SDR hardware
* Optional **GPS/PPS synchronisation hardware** for the TDMA test mode

---

## Software Dependencies

The modem and DSP pipeline are implemented in Python.

Install the main dependencies with:

```bash
pip install numpy scipy matplotlib ldpc
```

The project makes use of libraries such as:

* **NumPy** for numerical and vectorised processing
* **SciPy** for signal-processing functionality
* **Matplotlib** for analysis and visualisation
* **LDPC-related Python tooling** for channel coding and decoding

---

## Project Structure

The repository contains the main SDR/modem implementation, signal-processing functions, test scripts and supporting analysis material.

The most important processing stages are:

```text
Payload
   ↓
LDPC Encoding
   ↓
Hybrid Symbol Mapping
(CSS + SSK + DQPSK)
   ↓
I/Q Signal Generation
   ↓
SDR Transmission
   ↓
SDR Reception
   ↓
Packet Detection + CFO Correction
   ↓
De-Chirping / FFT Demodulation
   ↓
LDPC Decoding
   ↓
Recovered Payload
```

---

## Status

**Research prototype / experimental project**

The current implementation is intended for experimentation, education and further development rather than commercial deployment.

Future work includes FPGA-based acceleration, improved synchronisation and packet detection, adaptive modulation and a controlled comparison with established systems such as LoRa and WLAN.

---

## License

This project is open-source. Feel free to use, study and modify the code.
