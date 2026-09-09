import adi
import time
import numpy as np

# -------------------------------------------------------
# Pluto Einstellungen
# -------------------------------------------------------

SDR_IP = "ip:192.168.1.10"

FS = 2_000_000
BW = 1_000_000

BUFFER_SIZE = 20_000      # 10 ms

CENTER_FREQ = 2_450_000_000

# -------------------------------------------------------

sdr = adi.Pluto(SDR_IP)

sdr.sample_rate = FS
sdr.rx_rf_bandwidth = BW
sdr.rx_lo = CENTER_FREQ
sdr.rx_buffer_size = BUFFER_SIZE
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 70

# Buffer leeren
for _ in range(10):
    sdr.rx()

print("Starte Test...\n")

expected_time = BUFFER_SIZE / FS

print(f"Erwartete Bufferdauer : {expected_time*1000:.3f} ms\n")

counter = 0

worst = 0
avg = []

start = time.time()

try:

    while True:

        t0 = time.perf_counter()

        rx = sdr.rx()

        dt = time.perf_counter() - t0

        counter += len(rx)

        avg.append(dt)

        if dt > worst:
            worst = dt

        expected_samples = (time.time()-start)*FS

        lost = expected_samples-counter

        print(
            f"RX {len(rx):7d} Samples | "
            f"{dt*1000:7.3f} ms | "
            f"geschätzt verloren: {lost:10.0f} Samples",
            end="\r"
        )

except KeyboardInterrupt:

    print()

    print("-------------------------------------")

    print(f"Empfangene Samples : {counter:,}")

    print(f"Mittlere RX Zeit   : {np.mean(avg)*1000:.3f} ms")

    print(f"Max RX Zeit        : {worst*1000:.3f} ms")

    print(f"Erwartet           : {expected_time*1000:.3f} ms")

    print("-------------------------------------")

    if worst > expected_time*1.5:

        print("\nWARNUNG:")
        print("Python liest den SDR nicht schnell genug.")
        print("Sehr wahrscheinlich wurden Samples verworfen.")

    else:

        print("\nKeine auffälligen Verzögerungen erkannt.")