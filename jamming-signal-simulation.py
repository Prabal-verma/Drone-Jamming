import numpy as np
import matplotlib.pyplot as plt

# 1. Define Signal and Jamming Parameters
np.random.seed(42)  # For reproducibility

# Simulation parameters
signal_length = 1000  # Number of samples
signal_power = 1.0    # Power of the drone signal
noise_power = 0.1     # Power of background noise
jamming_power = 5.0   # Power of the jamming signal
jam_start, jam_end = 400, 600  # Jamming duration (indices)

# 2. Simulate Drone Communication Signal
time = np.linspace(0, 1, signal_length)
drone_signal = np.sqrt(signal_power) * np.sin(2 * np.pi * 10 * time)  # Sine wave signal
noise = np.sqrt(noise_power) * np.random.randn(signal_length)  # Gaussian noise

# Add noise to the clean drone signal
received_signal = drone_signal + noise

# 3. Introduce Jamming Signal
jamming_signal = np.zeros(signal_length)
jamming_signal[jam_start:jam_end] = np.sqrt(jamming_power) * np.random.randn(jam_end - jam_start)

# Received signal with jamming
jammed_signal = received_signal + jamming_signal

# 4. Jamming Detection Based on SNR
def calculate_snr(signal, noise_power):
    signal_power = np.mean(signal**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Calculate SNR before, during, and after jamming
snr_before_jamming = calculate_snr(received_signal[:jam_start], noise_power)
snr_during_jamming = calculate_snr(jammed_signal[jam_start:jam_end], noise_power)
snr_after_jamming = calculate_snr(jammed_signal[jam_end:], noise_power)

print(f"SNR Before Jamming: {snr_before_jamming:.2f} dB")
print(f"SNR During Jamming: {snr_during_jamming:.2f} dB (Drop Detected)")
print(f"SNR After Jamming: {snr_after_jamming:.2f} dB")

# 5. Countermeasure: Adaptive Frequency Hopping
def adaptive_frequency_hopping(signal, jammed_indices):
    """
    Simulate frequency hopping by replacing jammed signal with clean signal at new frequency.
    """
    clean_signal = np.copy(signal)
    for idx in jammed_indices:
        clean_signal[idx] = np.sin(2 * np.pi * 15 * time[idx])  # Hop to 15 Hz frequency
    return clean_signal

# Detect indices where SNR drops significantly (jamming detected)
jammed_indices = np.arange(jam_start, jam_end)
mitigated_signal = adaptive_frequency_hopping(jammed_signal, jammed_indices)

# 6. Plot Results
plt.figure(figsize=(12, 8))

# Original signal
plt.subplot(4, 1, 1)
plt.plot(time, drone_signal, label="Original Drone Signal")
plt.title("Original Drone Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Jammed signal
plt.subplot(4, 1, 2)
plt.plot(time, jammed_signal, label="Jammed Signal", color="r")
plt.axvspan(time[jam_start], time[jam_end], color='red', alpha=0.3, label="Jamming")
plt.title("Jammed Signal (Jamming Introduced)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Mitigated signal (Frequency Hopping Applied)
plt.subplot(4, 1, 3)
plt.plot(time, mitigated_signal, label="Mitigated Signal", color="g")
plt.title("Mitigated Signal with Frequency Hopping")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# SNR Drop Visualization
plt.subplot(4, 1, 4)
snr_values = [snr_before_jamming, snr_during_jamming, snr_after_jamming]
plt.bar(["Before Jamming", "During Jamming", "After Jamming"], snr_values, color=["blue", "red", "green"])
plt.title("SNR Before, During, and After Jamming")
plt.ylabel("SNR (dB)")

plt.tight_layout()
plt.legend()
plt.show()
