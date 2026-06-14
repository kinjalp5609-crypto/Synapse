#  SYNAPSE (EEG Module)
#  Reads brain signals using BioAmp EXG Pill by Upside Down Labs
#  Connection: BioAmp OUT → Arduino Uno (A0) → USB → Laptop

import serial
import serial.tools.list_ports
import time
import threading
import numpy as np
from collections import deque

class SynapseEEG:

    def __init__(self):
        self.connection  = None
        self.connected   = False

        # Brain state outputs
        self.attention   = 50     # 0-100
        self.brain_state = "NEUTRAL"
        self.speed_level = 2      # 1=slow 2=medium 3=fast

        # Signal processing buffers
        # BioAmp samples at 256 Hz
        # We keep 2 seconds of data = 512 samples
        self.SAMPLE_RATE = 256
        self.BUFFER_SIZE = 512
        self.signal_buffer = deque(maxlen=self.BUFFER_SIZE)

        # Threading
        self._running = False
        self._thread  = None
        self._lock    = threading.Lock()

    # Connection
    def find_port(self):
        """Find Arduino port automatically"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"Found: {port.device} — {port.description}")
            if any(name in port.description.upper() for name in
                   ['ARDUINO', 'CH340', 'CP210', 'USB SERIAL', 'ACM']):
                return port.device
        return None

    def connect(self, port=None):
        """Connect to Arduino running BioAmp sketch"""
        try:
            if port is None:
                port = self.find_port()

            if port is None:
                ports = list(serial.tools.list_ports.comports())
                if not ports:
                    print("No ports found. Using simulated EEG.")
                    return False
                print("Available ports:")
                for i, p in enumerate(ports):
                    print(f"  {i}: {p.device} — {p.description}")
                choice = input("Select port number: ")
                port = ports[int(choice)].device

            # BioAmp Arduino sketch uses 115200 baud
            self.connection = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
            self.connected  = True
            print(f"EEG connected on {port}")

            self._running = True
            self._thread  = threading.Thread(
                target=self._read_loop, daemon=True
            )
            self._thread.start()
            return True

        except Exception as e:
            print(f"EEG connection failed: {e}")
            return False

    #Background reading thread
    def _read_loop(self):
        """
        BioAmp EXG Pill Arduino sketch sends one integer
        per line at 256 Hz. Values are raw ADC (0-1023).
        Example output:
            512
            518
            504
            ...
        """
        while self._running and self.connected:
            try:
                if self.connection.in_waiting > 0:
                    line = self.connection.readline()
                    line = line.decode('utf-8', errors='ignore').strip()
                    if line.lstrip('-').isdigit():
                        value = int(line)
                        with self._lock:
                            self.signal_buffer.append(value)
                        self._process_signal()
            except Exception as e:
                print(f"EEG read error: {e}")
                self.connected = False
                break
            time.sleep(0.001)

    # Signal processing
    def _process_signal(self):
        """
        Compute attention from Beta band power (13-30 Hz).
        Beta waves = active thinking / focus.
        High beta power = high attention = robot goes fast.
        Steps:
        1. Take last 256 samples (1 second of data)
        2. Apply FFT to get frequency components
        3. Compute power in Beta band (13-30 Hz)
        4. Compute power in Alpha band (8-13 Hz)
        5. Attention = Beta / (Alpha + Beta)
        6. Smooth with running average
        """
        with self._lock:
            if len(self.signal_buffer) < self.SAMPLE_RATE:
                return  # Not enough data yet
            samples = list(self.signal_buffer)[-self.SAMPLE_RATE:]

        # Convert to numpy array and remove DC offset
        sig = np.array(samples, dtype=float)
        sig = sig - np.mean(sig)

        # Apply FFT
        fft_vals  = np.abs(np.fft.rfft(sig))
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0/self.SAMPLE_RATE)

        # Extract band powers
        alpha_mask = (fft_freqs >= 8)  & (fft_freqs <= 13)
        beta_mask  = (fft_freqs >= 13) & (fft_freqs <= 30)

        alpha_power = np.sum(fft_vals[alpha_mask]  ** 2)
        beta_power  = np.sum(fft_vals[beta_mask]   ** 2)
        total_power = alpha_power + beta_power + 1e-6

        # Attention ratio (0.0 to 1.0)
        raw_attention = beta_power / total_power

        # Scale to 0-100 and smooth
        new_attention = int(raw_attention * 100)
        self.attention = int(self.attention * 0.85 + new_attention * 0.15)
        self.attention = max(0, min(100, self.attention))

        # Update brain state
        self._update_state()

    def _update_state(self):
        if self.attention >= 60:
            self.brain_state = "FOCUSED"
            self.speed_level = 3
        elif self.attention >= 35:
            self.brain_state = "NEUTRAL"
            self.speed_level = 2
        else:
            self.brain_state = "RELAXED"
            self.speed_level = 1

    # Public API
    def get_state(self):
        return {
            'attention':   self.attention,
            'brain_state': self.brain_state,
            'speed_level': self.speed_level,
            'connected':   self.connected
        }

    def get_speed(self):
        return self.speed_level

    def disconnect(self):
        self._running = False
        if self.connection:
            self.connection.close()
            self.connected = False
            print("EEG disconnected.")


# Arduino sketch to upload for BioAmp EXG Pill
ARDUINO_SKETCH = """
// Upload this to Arduino Uno before connecting BioAmp EXG Pill
// BioAmp OUT pin → Arduino A0
// BioAmp VCC     → Arduino 5V
// BioAmp GND     → Arduino GND

#define SAMPLE_RATE 256
#define BAUD_RATE   115200
#define INPUT_PIN   A0

void setup() {
    Serial.begin(BAUD_RATE);
}

void loop() {
    static unsigned long past = 0;
    unsigned long present = micros();
    unsigned long interval = present - past;
    past = present;

    static long timer = 0;
    timer -= interval;

    if (timer < 0) {
        timer += (1000000 / SAMPLE_RATE);
        int sensor_value = analogRead(INPUT_PIN);
        Serial.println(sensor_value);
    }
}
"""


# Simulated EEG (hardware is not available)
class SimulatedEEG:
    """
    Simulates brain states for testing without hardware.
    Cycles: FOCUSED (30s) → NEUTRAL (30s) → RELAXED (30s)
    """
    def __init__(self):
        self.attention   = 50
        self.brain_state = "NEUTRAL"
        self.speed_level = 2
        self._start_time = time.time()

    def connect(self, port=None):
        print("Simulated EEG active. No hardware needed.")
        self._start_time = time.time()
        return True

    def get_state(self):
        elapsed = (time.time() - self._start_time) % 90

        if elapsed < 30:
            self.attention   = 70 + int(np.sin(elapsed) * 10)
            self.brain_state = "FOCUSED"
            self.speed_level = 3
        elif elapsed < 60:
            self.attention   = 50 + int(np.sin(elapsed) * 8)
            self.brain_state = "NEUTRAL"
            self.speed_level = 2
        else:
            self.attention   = 25 + int(np.sin(elapsed) * 8)
            self.brain_state = "RELAXED"
            self.speed_level = 1

        return {
            'attention':   self.attention,
            'brain_state': self.brain_state,
            'speed_level': self.speed_level,
            'connected':   True
        }

    def get_speed(self):
        return self.speed_level

    def disconnect(self):
        print("Simulated EEG disconnected.")


# Test
if __name__ == "__main__":
    print("No hardware yet — running simulation.")
    print()
    print("=" * 50)
    print("ARDUINO SKETCH TO UPLOAD IN JULY:")
    print("=" * 50)
    print(ARDUINO_SKETCH)
    print("=" * 50)
    print()

    eeg = SimulatedEEG()
    eeg.connect()

    print("Simulating 15 seconds of brain activity...")
    print()
    for i in range(30):
        state = eeg.get_state()
        bar   = "█" * (state['attention'] // 5)
        print(f"Attention: {state['attention']:3d} {bar:20s} "
              f"| {state['brain_state']:8s} | Speed: {state['speed_level']}")
        time.sleep(0.5)

    eeg.disconnect()
    print()
    print("eeg.py is ready!")