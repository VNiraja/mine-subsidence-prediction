import network
import urequests
import time
import json
from machine import Pin, ADC

# ============================================================
# SIH26025 — ESP32 Wokwi MicroPython Client
# AI-Enabled Mine Subsidence Real-Time Sensor Node
# ============================================================

WIFI_SSID = "Wokwi-GUEST"
WIFI_PASSWORD = ""

# Replace host with active LocalTunnel HTTPS URL or local IP
# Target endpoint: POST /predict (sensor telemetry) or GET /predict/latest
SERVER_URL = "https://violet-groups-lie.loca.lt/predict"
POLL_INTERVAL_SECONDS = 5

# ============================================================
# GPIO PIN CONFIGURATION
# ============================================================
# Output Actuators (LEDs + Buzzer)
green_led = Pin(12, Pin.OUT)
yellow_led = Pin(14, Pin.OUT)
red_led = Pin(27, Pin.OUT)
buzzer = Pin(26, Pin.OUT)

# Sensor Inputs (Analog Potentiometers & Ultrasonic HC-SR04)
try:
    pot_crack = ADC(Pin(34))
    pot_crack.atten(ADC.ATTN_11DB)
except Exception:
    pot_crack = None

try:
    pot_strain = ADC(Pin(35))
    pot_strain.atten(ADC.ATTN_11DB)
except Exception:
    pot_strain = None

try:
    trig_pin = Pin(5, Pin.OUT)
    echo_pin = Pin(18, Pin.IN)
except Exception:
    trig_pin = None
    echo_pin = None

wifi = network.WLAN(network.STA_IF)


def set_signals(risk):
    """Set LED and Buzzer outputs based on AI risk prediction."""
    green_led.value(0)
    yellow_led.value(0)
    red_led.value(0)
    buzzer.value(0)

    risk_upper = str(risk).upper()
    if risk_upper == "LOW":
        green_led.value(1)
    elif risk_upper in ("MEDIUM", "MODERATE", "WARNING"):
        yellow_led.value(1)
    elif risk_upper in ("HIGH", "CRITICAL"):
        red_led.value(1)
        buzzer.value(1)
    elif risk_upper != "OFF":
        print("Unknown risk level:", risk)


def read_sensor_payload():
    """Reads sensor values from physical/simulated inputs."""
    # Crack Width: 0-4095 mapped to 0.0 - 25.0 mm
    if pot_crack is not None:
        try:
            crack_raw = pot_crack.read()
            crack_width = round((crack_raw / 4095.0) * 25.0, 2)
        except Exception:
            crack_width = 1.2
    else:
        crack_width = 1.2

    # Strain: 0-4095 mapped to 0.0000 - 0.0100 microstrain
    if pot_strain is not None:
        try:
            strain_raw = pot_strain.read()
            strain = round((strain_raw / 4095.0) * 0.0100, 4)
        except Exception:
            strain = 0.0035
    else:
        strain = 0.0035

    # HC-SR04 Ultrasonic Distance (Displacement mm)
    displacement = 12.5
    if trig_pin is not None and echo_pin is not None:
        try:
            trig_pin.value(0)
            time.sleep_us(2)
            trig_pin.value(1)
            time.sleep_us(10)
            trig_pin.value(0)
            
            timeout = 30000
            t_start = time.ticks_us()
            while echo_pin.value() == 0:
                if time.ticks_diff(time.ticks_us(), t_start) > timeout:
                    break
            t1 = time.ticks_us()
            while echo_pin.value() == 1:
                if time.ticks_diff(time.ticks_us(), t1) > timeout:
                    break
            t2 = time.ticks_us()
            pulse_duration = time.ticks_diff(t2, t1)
            dist_mm = (pulse_duration * 0.343) / 2.0
            if 0 < dist_mm < 4000:
                displacement = round(dist_mm, 1)
        except Exception:
            displacement = 12.5

    # MPU6050 Tilt angle (degrees)
    tilt = 2.5

    return {
        "tilt": float(tilt),
        "displacement": float(displacement),
        "crack_width": float(crack_width),
        "strain": float(strain)
    }


def connect_wifi():
    if wifi.isconnected():
        return

    wifi.active(True)
    wifi.connect(WIFI_SSID, WIFI_PASSWORD)
    for _ in range(30):
        if wifi.isconnected():
            print("WiFi connected; IP:", wifi.ifconfig()[0])
            return
        time.sleep(1)
    raise OSError("WiFi connection timed out")


print("SIH26025 Mine Subsidence Sensor Node Initialized.")

while True:
    response = None
    try:
        connect_wifi()
        payload = read_sensor_payload()
        print("Transmitting Sensor Payload:", payload)

        headers = {
            "Content-Type": "application/json",
            "bypass-tunnel-reminder": "true"
        }

        if "/predict/latest" in SERVER_URL:
            response = urequests.get(
                SERVER_URL,
                headers=headers,
                timeout=10
            )
        else:
            response = urequests.post(
                SERVER_URL,
                json=payload,
                headers=headers,
                timeout=10
            )

        if response.status_code != 200:
            raise OSError("API returned HTTP " + str(response.status_code))

        result = response.json()
        risk = str(result.get("risk", result.get("risk_level", "LOW"))).upper()
        confidence = result.get("confidence", 0.0)
        action = result.get("action_signal", result.get("led_signal", "GREEN"))

        print("AI Risk Prediction:", risk, "| Conf:", confidence, "| Action:", action)
        set_signals(risk)

    except Exception as error:
        set_signals("OFF")
        print("Prediction request failed; LEDs off:", error)
    finally:
        if response is not None:
            response.close()

    time.sleep(POLL_INTERVAL_SECONDS)