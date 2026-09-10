"""
ugv/esp32_comm.py
=================
Handles HTTP communication with the UGV Beast's main Flask server (port 5000).
The Pi relays JSON commands to the ESP32 slave controller internally via serial.

JSON command format (from Waveshare docs):
  {"T": <type>, <key>: <value>, ...}

Motor commands  : T=1  (left, right speed -1.0..1.0)
Pan/Tilt        : T=100 (pan) / T=101 (tilt)  angle in degrees
Stop            : T=0
LED             : T=132 (io: 1/0)
Battery/IMU info: GET /get_info_all
"""

import requests
import time
import threading
from typing import Optional, Dict, Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_lock = threading.Lock()
_last_telemetry: Dict[str, Any] = {}
_telemetry_ts: float = 0.0
TELEMETRY_TTL = 2.0  # seconds


def _post_cmd(payload: dict, timeout: float = 1.5) -> bool:
    """POST a JSON command to the UGV Beast Flask server."""
    try:
        url = f"{config.UGV_BASE_URL}/js"
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


# ── Motor Control ─────────────────────────────────────────────────────────────

def move(left: float, right: float) -> bool:
    """
    Drive both tracks.  Values in range [-1.0, 1.0].
    Positive = forward, negative = backward.
    """
    left  = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    return _post_cmd({"T": 1, "L": round(left, 2), "R": round(right, 2)})


def stop() -> bool:
    return _post_cmd({"T": 0})


def forward(speed: float = 0.5) -> bool:
    return move(speed, speed)


def backward(speed: float = 0.5) -> bool:
    return move(-speed, -speed)


def turn_left(speed: float = 0.5) -> bool:
    return move(-speed, speed)


def turn_right(speed: float = 0.5) -> bool:
    return move(speed, -speed)


# ── Pan/Tilt ──────────────────────────────────────────────────────────────────

def set_pan(angle: int) -> bool:
    """Set pan servo angle (0–180°)."""
    angle = max(config.PAN_MIN, min(config.PAN_MAX, int(angle)))
    return _post_cmd({"T": 100, "X": angle, "SPD": 0, "ACC": 0})


def set_tilt(angle: int) -> bool:
    """Set tilt servo angle (30–150°)."""
    angle = max(config.TILT_MIN, min(config.TILT_MAX, int(angle)))
    return _post_cmd({"T": 101, "Y": angle, "SPD": 0, "ACC": 0})


def pan_tilt_center() -> bool:
    ok1 = set_pan(config.PAN_CENTER)
    ok2 = set_tilt(config.TILT_CENTER)
    return ok1 and ok2


# ── LED ───────────────────────────────────────────────────────────────────────

def set_led(on: bool) -> bool:
    return _post_cmd({"T": 132, "IO": 1 if on else 0})


# ── Telemetry ─────────────────────────────────────────────────────────────────

def get_telemetry(force: bool = False) -> Dict[str, Any]:
    """
    Fetch robot telemetry (battery voltage, current, IMU, CPU, etc.)
    Results are cached for TELEMETRY_TTL seconds.
    Returns empty dict on failure (demo / offline).
    """
    global _last_telemetry, _telemetry_ts
    now = time.time()
    if not force and (now - _telemetry_ts) < TELEMETRY_TTL:
        return _last_telemetry

    try:
        r = requests.get(f"{config.UGV_BASE_URL}/get_info_all", timeout=1.5)
        if r.status_code == 200:
            data = r.json()
            _last_telemetry = data
            _telemetry_ts = now
            return data
    except Exception:
        pass
    return _last_telemetry  # stale or empty


def is_robot_online() -> bool:
    """Ping the robot and return True if reachable."""
    try:
        r = requests.get(f"{config.UGV_BASE_URL}/", timeout=1.0)
        return r.status_code < 500
    except Exception:
        return False
