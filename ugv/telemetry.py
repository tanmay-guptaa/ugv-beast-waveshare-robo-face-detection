"""
ugv/telemetry.py
================
Collects and caches system telemetry for the dashboard.

On the Pi: reads CPU/RAM via psutil + calls the UGV Flask API for
           battery voltage/current (INA219) and IMU data (ICM20948).

On Windows dev machine: returns simulated values so the UI looks live.
"""

import time
import random
import psutil
from typing import Dict, Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ugv import esp32_comm

_cache: Dict[str, Any] = {}
_cache_ts: float = 0.0
TTL = 2.0   # seconds


def _simulated() -> Dict[str, Any]:
    """Return realistic-looking simulated telemetry for demo/dev mode."""
    t = time.time()
    return {
        "cpu_percent":    round(random.uniform(18, 55), 1),
        "ram_percent":    round(random.uniform(35, 65), 1),
        "cpu_temp":       round(random.uniform(42.0, 62.0), 1),
        "disk_percent":   round(random.uniform(20, 45), 1),
        "battery_voltage":round(11.4 + random.uniform(-0.3, 0.3), 2),
        "battery_current":round(random.uniform(0.4, 1.8), 2),
        "battery_percent":round(random.uniform(60, 95), 1),
        "imu_roll":       round(random.uniform(-5, 5), 1),
        "imu_pitch":      round(random.uniform(-8, 8), 1),
        "imu_yaw":        round(random.uniform(-180, 180), 1),
        "robot_online":   False,  # in demo mode robot isn't connected
        "esp32_firmware": "demo",
        "camera_fps":     30.0,
        "camera_res":     "640×480",
    }


def get() -> Dict[str, Any]:
    """Return (possibly cached) telemetry dict."""
    global _cache, _cache_ts
    now = time.time()
    if (now - _cache_ts) < TTL:
        return _cache

    if config.DEMO_MODE:
        _cache = _simulated()
        _cache_ts = now
        return _cache

    # ── Live mode ──
    data: Dict[str, Any] = {}

    # Pi system stats
    data["cpu_percent"]  = psutil.cpu_percent(interval=0.2)
    data["ram_percent"]  = psutil.virtual_memory().percent
    data["disk_percent"] = psutil.disk_usage("/").percent

    # CPU temperature (Pi)
    try:
        temps = psutil.sensors_temperatures()
        cpu_t = temps.get("cpu_thermal") or temps.get("coretemp") or []
        data["cpu_temp"] = cpu_t[0].current if cpu_t else 0.0
    except Exception:
        data["cpu_temp"] = 0.0

    # UGV telemetry from Flask API
    tele = esp32_comm.get_telemetry()
    data["battery_voltage"] = tele.get("battery_voltage", 0.0)
    data["battery_current"] = tele.get("battery_current", 0.0)
    data["battery_percent"] = tele.get("battery_percent", 0.0)
    data["imu_roll"]        = tele.get("roll", 0.0)
    data["imu_pitch"]       = tele.get("pitch", 0.0)
    data["imu_yaw"]         = tele.get("yaw", 0.0)
    data["robot_online"]    = esp32_comm.is_robot_online()
    data["esp32_firmware"]  = tele.get("firmware", "unknown")
    data["camera_fps"]      = tele.get("fps", 0.0)
    data["camera_res"]      = tele.get("resolution", "640×480")

    _cache = data
    _cache_ts = now
    return data
