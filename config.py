"""
UGV Beast Dashboard Configuration
-----------------------------------
Edit UGV_IP to match your robot's IP shown on the OLED screen.
  - AP (hotspot) mode default : 192.168.50.5
  - STA (home WiFi) mode      : whatever IP your router assigned

The robot's web API runs on port 5000.
Streamlit dashboard runs on port 8501.
"""

import os

# ─── Load .env file ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.isfile(_env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # Fallback parser if python-dotenv is not present
        with open(_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

# ─── Robot Connection ─────────────────────────────────────────────────────────
UGV_IP   = os.environ.get("UGV_IP",   "192.168.50.5")
UGV_PORT = int(os.environ.get("UGV_PORT", "5000"))
UGV_BASE_URL = f"http://{UGV_IP}:{UGV_PORT}"

# ─── Demo / Dev mode ──────────────────────────────────────────────────────────
# Set True when running on Windows without the Pi nearby.
DEMO_MODE = os.environ.get("UGV_DEMO", "1") == "1"

# ─── Streamlit server ─────────────────────────────────────────────────────────
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8501"))

# ─── Camera Configuration ─────────────────────────────────────────────────────
_cam_env = os.environ.get("CAMERA_SOURCE", os.environ.get("CAMERA_INDEX", "")).strip()
if not _cam_env:
    CAMERA_SOURCE = f"http://{UGV_IP}:{UGV_PORT}/video_feed" if not DEMO_MODE else 0
elif _cam_env.isdigit():
    CAMERA_SOURCE = int(_cam_env)
else:
    CAMERA_SOURCE = _cam_env

DEFAULT_CAMERA_INDEX = CAMERA_SOURCE

# ─── File System Paths ────────────────────────────────────────────────────────
DATASET_DIR       = os.path.join(BASE_DIR, "dataset", "known_faces")
DATABASE_PKL      = os.path.join(BASE_DIR, "dataset", "database.pkl")
CAPTURES_DIR      = os.path.join(BASE_DIR, "static", "captures")
DATABASE_JSON     = os.path.join(BASE_DIR, "database.json")

# ─── Face Recognition ─────────────────────────────────────────────────────────
FACE_MATCH_TOLERANCE   = float(os.environ.get("FACE_MATCH_TOLERANCE", "0.55"))   # lower = stricter (0.0–1.0)
FACE_MODEL             = os.environ.get("FACE_MODEL", "hog")  # "hog" (fast, CPU) or "cnn" (accurate, GPU)
CAPTURE_COOLDOWN_SEC   = int(os.environ.get("CAPTURE_COOLDOWN_SEC", "3"))      # seconds between saves

# ─── Pan/Tilt defaults ────────────────────────────────────────────────────────
PAN_CENTER  = int(os.environ.get("PAN_CENTER", "90"))   # degrees
TILT_CENTER = int(os.environ.get("TILT_CENTER", "90"))   # degrees
PAN_MIN     = 0
PAN_MAX     = 180
TILT_MIN    = 30
TILT_MAX    = 150

# ─── Motor speed ──────────────────────────────────────────────────────────────
DEFAULT_SPEED = float(os.environ.get("DEFAULT_SPEED", "0.5"))   # 0.0 – 1.0

# ─── Known person displayed in demo sidebar ───────────────────────────────────
DEMO_PERSON = {
    "name":         "Vikas",
    "role":         "Hackathon Team Leader",
    "access_level": "Authorized Admin",
    "info":         "Primary authorized user of UGV Beast",
}
