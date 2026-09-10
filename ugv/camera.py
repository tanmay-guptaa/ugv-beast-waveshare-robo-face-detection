"""
ugv/camera.py
=============
Background camera capture thread.

- On Pi: uses cv2.VideoCapture(0) with the 5MP 160 degree USB camera.
- On Windows dev machine: uses the default webcam (also index 0).
- Thread-safe latest-frame access via get_frame().
- Auto-saves JPEG snapshots to CAPTURES_DIR on request.

Performance optimisations
-------------------------
- Windows: opens with CAP_DSHOW backend to skip multi-driver probing
  (cuts startup from ~2-3 s down to ~0.3 s).
- Buffer size set to 1 so get_frame() always returns the freshest frame.
- _ready_event fires once the first real frame is captured, so callers can
  tell the difference between "still opening" and "open but no face yet".
"""

import cv2
import threading
import time
import os
import sys
from datetime import datetime
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_frame_lock   = threading.Lock()
_latest_frame: Optional[np.ndarray] = None
_camera_running = False
_camera_thread: Optional[threading.Thread] = None
_fps          = 0.0
_frame_count  = 0
_ready_event  = threading.Event()   # set once first frame arrives


def _capture_loop(source=None):
    global _latest_frame, _camera_running, _fps, _frame_count

    if source is None:
        source = getattr(config, "CAMERA_SOURCE", 0)

    # Use DirectShow on Windows ONLY for local hardware webcams (numeric index).
    # For HTTP/RTSP URLs (e.g. UGV Beast video_feed), use default FFMPEG backend.
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
    else:
        cap = cv2.VideoCapture(str(source))

    # Resolution & frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Keep internal buffer at 1 frame so get_frame() always gets the freshest image
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    t0 = time.time()
    local_count = 0

    while _camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        with _frame_lock:
            _latest_frame = frame.copy()

        local_count += 1
        _frame_count = local_count
        elapsed = time.time() - t0
        if elapsed > 0:
            _fps = round(local_count / elapsed, 1)

        # Signal that the camera produced its first real frame
        if not _ready_event.is_set():
            _ready_event.set()

    cap.release()
    _ready_event.clear()


_active_source = None


def start(source=None):
    """Start background capture thread."""
    global _camera_running, _camera_thread, _active_source
    if source is None:
        source = getattr(config, "CAMERA_SOURCE", 0)

    # If already running with this exact source, don't restart
    if _camera_running and _active_source == source:
        return

    # If running with a different source, stop the current camera first
    if _camera_running:
        stop()
        time.sleep(0.2)

    _active_source = source
    _ready_event.clear()
    _camera_running = True
    _camera_thread = threading.Thread(target=_capture_loop, args=(source,), daemon=True)
    _camera_thread.start()


def stop():
    """Stop the capture thread and release the camera."""
    global _camera_running, _latest_frame, _active_source
    _camera_running = False
    _active_source = None
    _ready_event.clear()
    with _frame_lock:
        _latest_frame = None


def is_ready() -> bool:
    """Return True once the first frame has been captured (camera fully open)."""
    return _ready_event.is_set()


def wait_until_ready(timeout: float = 5.0) -> bool:
    """Block until the camera produces its first frame, or timeout expires."""
    return _ready_event.wait(timeout=timeout)


def get_frame() -> Optional[np.ndarray]:
    """Return the latest captured frame (or None if not started yet)."""
    with _frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()


def get_fps() -> float:
    return _fps


def save_capture(frame: np.ndarray, label: str = "capture") -> str:
    """
    Save frame as JPEG to CAPTURES_DIR.
    Returns the saved file path.
    """
    os.makedirs(config.CAPTURES_DIR, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{label}_{ts}.jpg"
    fpath = os.path.join(config.CAPTURES_DIR, fname)
    cv2.imwrite(fpath, frame)
    return fpath


def list_captures() -> list[str]:
    """Return sorted list of capture file paths (newest first)."""
    if not os.path.isdir(config.CAPTURES_DIR):
        return []
    files = [
        os.path.join(config.CAPTURES_DIR, f)
        for f in os.listdir(config.CAPTURES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return sorted(files, key=os.path.getmtime, reverse=True)
