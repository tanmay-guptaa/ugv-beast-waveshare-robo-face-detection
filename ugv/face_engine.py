"""
ugv/face_engine.py
==================
Face detection and recognition engine.

- OpenCV 5: uses YuNet ONNX DNN face detector (models/face_detection_yunet.onnx)
- Identity matching: `face_recognition` library (dlib 128-D embeddings)
- Falls back gracefully when `face_recognition` is not installed —
  bounding boxes are still drawn, but names are shown as "Unknown".

Database layout
---------------
  dataset/known_faces/<Person>/  → reference JPEG images
  dataset/database.pkl           → { name: [np.array(128,), ...] }
  database.json                  → [ { name, role, access_level, info,
                                       first_seen, last_seen, capture_count } ]
"""

import os
import cv2
import json
import pickle
import time
import threading
import requests
import numpy as np
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Optional dependency — graceful fallback
try:
    import face_recognition as fr
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False

import threading

# ── Face detector (OpenCV 5 — YuNet ONNX model) ───────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YUNET_MODEL = os.path.join(_BASE_DIR, "models", "face_detection_yunet.onnx")
_SFACE_MODEL = os.path.join(_BASE_DIR, "models", "face_recognition_sface.onnx")

_detector    = None   # initialized lazily per frame size
_recognizer  = None   # SFace 128-D embedding extractor
_cur_w, _cur_h = 0, 0
_detect_lock = threading.Lock()

# SFace cosine similarity threshold (0.48 reliably separates enrolled faces from strangers)
_SFACE_THRESHOLD = 0.48

def _get_detector(w: int, h: int):
    """Return a YuNet FaceDetectorYN configured for the given frame size (thread-safe)."""
    global _detector, _cur_w, _cur_h
    if not os.path.isfile(_YUNET_MODEL):
        return None
    try:
        if _detector is None:
            _detector = cv2.FaceDetectorYN_create(
                _YUNET_MODEL, "", (w, h),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5,
            )
            _cur_w, _cur_h = w, h
        elif (_cur_w, _cur_h) != (w, h):
            _detector.setInputSize((w, h))
            _cur_w, _cur_h = w, h
    except Exception:
        try:
            _detector = cv2.FaceDetectorYN_create(
                _YUNET_MODEL, "", (w, h),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5,
            )
            _cur_w, _cur_h = w, h
        except Exception:
            _detector = None
    return _detector


def _get_recognizer():
    """Return SFace FaceRecognizerSF for 128-D embedding extraction."""
    global _recognizer
    if _recognizer is None and os.path.isfile(_SFACE_MODEL):
        try:
            _recognizer = cv2.FaceRecognizerSF_create(_SFACE_MODEL, "")
        except Exception:
            _recognizer = None
    return _recognizer

# ── In-memory DB ──────────────────────────────────────────────────────────────
_embeddings: dict[str, list] = {}   # name → [128-D np.array, ...]
_metadata:   list[dict]      = []   # list of person records

# ── Last detection state ──────────────────────────────────────────────────────
_last_detection = {
    "name":         "No Face",
    "role":         "—",
    "access_level": "—",
    "info":         "—",
    "confidence":   0.0,
    "is_known":     False,
    "timestamp":    "",
    "box":          None,   # (top, right, bottom, left) or None
}

_last_capture_time: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_database():
    """Load embeddings and metadata from disk into memory."""
    global _embeddings, _metadata

    # Load embeddings pickle
    if os.path.isfile(config.DATABASE_PKL):
        try:
            with open(config.DATABASE_PKL, "rb") as f:
                _embeddings = pickle.load(f)
        except Exception:
            _embeddings = {}
    else:
        _embeddings = {}

    # Also load/merge from database_embeddings.json if available
    json_embs_path = os.path.join(os.path.dirname(config.DATABASE_PKL), "database_embeddings.json")
    if os.path.isfile(json_embs_path):
        try:
            with open(json_embs_path, "r", encoding="utf-8") as f:
                readable = json.load(f)
            for p_name, vecs in readable.items():
                if p_name not in _embeddings or len(_embeddings[p_name]) == 0:
                    _embeddings[p_name] = [
                        np.array(v, dtype=np.float32).reshape(1, -1) for v in vecs
                    ]
        except Exception:
            pass

    # Load JSON metadata
    if os.path.isfile(config.DATABASE_JSON):
        try:
            with open(config.DATABASE_JSON, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        except Exception:
            _metadata = []
    # Pre-generate greeting audio on the Pi in background for each known person
    for p_name in list(_embeddings.keys()):
        pregenerate_greeting_audio(p_name)


def save_database():
    """Persist in-memory DB to disk."""
    os.makedirs(os.path.dirname(config.DATABASE_PKL), exist_ok=True)
    with open(config.DATABASE_PKL, "wb") as f:
        pickle.dump(_embeddings, f)
    with open(config.DATABASE_JSON, "w") as f:
        json.dump(_metadata, f, indent=2, default=str)

    # Also save readable embeddings in JSON so user can open and view them directly
    json_embs_path = os.path.join(os.path.dirname(config.DATABASE_PKL), "database_embeddings.json")
    try:
        readable_embs = {}
        for name, vecs in _embeddings.items():
            readable_embs[name] = [
                [round(float(val), 6) for val in (vec.flatten() if hasattr(vec, "flatten") else vec)]
                for vec in vecs
            ]
        with open(json_embs_path, "w", encoding="utf-8") as f:
            json.dump(readable_embs, f, indent=2)
    except Exception:
        pass

def get_serializable_embeddings() -> dict:
    """Return in-memory embeddings as a JSON-serializable dict of float lists."""
    result = {}
    for name, vecs in _embeddings.items():
        result[name] = [
            [round(float(v), 6) for v in (vec.flatten() if hasattr(vec, "flatten") else vec)]
            for vec in vecs
        ]
    return result


def sync_to_ugv(ip: Optional[str] = None, port: Optional[int] = None, timeout: float = 4.0) -> tuple[bool, str]:
    """
    Sync embeddings and metadata to the Raspberry Pi.
    Strategy 1: Direct SFTP file copy (zero Pi-side config needed).
    Strategy 2: HTTP POST fallback if SFTP unavailable.
    """
    target_ip   = ip   or getattr(config, "UGV_IP",       "192.168.80.154")
    ssh_user    = getattr(config, "UGV_SSH_USER",  "ws")
    ssh_pass    = getattr(config, "UGV_SSH_PASS",  "ws")
    ssh_port    = getattr(config, "UGV_SSH_PORT",  22)

    # Ensure files are flushed to disk before transfer
    save_database()
    local_embs = os.path.join(os.path.dirname(config.DATABASE_PKL), "database_embeddings.json")
    local_db   = config.DATABASE_JSON

    # ── Strategy 1: Direct SFTP ──────────────────────────────────────────────
    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, port=ssh_port, username=ssh_user, password=ssh_pass, timeout=timeout)
        sftp = ssh.open_sftp()

        remote_dirs = [
            f"/home/{ssh_user}/ugv_rpi/dataset",
            f"/home/{ssh_user}/dataset",
        ]
        for r_dir in remote_dirs:
            try:
                sftp.mkdir(r_dir)
            except Exception:
                pass

        dest_embs = f"/home/{ssh_user}/ugv_rpi/dataset/database_embeddings.json"
        dest_db   = f"/home/{ssh_user}/ugv_rpi/database.json"
        sftp.put(local_embs, dest_embs)
        sftp.put(local_db,   dest_db)

        # Backup copy in home dir
        try:
            sftp.put(local_embs, f"/home/{ssh_user}/dataset/database_embeddings.json")
            sftp.put(local_db,   f"/home/{ssh_user}/database.json")
        except Exception:
            pass

        sftp.close()
        ssh.close()
        return True, f"Directly saved {len(_embeddings)} profiles to Raspberry Pi ({dest_embs})"
    except Exception:
        pass

    # ── Strategy 2: HTTP REST fallback ───────────────────────────────────────
    target_port = port or getattr(config, "UGV_PORT", 5000)
    url = f"http://{target_ip}:{target_port}/api/sync_embeddings"
    payload = {
        "embeddings": get_serializable_embeddings(),
        "metadata":   _metadata,
        "timestamp":  datetime.now().isoformat(),
        "count":      len(_embeddings),
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            try:
                msg = resp.json().get("message", f"Synced {len(_embeddings)} persons via HTTP")
            except Exception:
                msg = f"Synced {len(_embeddings)} persons via HTTP"
            return True, msg
        return False, f"Pi returned HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Could not reach UGV Pi ({target_ip}). Is the robot on and connected?"
    except Exception as e:
        return False, f"Sync error: {str(e)}"


def sync_from_ugv(ip: Optional[str] = None, timeout: float = 4.0) -> tuple[bool, str]:
    """
    Download embeddings and metadata stored on Raspberry Pi to the local dashboard.
    Enables recognizing faces using embeddings created or saved on the Pi.
    """
    target_ip = ip or getattr(config, "UGV_IP", "192.168.80.154")
    ssh_user  = getattr(config, "UGV_SSH_USER", "ws")
    ssh_pass  = getattr(config, "UGV_SSH_PASS", "ws")
    ssh_port  = getattr(config, "UGV_SSH_PORT", 22)

    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, port=ssh_port, username=ssh_user, password=ssh_pass, timeout=timeout)
        sftp = ssh.open_sftp()

        remote_embs = f"/home/{ssh_user}/ugv_rpi/dataset/database_embeddings.json"
        remote_db   = f"/home/{ssh_user}/ugv_rpi/database.json"

        local_embs = os.path.join(os.path.dirname(config.DATABASE_PKL), "database_embeddings.json")
        local_db   = config.DATABASE_JSON

        os.makedirs(os.path.dirname(local_embs), exist_ok=True)
        sftp.get(remote_embs, local_embs)
        try:
            sftp.get(remote_db, local_db)
        except Exception:
            pass

        sftp.close()
        ssh.close()

        # Reload newly pulled database into memory
        load_database()
        return True, f"Successfully loaded {len(_embeddings)} face profile(s) from Raspberry Pi"
    except Exception as exc:
        return False, f"Failed to pull embeddings from Pi: {exc}"


def inspect_ugv_storage() -> tuple[bool, str]:
    """SSH into the Pi and read back what is physically saved there."""
    target_ip = getattr(config, "UGV_IP",      "192.168.80.154")
    ssh_user  = getattr(config, "UGV_SSH_USER", "ws")
    ssh_pass  = getattr(config, "UGV_SSH_PASS", "ws")
    ssh_port  = getattr(config, "UGV_SSH_PORT", 22)
    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, port=ssh_port, username=ssh_user, password=ssh_pass, timeout=3.0)
        cmd = (
            f"echo '=== File on Pi ==='; "
            f"ls -lh /home/{ssh_user}/ugv_rpi/dataset/database_embeddings.json; "
            f"echo ''; "
            f"echo '=== Registered persons ==='; "
            f"cat /home/{ssh_user}/ugv_rpi/database.json"
        )
        _, stdout, _ = ssh.exec_command(cmd)
        output = stdout.read().decode()
        ssh.close()
        return True, output
    except Exception as e:
        return False, f"Could not reach Raspberry Pi: {str(e)}"


# ─── Voice Greeting Engine ────────────────────────────────────────────────────
# Tracks last greeting time per person to avoid repeating every frame
_last_greeted: dict[str, float] = {}
_GREET_COOLDOWN_SEC = 25.0   # seconds between greetings for the same person

def _generate_piper_wav_on_pi(target_ip: str, name: str, wav_filename: str) -> bool:
    """Generate greeting WAV file on Raspberry Pi using Piper neural TTS."""
    ssh_user = getattr(config, "UGV_SSH_USER", "ws")
    ssh_pass = getattr(config, "UGV_SSH_PASS", "ws")
    ssh_port = getattr(config, "UGV_SSH_PORT", 22)
    first_name = name.strip().split()[0] if name.strip() else name
    greeting = f"Hello {first_name}"

    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, port=ssh_port, username=ssh_user, password=ssh_pass, timeout=3.0)

        remote_wav = f"/home/{ssh_user}/ugv_rpi/sounds/greetings/{wav_filename}"
        piper_bin = f"/home/{ssh_user}/ugv_rpi/piper/piper/piper"
        piper_model = f"/home/{ssh_user}/ugv_rpi/piper/en_US-lessac-medium.onnx"
        espeak_data = f"/home/{ssh_user}/ugv_rpi/piper/piper/espeak-ng-data"
        piper_lib = f"/home/{ssh_user}/ugv_rpi/piper/piper"

        gen_cmd = (
            f"mkdir -p /home/{ssh_user}/ugv_rpi/sounds/greetings && "
            f"echo '{greeting}' | LD_LIBRARY_PATH={piper_lib} {piper_bin} "
            f"-m {piper_model} --output_file {remote_wav} --espeak_data {espeak_data}"
        )
        _, stdout, _ = ssh.exec_command(gen_cmd)
        exit_code = stdout.channel.recv_exit_status()
        ssh.close()
        return exit_code == 0
    except Exception as exc:
        print(f"[TTS] Error generating Piper WAV on Pi: {exc}")
        return False


def _run_tts_on_pi(name: str) -> None:
    """
    Execute speech greeting on the UGV Beast onboard speaker exactly ONCE.
    Uses ALSA plug:dmix_out directly over SSH for clean, conflict-free audio output.
    1. Plays pre-generated WAV /home/ws/ugv_rpi/sounds/greetings/greet_{name}.wav
    2. If not generated yet, generates it with Piper and plays it.
    3. Fast fallback: espeak piped to aplay -D plug:dmix_out.
    Runs in a background thread so the camera stream is never delayed.
    """
    target_ip = getattr(config, "UGV_IP", "192.168.80.154")
    ssh_user = getattr(config, "UGV_SSH_USER", "ws")
    ssh_pass = getattr(config, "UGV_SSH_PASS", "ws")
    ssh_port = getattr(config, "UGV_SSH_PORT", 22)

    first_name = name.strip().split()[0] if name.strip() else name
    safe_name = "".join(c for c in first_name.lower() if c.isalnum())
    wav_filename = f"greet_{safe_name}.wav"
    greeting = f"Hello {first_name}"
    remote_wav = f"/home/{ssh_user}/ugv_rpi/sounds/greetings/{wav_filename}"

    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, port=ssh_port, username=ssh_user, password=ssh_pass, timeout=3.0)

        # 1. Primary: If the WAV exists, play it instantly via plug:dmix_out
        chk_cmd = f"test -f {remote_wav} && aplay -q -D plug:dmix_out {remote_wav}"
        _, stdout, _ = ssh.exec_command(chk_cmd)
        if stdout.channel.recv_exit_status() == 0:
            print(f"[TTS] Played greeting for '{name}' via robot speaker (aplay dmix)")
            ssh.close()
            return

        # 2. If WAV doesn't exist, generate it with Piper and play
        piper_bin = f"/home/{ssh_user}/ugv_rpi/piper/piper/piper"
        piper_model = f"/home/{ssh_user}/ugv_rpi/piper/en_US-lessac-medium.onnx"
        espeak_data = f"/home/{ssh_user}/ugv_rpi/piper/piper/espeak-ng-data"
        piper_lib = f"/home/{ssh_user}/ugv_rpi/piper/piper"

        gen_and_play = (
            f"mkdir -p /home/{ssh_user}/ugv_rpi/sounds/greetings && "
            f"echo '{greeting}' | LD_LIBRARY_PATH={piper_lib} {piper_bin} "
            f"-m {piper_model} --output_file {remote_wav} --espeak_data {espeak_data} && "
            f"aplay -q -D plug:dmix_out {remote_wav}"
        )
        _, stdout, _ = ssh.exec_command(gen_and_play)
        if stdout.channel.recv_exit_status() == 0:
            print(f"[TTS] Generated and played greeting for '{name}' via Piper on robot")
            ssh.close()
            return

        # 3. Fast fallback: espeak directly via plug:dmix_out
        espeak_cmd = f"espeak '{greeting}' --stdout | aplay -q -D plug:dmix_out"
        _, stdout, _ = ssh.exec_command(espeak_cmd)
        if stdout.channel.recv_exit_status() == 0:
            print(f"[TTS] Played greeting for '{name}' via espeak fallback")
            ssh.close()
            return

        ssh.close()
    except Exception as exc:
        print(f"[TTS] SSH audio execution failed: {exc}")
        # Last resort fallback: UDP port 5055 to voice_cmd.py
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(greeting.encode("utf-8"), (target_ip, 5055))
            sock.close()
            print(f"[TTS] Sent greeting to UDP fallback 5055")
        except Exception:
            pass


def pregenerate_greeting_audio(name: str) -> None:
    """Pre-generate the greeting WAV on the robot so playback is instant."""
    def _task():
        target_ip = getattr(config, "UGV_IP", "192.168.80.154")
        first_name = name.strip().split()[0] if name.strip() else name
        safe_name = "".join(c for c in first_name.lower() if c.isalnum())
        wav_filename = f"greet_{safe_name}.wav"
        _generate_piper_wav_on_pi(target_ip, name, wav_filename)
    threading.Thread(target=_task, daemon=True).start()


def speak_on_ugv(name: str) -> bool:
    """
    Say 'Hello [Name]' on the UGV Beast's onboard speaker if cooldown has passed.
    Call this each time a known face is detected.
    Returns True if greeting was triggered, False if cooldown still active.
    """
    if not name or name in ("Stranger", "Unknown", "No Face", ""):
        return False

    now = time.time()
    last = _last_greeted.get(name, 0.0)
    if now - last < _GREET_COOLDOWN_SEC:
        return False   # Still within cooldown — do not repeat

    _last_greeted[name] = now

    # Fire-and-forget in background thread — does NOT block the camera loop
    t = threading.Thread(target=_run_tts_on_pi, args=(name,), daemon=True)
    t.start()
    return True


def get_metadata(name: str) -> Optional[dict]:
    """Return the metadata record for a person by name."""
    for rec in _metadata:
        if rec.get("name", "").lower() == name.lower():
            return rec
    return None


def all_persons() -> list[dict]:
    return list(_metadata)


def add_person(name: str, role: str = "User",
               access_level: str = "Standard",
               info: str = "") -> dict:
    """Add or update a person record in the JSON metadata."""
    rec = get_metadata(name)
    if rec is None:
        rec = {
            "name":          name,
            "role":          role,
            "access_level":  access_level,
            "info":          info,
            "first_seen":    datetime.now().isoformat(),
            "last_seen":     datetime.now().isoformat(),
            "capture_count": 0,
        }
        _metadata.append(rec)
    else:
        rec["role"]         = role
        rec["access_level"] = access_level
        rec["info"]         = info
    save_database()
    return rec


def enroll_images(name: str, image_paths: list[str]) -> int:
    """
    Compute 128-D face embeddings for uploaded images and add to database.pkl.
    Uses OpenCV SFace (128-D) or face_recognition (128-D).

    Images are NOT stored on disk — only the compact 128-D embedding vector
    is saved (512 bytes per image vs ~80 KB per JPEG). This keeps disk usage
    negligible regardless of how many users or photos are enrolled.

    Returns the number of embeddings added.
    """
    added = 0
    if name not in _embeddings:
        _embeddings[name] = []

    rec = _get_recognizer()

    for path in image_paths:
        if not os.path.isfile(path):
            continue

        embedding_extracted = False

        # Method 1: OpenCV SFace (native ONNX, fast, no dlib needed)
        if rec is not None:
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                with _detect_lock:
                    det = _get_detector(w, h)
                    if det is not None:
                        try:
                            _, faces = det.detect(img)
                            if faces is not None and len(faces) > 0:
                                # Pick the most prominent face
                                face = max(faces, key=lambda f: f[2] * f[3])
                                aligned = rec.alignCrop(img, face)
                                feat = rec.feature(aligned)
                                if feat is not None:
                                    _embeddings[name].append(feat)
                                    added += 1
                                    embedding_extracted = True
                        except Exception:
                            pass

        # Method 2: face_recognition (dlib) fallback
        if not embedding_extracted and _FR_AVAILABLE:
            try:
                img_fr = fr.load_image_file(path)
                locs = fr.face_locations(img_fr, model=config.FACE_MODEL)
                if locs:
                    enc_list = fr.face_encodings(img_fr, locs)
                    if enc_list:
                        _embeddings[name].append(enc_list[0])
                        added += 1
                        embedding_extracted = True
            except Exception:
                pass

        # Delete the image after extracting its embedding — only the vector is needed.
        # This keeps disk usage at ~512 bytes per image instead of ~80 KB per JPEG.
        try:
            os.remove(path)
        except Exception:
            pass

    save_database()
    return added


def enroll_frame(name: str, frame: np.ndarray, filename: str = None) -> bool:
    """
    Extract 128-D embedding from an in-memory BGR frame and enroll for the given person.
    The raw frame is NOT saved to disk — only the compact 128-D embedding vector
    is persisted (512 bytes vs ~80 KB per JPEG). Zero disk growth from images.
    Returns True if face was detected and enrolled.
    """
    if frame is None or frame.size == 0:
        return False

    rec = _get_recognizer()
    h, w = frame.shape[:2]
    feat = None

    if rec is not None:
        with _detect_lock:
            det = _get_detector(w, h)
            if det is not None:
                try:
                    _, faces = det.detect(frame)
                    if faces is not None and len(faces) > 0:
                        face = max(faces, key=lambda f: f[2] * f[3])
                        aligned = rec.alignCrop(frame, face)
                        feat = rec.feature(aligned)
                except Exception:
                    pass

    if feat is None and _FR_AVAILABLE:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = fr.face_locations(rgb, model=config.FACE_MODEL)
            if locs:
                enc_list = fr.face_encodings(rgb, locs)
                if enc_list:
                    feat = enc_list[0]
        except Exception:
            pass

    if feat is not None:
        if name not in _embeddings:
            _embeddings[name] = []
        _embeddings[name].append(feat)
        # Only the embedding is saved — no image written to disk.
        save_database()
        return True

    return False


def rebuild_from_folder() -> int:
    """
    Scan dataset/known_faces/<Name>/*.jpg and rebuild database.pkl.
    Returns total embeddings built.
    """
    total = 0
    _embeddings.clear()

    if not os.path.isdir(config.DATASET_DIR):
        return 0

    for person_name in os.listdir(config.DATASET_DIR):
        person_dir = os.path.join(config.DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        images = [
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        added = enroll_images(person_name, images)
        total += added
        # Ensure metadata exists
        if get_metadata(person_name) is None:
            add_person(person_name)

    save_database()
    return total


# ─────────────────────────────────────────────────────────────────────────────
#  Detection & Recognition
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_annotate(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run face detection + 128-D recognition on a BGR frame.

    Returns
    -------
    annotated_frame : np.ndarray  (BGR, same size as input)
    detection_info  : dict        (name, role, confidence, is_known, box)
    """
    global _last_detection, _last_capture_time, _detector

    annotated = frame.copy()
    info = dict(_last_detection)  # start from last state
    h_fr, w_fr = frame.shape[:2]

    with _detect_lock:
        det = _get_detector(w_fr, h_fr)
        rec = _get_recognizer()
        faces_raw = None

        if det is not None:
            try:
                _, faces_raw = det.detect(frame)
            except Exception:
                _detector = None

        if faces_raw is not None and len(faces_raw) > 0:
            for face in faces_raw:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

                best_name = "Stranger"
                best_sim  = -1.0
                is_known  = False

                # 1. Extract 128-D embedding with SFace & match
                if rec is not None:
                    try:
                        aligned = rec.alignCrop(frame, face)
                        live_feat = rec.feature(aligned)

                        for person_name, ref_encs in _embeddings.items():
                            for ref_feat in ref_encs:
                                if isinstance(ref_feat, np.ndarray) and ref_feat.shape == live_feat.shape:
                                    sim = float(rec.match(live_feat, ref_feat, cv2.FaceRecognizerSF_FR_COSINE))
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_name = person_name
                    except Exception:
                        pass

                # 2. Fallback: dlib face_recognition embeddings
                elif _FR_AVAILABLE and _embeddings:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        loc = (max(0, y), min(w_fr, x + w), min(h_fr, y + h), max(0, x))
                        encs = fr.face_encodings(rgb, [loc])
                        if encs:
                            for person_name, ref_encs in _embeddings.items():
                                for ref_feat in ref_encs:
                                    dist = float(fr.face_distance([ref_feat], encs[0])[0])
                                    sim = 1.0 - dist
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_name = person_name
                    except Exception:
                        pass

                # 3. Decision based on cosine similarity threshold
                if best_sim >= _SFACE_THRESHOLD and best_name != "Stranger":
                    is_known = True
                    confidence = round(min(99.9, 50.0 + (best_sim - _SFACE_THRESHOLD) / (1.0 - _SFACE_THRESHOLD) * 49.9), 1)
                    color = (0, 220, 80)   # GREEN
                    label = f"{best_name} (Match)  {confidence}%"
                elif _embeddings:
                    is_known = False
                    confidence = round(max(0.0, best_sim * 100), 1) if best_sim > 0 else 0.0
                    color = (0, 60, 220)   # RED
                    label = "Stranger Detected"
                else:
                    is_known = False
                    confidence = 0.0
                    color = (0, 200, 255)  # YELLOW
                    label = "Face Detected (No DB)"

                # Draw bounding box & banner
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(annotated, (x, y + h - 28), (x + w, y + h), color, cv2.FILLED)
                cv2.putText(annotated, label, (x + 4, y + h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                meta = get_metadata(best_name) if is_known else {}
                info = {
                    "name":         best_name if is_known else "Stranger",
                    "role":         meta.get("role", "Unknown") if is_known else "—",
                    "access_level": meta.get("access_level", "Denied") if is_known else "Denied",
                    "info":         meta.get("info", "") if is_known else "",
                    "confidence":   confidence,
                    "is_known":     is_known,
                    "timestamp":    datetime.now().strftime("%H:%M:%S"),
                    "box":          (y, x + w, y + h, x),
                }
                _last_detection = info

        elif faces_raw is None or len(faces_raw) == 0:
            if info.get("name") not in ("No Face", ""):
                info = {
                    "name":         "No Face",
                    "role":         "—",
                    "access_level": "—",
                    "info":         "—",
                    "confidence":   0.0,
                    "is_known":     False,
                    "timestamp":    datetime.now().strftime("%H:%M:%S"),
                    "box":          None,
                }
                _last_detection = info

    # Overlay timestamp + system info
    ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(annotated, f"UGV-Beast  |  {ts_str}",
                (8, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return annotated, info


def get_last_detection() -> dict:
    return dict(_last_detection)


# Load DB at import time
load_database()
