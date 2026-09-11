"""
pages/01_📷_Live_Detection.py
==============================
Real-time face detection page with zero-flicker streaming via st.fragment.

- Opens webcam (cv2.VideoCapture(0)) or Pi camera.
- Runs face detection every frame in a dedicated fragment.
- Annotates with GREEN (known) / RED (stranger) / YELLOW (detected) boxes.
- Auto-saves snapshots to /static/captures/.
- Optionally sends pan/tilt tracking commands to the robot.
"""

import streamlit as st
import cv2
import numpy as np
import time
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ugv import camera, face_engine, esp32_comm

st.set_page_config(page_title="Live Detection | UGV Beast", layout="wide", page_icon="📷")

# Shared CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #071120 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.stButton > button { background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white !important;
    border: none; border-radius: 8px; padding: 8px 18px; font-weight: 600; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(14,165,233,0.4); }
[data-testid="metric-container"] { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 12px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700 !important; }
footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.detect-banner-known {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 12px 20px;
    text-align: center;
    margin-bottom: 12px;
}
.detect-banner-stranger {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 12px 20px;
    text-align: center;
    margin-bottom: 12px;
}
.detect-banner-neutral {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 10px 16px;
    text-align: center;
    margin-bottom: 12px;
}
.section-title { font-size: 0.9rem; font-weight: 700; color: #38bdf8; letter-spacing:0.6px;
    text-transform: uppercase; margin-bottom: 10px; border-bottom: 1px solid #1e3a5f; padding-bottom: 4px; }
.cam-idle-box {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 380px; background: #0f172a; border: 2px dashed #334155; border-radius: 14px;
    color: #94a3b8; font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Camera & Detection Controls ───────────────────────────────────────
st.sidebar.markdown('<div class="section-title">⚙️ Camera & Tracking</div>', unsafe_allow_html=True)

# Camera source switcher
cam_options = ["🤖 UGV Beast Camera", "💻 Laptop Webcam"]
# Default to UGV Beast Camera when robot is configured
default_idx = 0 if not isinstance(config.CAMERA_SOURCE, int) else 1
cam_source_choice = st.sidebar.radio("Camera Source", options=cam_options, index=default_idx)

current_target_source = config.CAMERA_SOURCE if "UGV" in cam_source_choice else 0

if "active_cam_source" not in st.session_state:
    st.session_state.active_cam_source = current_target_source

# If user switches camera source in UI, restart camera thread with new source
if st.session_state.active_cam_source != current_target_source:
    st.session_state.active_cam_source = current_target_source
    camera.stop()
    time.sleep(0.3)
    camera.start(current_target_source)
    st.session_state.cam_started = True
    st.rerun()

run_detection = st.sidebar.checkbox("👁 Run Face Recognition", value=True)
auto_track = st.sidebar.checkbox("🎯 Auto Pan/Tilt Tracking", value=True)

# ── Session state init ────────────────────────────────────────────────────────
if "cam_started" not in st.session_state:
    st.session_state.cam_started = True
    camera.start(st.session_state.active_cam_source)

# ── Page header ───────────────────────────────────────────────────────────────
col_title, col_actions = st.columns([5, 1])
with col_title:
    src_label = "🤖 UGV Beast Robot Stream" if "UGV" in cam_source_choice else "💻 Laptop Webcam"
    st.markdown(f"""
    <h2 style="font-size:1.6rem; font-weight:800; color:#f1f5f9; margin-bottom:4px;">
        📷 Live Face Detection
    </h2>
    <p style="color:#64748b; font-size:0.85rem; margin-bottom:12px;">
        Connected to: <span style="color:#38bdf8; font-weight:600;">{src_label}</span>
    </p>
    """, unsafe_allow_html=True)

with col_actions:
    if st.session_state.cam_started:
        if st.button("⏹ Stop", use_container_width=True):
            camera.stop()
            st.session_state.cam_started = False
            st.rerun()
    else:
        if st.button("▶ Start", use_container_width=True):
            camera.start(st.session_state.active_cam_source)
            st.session_state.cam_started = True
            st.rerun()


# ── Live Stream Fragment (No-flicker in-place updates) ─────────────────────────
@st.fragment(run_every=0.1)
def live_stream_fragment():
    if not st.session_state.cam_started:
        st.markdown("""
        <div class="cam-idle-box">
            <span style="font-size: 2.5rem; margin-bottom: 12px;">📷</span>
            <strong style="color: #f1f5f9;">Camera is stopped</strong>
            <span style="font-size: 0.9rem; color: #64748b; margin-top: 6px;">Click <b>▶ Start</b> above to resume live feed</span>
        </div>
        """, unsafe_allow_html=True)
        return

    frame = camera.get_frame()
    if frame is None:
        st.markdown("""
        <div class="cam-idle-box">
            <span style="font-size: 2.5rem; margin-bottom: 12px;">⏳</span>
            <strong style="color: #f1f5f9;">Connecting to camera index {}...</strong>
            <span style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">Ensure webcam or UGV camera is connected</span>
        </div>
        """.format(st.session_state.get("active_cam_source", "Live")), unsafe_allow_html=True)
        return

    # Run detection
    if run_detection:
        try:
            annotated, det_info = face_engine.detect_and_annotate(frame)
        except Exception:
            annotated = frame.copy()
            det_info = face_engine.get_last_detection()
    else:
        annotated = frame.copy()
        det_info  = face_engine.get_last_detection()

    is_known = det_info.get("is_known", False)
    name     = det_info.get("name", "No Face")
    conf     = det_info.get("confidence", 0.0)

    # 🔊 Voice greeting: say "Hello [Name]" on UGV Beast speaker when recognised
    # Temporal filter: require 2 consecutive frames of the same confirmed person
    # to completely eliminate single-frame glitches and guarantee strangers are never greeted.
    if is_known and name not in ("No Face", "Unknown", "Stranger", ""):
        if st.session_state.get("_detected_streak_person") == name:
            st.session_state["_detected_streak_count"] = st.session_state.get("_detected_streak_count", 0) + 1
        else:
            st.session_state["_detected_streak_person"] = name
            st.session_state["_detected_streak_count"] = 1

        if st.session_state["_detected_streak_count"] >= 2:
            face_engine.speak_on_ugv(name)
    else:
        st.session_state["_detected_streak_count"] = 0
        st.session_state["_detected_streak_person"] = None

    # Robot pan/tilt auto-tracking
    if auto_track and det_info.get("box") is not None and not config.DEMO_MODE:
        top, right, bottom, left = det_info["box"]
        face_cx = (left + right) // 2
        face_cy = (top + bottom) // 2
        frame_cx = frame.shape[1] // 2
        frame_cy = frame.shape[0] // 2
        err_x = face_cx - frame_cx
        err_y = face_cy - frame_cy
        pan_adj  = int(config.PAN_CENTER  - err_x * 0.05)
        tilt_adj = int(config.TILT_CENTER + err_y * 0.05)
        esp32_comm.set_pan(pan_adj)
        esp32_comm.set_tilt(tilt_adj)

    # Render video + metrics
    vid_col, info_col = st.columns([3, 1])

    with vid_col:
        # Status banner above video
        if name not in ("No Face", "Unknown", ""):
            if is_known:
                st.markdown(f"""
                <div class="detect-banner-known">
                    <span style="font-size:1.1rem; font-weight:800; color:#4ade80;">✅ Authorized: {name}</span>
                    <span style="font-size:0.85rem; color:#a7f3d0; margin-left:14px;">Match: {conf:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="detect-banner-stranger">
                    <span style="font-size:1.1rem; font-weight:800; color:#fca5a5;">⚠️ Unknown Person Detected</span>
                    <span style="font-size:0.85rem; color:#fecaca; margin-left:14px;">Not in database</span>
                </div>""", unsafe_allow_html=True)
        elif name == "Unknown":
            st.markdown("""
            <div class="detect-banner-neutral">
                <span style="font-size:0.95rem; font-weight:700; color:#38bdf8;">👤 Face Detected</span>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left:10px;">Scanning features...</span>
            </div>""", unsafe_allow_html=True)

        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, channels="RGB", use_container_width=True)

    with info_col:
        st.markdown('<div class="section-title">🎯 Detection Info</div>', unsafe_allow_html=True)
        st.metric("👤 Detected", name)
        st.metric("🎖 Role", det_info.get("role", "—"))
        st.metric("🔐 Access", det_info.get("access_level", "—"))
        st.metric("📊 Match", f"{conf:.1f}%")

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Live Metrics</div>', unsafe_allow_html=True)
        st.metric("🎞 FPS", f"{camera.get_fps():.0f}")


# Run the live stream fragment
live_stream_fragment()
