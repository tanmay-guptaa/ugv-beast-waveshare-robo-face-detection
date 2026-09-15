"""
app.py  ─  UGV Beast Face Detection Dashboard
================================================
Entry point for the Streamlit web application.
Run with:   streamlit run app.py --server.port 8501
"""

import streamlit as st
import os, json, time
from datetime import datetime

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="UGV Beast AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config
from ugv import face_engine, telemetry, camera, streamer

# ── Global CSS / theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif !important; }

/* ── Global dark background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #071120 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,180,255,0.15);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #4ade80 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    letter-spacing: 0.4px;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(14,165,233,0.5);
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
}

/* ── Cards / containers ── */
.ugv-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* ── Status badge (known / stranger) ── */
.badge-known {
    display: inline-block;
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    animation: pulse-green 2s infinite;
}
.badge-stranger {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    animation: pulse-red 2s infinite;
}
.badge-idle {
    display: inline-block;
    background: linear-gradient(135deg, #475569, #64748b);
    color: white;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.8rem;
    font-weight: 700;
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    50%       { box-shadow: 0 0 0 8px rgba(16,185,129,0); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
}

/* ── Section headers ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 12px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0ea5e9; }

/* ── st.info / warning / success / error override ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    font-weight: 500;
}

/* ── Image border ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 2px solid #1e3a5f;
}

/* ── Hide default Streamlit header/footer ── */
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:2.8rem;">🤖</div>
        <div style="font-size:1.2rem; font-weight:800; color:#38bdf8;
                    letter-spacing:1px; line-height:1.2;">UGV BEAST</div>
        <div style="font-size:0.7rem; color:#64748b; letter-spacing:2px;
                    text-transform:uppercase;">AI Face Detection Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detection Status Card ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Detection Status</div>', unsafe_allow_html=True)

    det = face_engine.get_last_detection()
    name         = det.get("name", "No Face")
    is_known     = det.get("is_known", False)
    confidence   = det.get("confidence", 0.0)
    role         = det.get("role", "—")
    access_level = det.get("access_level", "—")
    det_time     = det.get("timestamp", "—")

    if name in ("No Face", "Unknown", ""):
        badge_html = '<span class="badge-idle">⏸  Waiting</span>'
    elif is_known:
        badge_html = '<span class="badge-known">✅  Authorized</span>'
    else:
        badge_html = '<span class="badge-stranger">⚠️  Stranger</span>'

    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ugv-card" style="margin-top:10px;">
        <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
            <tr><td style="color:#94a3b8; padding:3px 0;">👤 Detected</td>
                <td style="color:#e2e8f0; font-weight:600;">{name}</td></tr>
            <tr><td style="color:#94a3b8; padding:3px 0;">🎖 Role</td>
                <td style="color:#e2e8f0;">{role}</td></tr>
            <tr><td style="color:#94a3b8; padding:3px 0;">🔐 Access</td>
                <td style="color:#e2e8f0;">{access_level}</td></tr>
            <tr><td style="color:#94a3b8; padding:3px 0;">📊 Confidence</td>
                <td style="color:#38bdf8; font-weight:700;">{confidence:.1f}%</td></tr>
            <tr><td style="color:#94a3b8; padding:3px 0;">🕐 Time</td>
                <td style="color:#e2e8f0;">{det_time}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── System Quick Stats ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">📡 System Status</div>', unsafe_allow_html=True)
    tele = telemetry.get()
    robot_online = tele.get("robot_online", False)
    demo_label   = " (DEMO)" if config.DEMO_MODE else ""

    conn_color = "#4ade80" if robot_online else "#f97316"
    conn_label = "🟢 Online" if robot_online else "🟠 Offline"

    st.markdown(f"""
    <div class="ugv-card" style="font-size:0.82rem;">
        <div style="margin-bottom:6px;">
            <span style="color:#94a3b8;">Robot</span>
            <span style="float:right; color:{conn_color}; font-weight:700;">{conn_label}{demo_label}</span>
        </div>
        <div style="margin-bottom:6px;">
            <span style="color:#94a3b8;">CPU</span>
            <span style="float:right; color:#38bdf8; font-weight:700;">{tele.get('cpu_percent', 0):.0f}%</span>
        </div>
        <div style="margin-bottom:6px;">
            <span style="color:#94a3b8;">RAM</span>
            <span style="float:right; color:#38bdf8; font-weight:700;">{tele.get('ram_percent', 0):.0f}%</span>
        </div>
        <div style="margin-bottom:6px;">
            <span style="color:#94a3b8;">Temp</span>
            <span style="float:right; color:#fb923c; font-weight:700;">{tele.get('cpu_temp', 0):.0f}°C</span>
        </div>
        <div style="margin-bottom:6px;">
            <span style="color:#94a3b8;">Battery</span>
            <span style="float:right; color:#4ade80; font-weight:700;">{tele.get('battery_voltage', 0):.2f}V</span>
        </div>
        <div>
            <span style="color:#94a3b8;">Camera FPS</span>
            <span style="float:right; color:#a78bfa; font-weight:700;">{tele.get('camera_fps', 0):.0f} fps</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── UGV IP Config ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Connection</div>', unsafe_allow_html=True)
    new_ip = st.text_input("UGV IP Address", value=config.UGV_IP,
                            help="AP mode default: 192.168.50.5")
    if new_ip != config.UGV_IP:
        config.UGV_IP      = new_ip
        config.UGV_BASE_URL = f"http://{new_ip}:{config.UGV_PORT}"
        st.success("IP updated!")

    demo_toggle = st.toggle("Demo Mode", value=config.DEMO_MODE,
                             help="Simulate robot data without hardware")
    config.DEMO_MODE = demo_toggle
    os.environ["UGV_DEMO"] = "1" if demo_toggle else "0"

    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}  |  Waveshare UGV-Beast")
    st.caption("📡 Pi 4B + ESP32  |  5MP 160° Camera")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CONTENT — Dashboard Home
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 24px 0;">
    <h1 style="font-size:2rem; font-weight:800; color:#f1f5f9; margin:0; line-height:1.2;">
        🤖 UGV Beast — Face Detection Dashboard
    </h1>
    <p style="color:#64748b; margin:6px 0 0 0; font-size:0.9rem;">
        Waveshare UGV Beast  ·  Raspberry Pi 4B + ESP32  ·  5MP 160° Camera  ·  Streamlit @ :8501
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ───────────────────────────────────────────────────────────────
tele = telemetry.get()
det  = face_engine.get_last_detection()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("👤 Detected",    det.get("name", "—"))
c2.metric("🎖 Role",        det.get("role", "—"))
c3.metric("🔐 Access",      det.get("access_level", "—"))
c4.metric("⚡ Battery",     f"{tele.get('battery_voltage', 0):.2f}V",
          delta=f"{tele.get('battery_percent', 0):.0f}%")
c5.metric("🌡 CPU Temp",    f"{tele.get('cpu_temp', 0):.0f}°C")
c6.metric("📷 Camera FPS",  f"{tele.get('camera_fps', 0):.0f}")

st.markdown("---")

# ── Architecture overview cards ───────────────────────────────────────────────
st.markdown('<div class="section-title">🏗 System Architecture</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="ugv-card" style="border-top: 3px solid #0ea5e9;">
        <div style="font-size:0.65rem; font-weight:700; color:#0ea5e9; letter-spacing:1.5px; text-transform:uppercase;">🧠 Brain</div>
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; margin:6px 0 10px 0;">Raspberry Pi 4B</div>
        <ul style="margin:0; padding-left:16px; font-size:0.8rem; color:#94a3b8; line-height:1.8;">
            <li>OpenCV Camera Capture</li>
            <li>Face Vector Matching Engine</li>
            <li>Local Folder Storage (SD Card)</li>
            <li>Hosts Streamlit Web Server</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="ugv-card" style="border-top: 3px solid #10b981;">
        <div style="font-size:0.65rem; font-weight:700; color:#10b981; letter-spacing:1.5px; text-transform:uppercase;">💪 Muscles</div>
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; margin:6px 0 10px 0;">ESP32 Unit</div>
        <ul style="margin:0; padding-left:16px; font-size:0.8rem; color:#94a3b8; line-height:1.8;">
            <li>Motor PID & Track Drive</li>
            <li>Pan/Tilt Servo Movement</li>
            <li>Battery Telemetry (INA219)</li>
            <li>IMU Feedback (ICM20948)</li>
            <li>OLED Display Output</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="ugv-card" style="border-top: 3px solid #f59e0b;">
        <div style="font-size:0.65rem; font-weight:700; color:#f59e0b; letter-spacing:1.5px; text-transform:uppercase;">🖥 UI Panel</div>
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; margin:6px 0 10px 0;">Streamlit Dashboard</div>
        <ul style="margin:0; padding-left:16px; font-size:0.8rem; color:#94a3b8; line-height:1.8;">
            <li>Live Video Feed (st.image)</li>
            <li>Real-Time Identity Recognition</li>
            <li>Multi-Angle Face Enrollment Wizard</li>
            <li>Telemetry & Hardware Sensors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Quick nav instructions ────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📖 Navigation</div>', unsafe_allow_html=True)

nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    st.markdown("""
    <div class="ugv-card">
        <div style="font-size:0.9rem; font-weight:700; color:#38bdf8; margin-bottom:10px;">Use the Pages in the sidebar ↗</div>
        <table style="width:100%; font-size:0.82rem; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #334155;">
                <td style="padding:6px 0; color:#a78bfa;">📷 Live Detection</td>
                <td style="color:#94a3b8;">Real-time camera + face recognition</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
                <td style="padding:6px 0; color:#a78bfa;">👤 Enroll Face</td>
                <td style="color:#94a3b8;">Add new persons to the database</td>
            </tr>
            <tr>
                <td style="padding:6px 0; color:#a78bfa;">📊 System Info</td>
                <td style="color:#94a3b8;">Hardware specs + live telemetry</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    persons = face_engine.all_persons()
    n_persons  = len(persons)
    n_captures = len(camera.list_captures())
    n_embeds   = sum(len(v) for v in face_engine._embeddings.values())

    st.markdown(f"""
    <div class="ugv-card">
        <div style="font-size:0.9rem; font-weight:700; color:#38bdf8; margin-bottom:10px;">📂 Database Summary</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; text-align:center;">
            <div style="background:#0f172a; border-radius:10px; padding:14px;">
                <div style="font-size:1.6rem; font-weight:800; color:#a78bfa;">{n_persons}</div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">Enrolled Persons</div>
            </div>
            <div style="background:#0f172a; border-radius:10px; padding:14px;">
                <div style="font-size:1.6rem; font-weight:800; color:#4ade80;">{n_embeds}</div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">Face Embeddings</div>
            </div>
            <div style="background:#0f172a; border-radius:10px; padding:14px;">
                <div style="font-size:1.6rem; font-weight:800; color:#38bdf8;">{n_captures}</div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">Capture Snapshots</div>
            </div>
            <div style="background:#0f172a; border-radius:10px; padding:14px;">
                <div style="font-size:1.6rem; font-weight:800; color:#fb923c;">{'✓' if config.DEMO_MODE else '●'}</div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">{'Demo Mode' if config.DEMO_MODE else 'Live Mode'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Known persons table ───────────────────────────────────────────────────────
if persons:
    st.markdown("---")
    st.markdown('<div class="section-title">👥 Enrolled Persons</div>', unsafe_allow_html=True)
    import pandas as pd
    df = pd.DataFrame(persons)[["name", "role", "access_level", "first_seen", "last_seen", "capture_count"]]
    df.columns = ["Name", "Role", "Access Level", "First Seen", "Last Seen", "Captures"]
    st.dataframe(df, use_container_width=True, hide_index=True)
