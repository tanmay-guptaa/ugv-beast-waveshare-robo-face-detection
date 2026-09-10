"""
pages/05_📊_System_Info.py
===========================
Full hardware and system information page.
Displays UGV Beast technical specifications + live Pi + robot stats.
"""

import streamlit as st
import os
import sys
import platform
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ugv import telemetry, esp32_comm

st.set_page_config(page_title="System Info | UGV Beast", layout="wide", page_icon="📊")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.ugv-card { background: linear-gradient(135deg,#1e293b,#0f172a); border:1px solid #334155;
    border-radius:14px; padding:16px 20px; margin-bottom:12px; }
.section-title { font-size:0.9rem; font-weight:700; color:#38bdf8; letter-spacing:0.6px;
    text-transform:uppercase; margin-bottom:10px; border-bottom:1px solid #1e3a5f; padding-bottom:4px;}
[data-testid="metric-container"] { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:12px 16px; }
[data-testid="stMetricValue"] { color:#38bdf8 !important; font-weight:700 !important; }
.spec-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.spec-table td { padding: 7px 12px; border-bottom: 1px solid #1e3a5f; }
.spec-table td:first-child { color:#94a3b8; width:35%; }
.spec-table td:last-child  { color:#e2e8f0; font-weight:500; }
.spec-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="font-size:1.6rem; font-weight:800; color:#f1f5f9; margin-bottom:4px;">📊 System Information</h2>
<p style="color:#64748b; font-size:0.85rem; margin-bottom:16px;">
Hardware specs · Live telemetry · UGV Beast technical details
</p>
""", unsafe_allow_html=True)

tele = telemetry.get()

# ── Top metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("🖥 CPU",          f"{tele.get('cpu_percent',0):.0f}%")
m2.metric("🧠 RAM",          f"{tele.get('ram_percent',0):.0f}%")
m3.metric("💾 Disk",         f"{tele.get('disk_percent',0):.0f}%")
m4.metric("🌡 Temp",         f"{tele.get('cpu_temp',0):.0f}°C")
m5.metric("🔋 Battery",      f"{tele.get('battery_voltage',0):.2f}V")
m6.metric("📷 Camera FPS",   f"{tele.get('camera_fps',0):.0f}")

st.markdown("---")

# ── Three column layout ───────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([2, 2, 2])

# ──── Left: UGV Specs ────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">🤖 UGV Beast Technical Specs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ugv-card">
    <table class="spec-table">
        <tr><td>Type</td><td>Open-source off-road tracked AI robot</td></tr>
        <tr><td>Main Controller</td><td>Raspberry Pi 4B</td></tr>
        <tr><td>Sub-Controller</td><td>ESP32 (motor/sensor)</td></tr>
        <tr><td>Architecture</td><td>Dual-controller design</td></tr>
        <tr><td>Chassis</td><td>2 mm aluminum alloy body</td></tr>
        <tr><td>Drive</td><td>Tracked off-road platform</td></tr>
        <tr><td>Turning</td><td>0 m radius / in-place rotation</td></tr>
        <tr><td>Max Speed</td><td>0.35 m/s</td></tr>
        <tr><td>Motors</td><td>Encoder-equipped, PID control</td></tr>
        <tr><td>IMU</td><td>ICM20948 (9-axis)</td></tr>
        <tr><td>Battery Monitor</td><td>INA219 voltage/current</td></tr>
        <tr><td>Battery</td><td>3S LiPo UPS (18650 × 3)</td></tr>
        <tr><td>Interfaces</td><td>UART, I²C, Bus-Servo, GPIO</td></tr>
        <tr><td>Camera</td><td>5MP 160° wide-angle USB</td></tr>
        <tr><td>Pan/Tilt</td><td>2-DOF Bus Servo (PT variant)</td></tr>
        <tr><td>Web API</td><td>Flask on Pi @ port 5000</td></tr>
        <tr><td>Dashboard</td><td>Streamlit @ port 8501</td></tr>
        <tr><td>Wireless</td><td>Wi-Fi (AP / STA modes)</td></tr>
        <tr><td>OS</td><td>Raspberry Pi OS (Debian Bookworm)</td></tr>
        <tr><td>LED</td><td>High-brightness spotlight</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ──── Middle: Pi + live graphs ────────────────────────────────────────────────
with col_mid:
    st.markdown('<div class="section-title">🖥 Raspberry Pi 4B — Live Stats</div>', unsafe_allow_html=True)

    # CPU donut
    cpu_val = tele.get("cpu_percent", 0)
    fig_cpu = go.Figure(go.Pie(
        values=[cpu_val, 100 - cpu_val],
        labels=["Used", "Free"],
        hole=0.72,
        marker_colors=["#0ea5e9", "#1e293b"],
        textinfo="none",
    ))
    fig_cpu.add_annotation(text=f"<b>{cpu_val:.0f}%</b>", x=0.5, y=0.5,
                           font_size=22, font_color="#38bdf8", showarrow=False)
    fig_cpu.update_layout(
        title={"text": "CPU Usage", "font": {"size": 13, "color": "#94a3b8"}},
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        showlegend=False, height=200,
        margin=dict(t=40, b=5, l=5, r=5),
    )
    st.plotly_chart(fig_cpu, use_container_width=True)

    # RAM donut
    ram_val = tele.get("ram_percent", 0)
    fig_ram = go.Figure(go.Pie(
        values=[ram_val, 100 - ram_val],
        labels=["Used", "Free"],
        hole=0.72,
        marker_colors=["#a78bfa", "#1e293b"],
        textinfo="none",
    ))
    fig_ram.add_annotation(text=f"<b>{ram_val:.0f}%</b>", x=0.5, y=0.5,
                           font_size=22, font_color="#a78bfa", showarrow=False)
    fig_ram.update_layout(
        title={"text": "RAM Usage", "font": {"size": 13, "color": "#94a3b8"}},
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        showlegend=False, height=200,
        margin=dict(t=40, b=5, l=5, r=5),
    )
    st.plotly_chart(fig_ram, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🖥 Host Machine</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ugv-card" style="font-size:0.82rem;">
        <table class="spec-table">
            <tr><td>OS</td><td>{platform.system()} {platform.release()}</td></tr>
            <tr><td>Python</td><td>{platform.python_version()}</td></tr>
            <tr><td>Machine</td><td>{platform.machine()}</td></tr>
            <tr><td>Dashboard Mode</td><td>{'🟠 Demo' if config.DEMO_MODE else '🟢 Live'}</td></tr>
            <tr><td>UGV IP</td><td>{config.UGV_IP}:{config.UGV_PORT}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ──── Right: Network + file system ───────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">📡 Network & Communication</div>', unsafe_allow_html=True)
    robot_online = esp32_comm.is_robot_online() if not config.DEMO_MODE else False
    conn_color = "#4ade80" if robot_online else "#ef4444"
    st.markdown(f"""
    <div class="ugv-card">
        <div style="font-size:1.0rem; font-weight:700; color:{conn_color};">
            {'🟢 Robot Online' if robot_online else '🔴 Robot Offline / Demo Mode'}
        </div>
        <table class="spec-table" style="margin-top:10px;">
            <tr><td>Robot Base URL</td><td><code style="color:#38bdf8;">{config.UGV_BASE_URL}</code></td></tr>
            <tr><td>Protocol</td><td>HTTP JSON (Flask @ :5000)</td></tr>
            <tr><td>WiFi Mode</td><td>AP mode: <code>192.168.50.5</code></td></tr>
            <tr><td>Hotspot SSID</td><td><code>AccessPopup</code></td></tr>
            <tr><td>Hotspot Pass</td><td><code>1234567890</code></td></tr>
            <tr><td>Dashboard</td><td>Streamlit @ <code>:8501</code></td></tr>
            <tr><td>JupyterLab</td><td><code>:8888</code></td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📁 File System</div>', unsafe_allow_html=True)

    # Calculate sizes
    def dir_size(path):
        total = 0
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    try: total += os.path.getsize(os.path.join(root, f))
                    except: pass
        return total

    captures_size = dir_size(config.CAPTURES_DIR)
    dataset_size  = dir_size(config.DATASET_DIR)
    pkl_size      = os.path.getsize(config.DATABASE_PKL) if os.path.isfile(config.DATABASE_PKL) else 0
    json_size     = os.path.getsize(config.DATABASE_JSON) if os.path.isfile(config.DATABASE_JSON) else 0

    def fmt(b):
        if b > 1024*1024: return f"{b/1024/1024:.1f} MB"
        elif b > 1024: return f"{b/1024:.1f} KB"
        else: return f"{b} B"

    st.markdown(f"""
    <div class="ugv-card" style="font-size:0.82rem;">
        <table class="spec-table">
            <tr><td>Base Dir</td><td><code>{os.path.basename(config.BASE_DIR)}/</code></td></tr>
            <tr><td>Captures</td><td><code>static/captures/</code>  ·  {fmt(captures_size)}</td></tr>
            <tr><td>Known Faces</td><td><code>dataset/known_faces/</code>  ·  {fmt(dataset_size)}</td></tr>
            <tr><td>database.pkl</td><td>{fmt(pkl_size)}</td></tr>
            <tr><td>database.json</td><td>{fmt(json_size)}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🔄 Actions</div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh All Stats", use_container_width=True):
        st.rerun()
    if st.button("🏗 Rebuild Face DB", use_container_width=True):
        from ugv import face_engine
        with st.spinner("Rebuilding…"):
            n = face_engine.rebuild_from_folder()
        st.success(f"Done! {n} embeddings")
