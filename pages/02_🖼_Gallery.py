"""
pages/02_🖼_Gallery.py
========================
Captured Images Gallery — shows all snapshots from /static/captures/
in a responsive grid with metadata overlay.
"""

import streamlit as st
import os
import sys
from PIL import Image
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ugv import camera

st.set_page_config(page_title="Gallery | UGV Beast", layout="wide", page_icon="🖼")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.stButton > button { background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white !important;
    border: none; border-radius: 8px; padding: 8px 18px; font-weight: 600; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-2px); }
footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.gallery-label {
    background: rgba(15,23,42,0.85);
    border-radius: 0 0 10px 10px;
    padding: 6px 10px;
    font-size: 0.72rem;
    color: #94a3b8;
    text-align: center;
}
.section-title { font-size:0.9rem; font-weight:700; color:#38bdf8; letter-spacing:0.6px;
    text-transform:uppercase; margin-bottom:10px; border-bottom:1px solid #1e3a5f; padding-bottom:4px;}
.ugv-card { background: #1e293b; border:1px solid #334155; border-radius:14px; padding:16px 20px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="font-size:1.6rem; font-weight:800; color:#f1f5f9; margin-bottom:4px;">🖼 Captures Gallery</h2>
<p style="color:#64748b; font-size:0.85rem; margin-bottom:16px;">
All face snapshots saved during detection · Sorted newest first
</p>
""", unsafe_allow_html=True)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filter")
    filter_text = st.text_input("Search by name", placeholder="e.g. vikas, stranger")
    cols_count  = st.slider("Columns", 2, 6, 4)
    st.markdown("---")
    if st.button("🗑 Delete ALL Captures", use_container_width=True):
        files = camera.list_captures()
        for f in files:
            try: os.remove(f)
            except: pass
        st.success(f"Deleted {len(files)} files")
        st.rerun()
    st.markdown("---")
    st.caption(f"📁 {config.CAPTURES_DIR}")

# ── Load captures ─────────────────────────────────────────────────────────────
captures = camera.list_captures()

if filter_text.strip():
    captures = [f for f in captures if filter_text.lower() in os.path.basename(f).lower()]

# ── Stats row ─────────────────────────────────────────────────────────────────
total = len(captures)
known_count = sum(1 for f in captures if "stranger" not in os.path.basename(f).lower()
                  and os.path.basename(f).lower() != "capture")
stranger_count = sum(1 for f in captures if "stranger" in os.path.basename(f).lower())

m1, m2, m3, m4 = st.columns(4)
m1.metric("📸 Total",    total)
m2.metric("✅ Known",    known_count)
m3.metric("⚠️ Strangers", stranger_count)
m4.metric("📁 Folder",   config.CAPTURES_DIR.split(os.sep)[-2] + "/" + config.CAPTURES_DIR.split(os.sep)[-1])

st.markdown("---")

if not captures:
    st.markdown("""
    <div style="text-align:center; padding:60px 0; color:#475569;">
        <div style="font-size:3rem;">📂</div>
        <div style="font-size:1.1rem; font-weight:600; margin-top:12px;">No captures yet</div>
        <div style="font-size:0.85rem; margin-top:6px;">
            Go to <b>Live Detection</b> to start detecting and auto-saving faces.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Grid display ──────────────────────────────────────────────────────────────
grid_cols = st.columns(cols_count)

for idx, fpath in enumerate(captures):
    with grid_cols[idx % cols_count]:
        fname = os.path.basename(fpath)
        # Parse label from filename: label_YYYYMMDD_HHMMSS.jpg
        parts = fname.rsplit("_", 2)
        label = parts[0].replace("_", " ").title() if len(parts) >= 3 else fname

        # Thumbnail color-code border
        is_stranger = "stranger" in fname.lower()
        border_color = "#ef4444" if is_stranger else "#10b981"

        try:
            img = Image.open(fpath)
            img.thumbnail((400, 300))
            st.image(img, use_container_width=True)
        except Exception:
            st.error(f"Cannot load {fname}")
            continue

        # Metadata label
        mtime = os.path.getmtime(fpath)
        ts    = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = os.path.getsize(fpath) // 1024
        st.markdown(f"""
        <div class="gallery-label"
             style="border: 1px solid {border_color}; border-top:none;">
            <b style="color:{'#ef4444' if is_stranger else '#4ade80'};">
                {'⚠️ Stranger' if is_stranger else '✅ ' + label}
            </b><br>
            {ts}  ·  {size_kb}KB
        </div>
        """, unsafe_allow_html=True)

        # Delete button
        if st.button(f"🗑", key=f"del_{idx}", help=f"Delete {fname}"):
            try:
                os.remove(fpath)
                st.success("Deleted")
                st.rerun()
            except Exception as e:
                st.error(str(e))
