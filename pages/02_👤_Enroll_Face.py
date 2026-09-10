"""
pages/03_👤_Enroll_Face.py
===========================
Multi-Image Face Enrollment Wizard.

Supports:
1. 📷 Live Camera Wizard: Guided multi-angle capture (Center, Left, Right, Up, Smile)
2. 📁 Batch Multi-Photo Upload: Upload 5–20 reference photos at once
3. Generates and stores 128-D SFace embeddings for all N samples
"""

import streamlit as st
import cv2
import os
import sys
import shutil
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ugv import face_engine, camera

st.set_page_config(page_title="Enroll Face Wizard | UGV Beast", layout="wide", page_icon="👤")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.stButton > button { background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white !important;
    border: none; border-radius: 8px; padding: 10px 22px; font-weight: 600; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(14,165,233,0.4); }
footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.ugv-card { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155;
    border-radius: 14px; padding: 18px 22px; margin-bottom: 12px; }
.section-title { font-size:0.9rem; font-weight:700; color:#38bdf8; letter-spacing:0.6px;
    text-transform:uppercase; margin-bottom:10px; border-bottom:1px solid #1e3a5f; padding-bottom:4px;}
.angle-tag {
    display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 600; margin: 3px; background: #1e293b; border: 1px solid #38bdf8; color: #38bdf8;
}
.wizard-step-header {
    font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State for Wizard ─────────────────────────────────────────────────────
if "wizard_samples" not in st.session_state:
    st.session_state.wizard_samples = []  # list of BGR np.ndarray

if "wizard_cam_on" not in st.session_state:
    st.session_state.wizard_cam_on = False

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0   # current guided angle (0–4)

# ── Angle step definitions ─────────────────────────────────────────────────────
ANGLE_STEPS = [
    {"emoji": "🧑", "label": "Center Front",   "key": "center",   "hint": "Face the camera straight on, keep head level"},
    {"emoji": "👈", "label": "Turn Left (20°)",  "key": "left",     "hint": "Slowly turn your face to YOUR left (~20°)"},
    {"emoji": "👉", "label": "Turn Right (20°)", "key": "right",    "hint": "Slowly turn your face to YOUR right (~20°)"},
    {"emoji": "👆", "label": "Tilt Up",         "key": "tilt_up",  "hint": "Tilt your chin slightly upward"},
]

def _analyse_pose(face_row, frame_w, frame_h):
    """
    Estimate head yaw & pitch from YuNet (FaceDetectorYN) landmark output.

    OpenCV FaceDetectorYN column layout (15 values per face):
      [0]  x1        bounding-box left
      [1]  y1        bounding-box top
      [2]  w         bounding-box width
      [3]  h         bounding-box height
      [4]  x_re      right-eye x         ← landmarks START at index 4
      [5]  y_re      right-eye y
      [6]  x_le      left-eye x
      [7]  y_le      left-eye y
      [8]  x_nt      nose-tip x
      [9]  y_nt      nose-tip y
      [10] x_rcm     right-mouth x
      [11] y_rcm     right-mouth y
      [12] x_lcm     left-mouth x
      [13] y_lcm     left-mouth y
      [14] score     detection confidence  ← score at the END

    Returns (yaw, pitch):
      yaw  > 0  → face turned to person's LEFT  (nose drifts image-right)
      yaw  < 0  → face turned to person's RIGHT
      pitch < 0.24 → face tilted UP
    """
    re_x, re_y = float(face_row[4]),  float(face_row[5])   # right eye
    le_x, le_y = float(face_row[6]),  float(face_row[7])   # left eye
    n_x,  n_y  = float(face_row[8]),  float(face_row[9])   # nose tip
    bh         = float(face_row[3])                        # bounding-box height

    eye_mid_x   = (re_x + le_x) / 2
    eye_dist    = float(np.hypot(re_x - le_x, re_y - le_y))
    yaw         = (n_x - eye_mid_x) / eye_dist if eye_dist > 0 else 0.0

    eye_mid_y     = (re_y + le_y) / 2
    # tilt metric: ratio of (eye→nose vertical distance) to bounding-box height.
    # When looking straight: ~0.28–0.38
    # When tilting chin UP:  ~0.15–0.25  (nose appears closer to eyes)
    eye_nose_frac = (n_y - eye_mid_y) / bh if bh > 0 else 0.3

    return yaw, eye_nose_frac


def _pose_ok(step_key, yaw, pitch, face_box, frame_w, frame_h):
    """
    Return True if the current head pose matches the required angle.
    Thresholds are intentionally lenient — guide the user, don't frustrate them.
    """
    fx, fy, fw, fh = face_box
    face_cx = fx + fw / 2

    if step_key == "center":
        # Accept face anywhere in middle 80% of frame, yaw within ±0.28
        centered = abs(face_cx - frame_w / 2) < frame_w * 0.40
        return centered and abs(yaw) < 0.28

    if step_key == "left":
        return yaw > 0.15       # nose drifts image-right when turning own left

    if step_key == "right":
        return yaw < -0.18      # nose drifts image-left when turning own right

    if step_key == "tilt_up":
        # eye_nose_frac < 0.24 means nose is very close to eyes → head tilted up
        return pitch < 0.24

    return False


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h2 style="font-size:1.6rem; font-weight:800; color:#f1f5f9; margin-bottom:4px;">
    👤 Multi-Image Face Enrollment Wizard
</h2>
<p style="color:#64748b; font-size:0.85rem; margin-bottom:16px;">
    Capture multiple angles (Front, Left, Right, Tilt) for robust, highly-accurate 128-D face detection.
</p>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
wizard_col, db_col = st.columns([3, 2])

with wizard_col:
    st.markdown("---")

    # Step 1: User Profile
    st.markdown('<div class="wizard-step-header">1️⃣ Enter Person Profile</div>', unsafe_allow_html=True)
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        p_name = st.text_input("Full Name *", placeholder="Enter full name...", key="w_name")
    with p_col2:
        p_role = st.text_input("Role", placeholder="e.g. Software Developer / Admin", key="w_role")

    p_info = st.text_input("Additional Notes (Optional)", placeholder="Department, permissions, etc.", key="w_info")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE CAMERA WIZARD
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="wizard-step-header">2️⃣ Multi-Angle Camera Capture</div>', unsafe_allow_html=True)


    # ── Dynamic step progress pills ────────────────────────────────────────────
    cur_step = st.session_state.wizard_step
    pills_html = '<div style="margin-bottom:12px; display:flex; flex-wrap:wrap; gap:6px;">'
    for i, step in enumerate(ANGLE_STEPS):
        n_done = len(st.session_state.wizard_samples)
        if i < n_done:                        # already captured
            bg, border, color = "#064e3b", "#10b981", "#4ade80"
            icon = "✓ "
        elif i == cur_step % len(ANGLE_STEPS): # current target
            bg, border, color = "#1e3a5f", "#38bdf8", "#e2e8f0"
            icon = "▶ "
        else:
            bg, border, color = "#1e293b", "#475569", "#64748b"
            icon = ""
        pills_html += (f'<span style="display:inline-flex;align-items:center;padding:5px 12px;'
                       f'border-radius:20px;font-size:0.75rem;font-weight:600;margin:2px;'
                       f'background:{bg};border:1px solid {border};color:{color};">')
        pills_html += f'{icon}{i+1}. {step["emoji"]} {step["label"]}</span>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    # Guidance hint for current step
    if st.session_state.wizard_cam_on and cur_step < len(ANGLE_STEPS):
        step_info = ANGLE_STEPS[cur_step % len(ANGLE_STEPS)]
        st.markdown(
            f'<div style="background:#1e293b;border-left:3px solid #38bdf8;padding:8px 14px;'
            f'border-radius:6px;font-size:0.82rem;color:#cbd5e1;margin-bottom:8px;">'
            f'<b>Step {cur_step+1}:</b> {step_info["hint"]}</div>',
            unsafe_allow_html=True
        )


    # Camera toggle
    cam_col1, cam_col2 = st.columns([2, 2])
    with cam_col1:
        if not st.session_state.wizard_cam_on:
            if st.button("▶ Open Camera", use_container_width=True):
                camera.stop()
                time.sleep(0.15)
                camera.start(0)  # Always use laptop webcam (index 0) for enrollment
                st.session_state.wizard_cam_on = True
                st.rerun()
        else:
            if st.button("⏹ Stop Camera", use_container_width=True):
                camera.stop()
                st.session_state.wizard_cam_on = False
                st.rerun()

    with cam_col2:
        if st.button("📸 Capture Angle", use_container_width=True, disabled=not st.session_state.wizard_cam_on):
            st.session_state["_do_capture"] = True

    # Ensure laptop camera index 0 is active when wizard camera is on
    if st.session_state.wizard_cam_on and getattr(camera, "_active_source", None) != 0:
        camera.start(0)

    # ── Live Camera Fragment with Pose-Guided Validation ─────────────────────
    @st.fragment(run_every=0.1 if st.session_state.wizard_cam_on else None)
    def enroll_cam_fragment():
        if not st.session_state.wizard_cam_on:
            return

        if not camera.is_ready():
            with st.spinner("⚡ Opening camera... please wait."):
                camera.wait_until_ready(timeout=5.0)
            st.rerun()
            return

        cur_frame = camera.get_frame()
        if cur_frame is None:
            st.info("⏳ Connecting to camera...")
            return

        h_f, w_f = cur_frame.shape[:2]
        disp_frame = cur_frame.copy()

        # ─ Detect face + landmarks ──────────────────────────────────────────
        det = face_engine._get_detector(w_f, h_f)
        has_face = False
        pose_good = False
        best_face = None

        if det is not None:
            try:
                _, f_raw = det.detect(cur_frame)
                if f_raw is not None and len(f_raw) > 0:
                    has_face = True
                    best_face = f_raw[0]   # largest / most confident face

                    # ─ Pose analysis using landmarks ───────────────────────
                    step_idx = st.session_state.wizard_step % len(ANGLE_STEPS)
                    step_key = ANGLE_STEPS[step_idx]["key"]
                    yaw, pitch = _analyse_pose(best_face, w_f, h_f)
                    face_box = (int(best_face[0]), int(best_face[1]),
                                int(best_face[2]), int(best_face[3]))
                    pose_good = _pose_ok(step_key, yaw, pitch, face_box, w_f, h_f)

                    # ─ Draw colored box: GREEN = correct, RED = wrong pose ─
                    bx, by, bw, bh = face_box
                    box_color = (0, 220, 80) if pose_good else (0, 60, 220)  # BGR

                    cv2.rectangle(disp_frame,
                                  (bx, by), (bx + bw, by + bh),
                                  box_color, 2)

                    label = "Good Pose!" if pose_good else ANGLE_STEPS[step_idx]["hint"][:32]
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(disp_frame,
                                  (bx, by - th - 10), (bx + tw + 6, by),
                                  box_color, -1)
                    cv2.putText(disp_frame, label, (bx + 3, by - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

                    # Dynamic debug overlay tailored to the active step
                    if step_key == "center":
                        debug_txt = f"yaw={yaw:+.2f}  (need |yaw|<0.28)"
                    elif step_key == "left":
                        debug_txt = f"yaw={yaw:+.2f}  (need >+0.15)"
                    elif step_key == "right":
                        debug_txt = f"yaw={yaw:+.2f}  (need <-0.18)"
                    elif step_key == "tilt_up":
                        debug_txt = f"tilt={pitch:.3f}  (need <0.24)"
                    else:
                        debug_txt = f"yaw={yaw:+.2f}  tilt={pitch:.3f}"

                    (dtw, dth), _ = cv2.getTextSize(debug_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                    cv2.rectangle(disp_frame, (6, h_f - dth - 14), (6 + dtw + 14, h_f - 4), (20, 20, 20), -1)
                    cv2.putText(disp_frame, debug_txt, (12, h_f - 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

                    # Draw 5 landmarks as dots (indices 4-13)
                    lm_pts = [
                        (int(best_face[4]),  int(best_face[5])),   # right eye
                        (int(best_face[6]),  int(best_face[7])),   # left eye
                        (int(best_face[8]),  int(best_face[9])),   # nose tip
                        (int(best_face[10]), int(best_face[11])),  # right mouth
                        (int(best_face[12]), int(best_face[13])),  # left mouth
                    ]
                    for pt in lm_pts:
                        cv2.circle(disp_frame, pt, 4, box_color, -1)

            except Exception:
                pass

        rgb_disp = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_disp, channels="RGB", use_container_width=True)

        # ─ Handle Capture click ──────────────────────────────────────────
        if st.session_state.pop("_do_capture", False):
            if not has_face:
                st.warning("⚠️ No face detected. Look directly at the camera.")
            elif not pose_good:
                step_idx = st.session_state.wizard_step % len(ANGLE_STEPS)
                st.warning(f"❌ Pose not right for '{ANGLE_STEPS[step_idx]['label']}'. "
                           f"Hint: {ANGLE_STEPS[step_idx]['hint']}")
            else:
                st.session_state.wizard_samples.append(cur_frame.copy())
                st.session_state.wizard_step += 1
                n = len(st.session_state.wizard_samples)
                st.toast(f"✅ Angle {n} captured!")
                st.rerun()

    enroll_cam_fragment()



    # Captured Samples Strip
    n_collected = len(st.session_state.wizard_samples)
    st.markdown(f"**Captured Angle Samples: `{n_collected}`** (3 to 10 recommended)")

    if n_collected > 0:
        cols = st.columns(min(n_collected, 6))
        for idx, sample in enumerate(st.session_state.wizard_samples):
            with cols[idx % 6]:
                rgb_s = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                st.image(rgb_s, use_container_width=True)
                if st.button(f"✖ Remove", key=f"rm_s_{idx}"):
                    st.session_state.wizard_samples.pop(idx)
                    st.rerun()

        col_sub1, col_sub2 = st.columns([2, 1])
        with col_sub1:
            if st.button("🚀 Finalize & Build Embeddings", use_container_width=True):
                if not p_name.strip():
                    st.error("❌ Full Name is required!")
                elif n_collected < 1:
                    st.error("❌ Capture at least 1 angle photo.")
                else:
                    with st.spinner(f"🧠 Computing 128-D embeddings for {n_collected} samples..."):
                        # Save person metadata
                        face_engine.add_person(p_name.strip(), role=p_role.strip(), info=p_info.strip())

                        # Enroll all in-memory frames
                        saved_count = 0
                        for idx, f in enumerate(st.session_state.wizard_samples):
                            fname = f"angle_{idx+1}.jpg"
                            ok = face_engine.enroll_frame(p_name.strip(), f, filename=fname)
                            if ok:
                                saved_count += 1

                    if saved_count > 0:
                        st.success(f"🎉 Successfully enrolled **{p_name}** with **{saved_count}** unique embeddings!")
                        st.session_state.wizard_samples = []
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("⚠️ Failed to detect face in captured samples. Try again with good lighting.")

        with col_sub2:
            if st.button("🗑 Clear All", use_container_width=True):
                st.session_state.wizard_samples = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN: ENROLLED PERSONS & EMBEDDING METRICS
# ══════════════════════════════════════════════════════════════════════════════
with db_col:
    st.markdown('<div class="section-title">📋 Enrolled Persons Database</div>', unsafe_allow_html=True)
    persons = face_engine.all_persons()

    if not persons:
        st.markdown("""
        <div class="ugv-card" style="text-align:center; color:#94a3b8;">
            <span style="font-size:2rem;">👤</span><br>
            <strong>No persons enrolled yet</strong><br>
            <span style="font-size:0.8rem;">Use the wizard on the left to add someone with multiple angles.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for rec in persons:
            p_name = rec["name"]
            embs = face_engine._embeddings.get(p_name, [])
            n_emb = len(embs)

            # Look for thumbnail in person's dataset folder
            person_dir = os.path.join(config.DATASET_DIR, p_name)
            preview = None
            sample_imgs = []
            if os.path.isdir(person_dir):
                sample_imgs = [
                    os.path.join(person_dir, f)
                    for f in os.listdir(person_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if sample_imgs:
                    try:
                        preview = Image.open(sample_imgs[0])
                        preview.thumbnail((70, 70))
                    except Exception:
                        preview = None

            with st.container():
                c1, c2, c3 = st.columns([1, 3, 1])
                with c1:
                    if preview:
                        st.image(preview)
                    else:
                        st.markdown('<div style="width:55px;height:55px;background:#334155;border-radius:8px;'
                                    'display:flex;align-items:center;justify-content:center;font-size:1.4rem;">👤</div>',
                                    unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="font-size:1.0rem; font-weight:700; color:#f1f5f9;">{p_name}</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">{rec.get('role', '—')}</div>
                    <div style="font-size:0.75rem; color:#4ade80; font-weight:700; margin-top:3px;">
                        ✨ {n_emb} Embeddings ({len(sample_imgs)} photos)
                    </div>
                    """, unsafe_allow_html=True)

                with c3:
                    if st.button("🗑", key=f"del_{p_name}", help=f"Remove {p_name}"):
                        face_engine._metadata = [r for r in face_engine._metadata if r["name"] != p_name]
                        face_engine._embeddings.pop(p_name, None)
                        face_engine.save_database()
                        if os.path.isdir(person_dir):
                            shutil.rmtree(person_dir)
                        st.success(f"Removed {p_name}")
                        st.rerun()

                # Expandable view to inspect angle photos
                if sample_imgs and len(sample_imgs) > 1:
                    with st.expander(f"🖼 View all {len(sample_imgs)} angle photos for {p_name}"):
                        g_cols = st.columns(min(len(sample_imgs), 5))
                        for i, img_p in enumerate(sample_imgs):
                            with g_cols[i % 5]:
                                st.image(img_p, use_container_width=True)

                # Expandable view to inspect raw 128-D embeddings in database
                person_embs = face_engine._embeddings.get(p_name, [])
                if person_embs:
                    with st.expander(f"🔬 Inspect 128-D Embeddings Vectors ({len(person_embs)} stored)"):
                        st.caption("Extracted by SFace and persisted in `dataset/database.pkl` (float32 array):")
                        for e_idx, ev in enumerate(person_embs, 1):
                            if hasattr(ev, 'flatten'):
                                flat = ev.flatten()
                                st.markdown(f"**Vector #{e_idx}** `(shape={ev.shape}, dtype={ev.dtype})`:")
                                st.code(str(flat.tolist()[:10])[:-1] + ", ... [128 floats total]]", language="python")

                st.markdown("<hr style='margin:8px 0; border-color:#334155;'>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔄 Maintenance</div>', unsafe_allow_html=True)
    if st.button("🏗 Rebuild Embeddings for All Persons", use_container_width=True):
        with st.spinner("Recomputing embeddings for all folders..."):
            total = face_engine.rebuild_from_folder()
        st.success(f"✅ Rebuilt! Total embeddings: {total}")
        st.rerun()
