# 🤖 UGV Beast — Face Detection Streamlit Dashboard

A fully functional, production-ready AI dashboard for the **Waveshare UGV-Beast** (Raspberry Pi 4B + ESP32).

---

## ✨ Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI metrics, architecture overview, enrolled persons table |
| 📷 **Live Detection** | Real-time webcam/camera feed with GREEN/RED face bounding boxes |
| 👤 **Enroll Face** | Multi-angle guided capture / photo upload → compute 128-D embeddings |
| 📊 **System Info** | Full UGV specs, live CPU/RAM donuts, network config |

---

## 🚀 Quick Start (Windows)

### 1. Create venv (already done)
```bash
py -3 -m venv .venv
```

### 2. Install dependencies
```bash
install.bat
```
Or manually:
```bash
.venv\Scripts\activate
pip install streamlit opencv-python numpy Pillow requests psutil plotly pandas
```

### 3. Run demo setup (creates folders + Vikas profile)
```bash
.venv\Scripts\python setup_demo.py
```

### 4. Launch dashboard
```bash
start_dashboard.bat
```
Or:
```bash
.venv\Scripts\streamlit run Dashboard.py
```
Open → **http://localhost:8501**

---

## 🔧 Connecting to Your UGV Beast

1. Power on UGV Beast — OLED shows IP (default AP mode: `192.168.50.5`)
2. Connect your PC to the `AccessPopup` Wi-Fi hotspot (password: `1234567890`)
3. In the dashboard sidebar, set **UGV IP** to the IP shown on OLED
4. Disable **Demo Mode** toggle
5. The robot is now live — camera and telemetry are real

---

## 👤 Enrolling a Face (Vikas)

1. Go to **Enroll Face** page
2. Enter Name: `Vikas`, Role: `Hackathon Team Leader`, Access: `Authorized Admin`
3. Upload 3+ clear front-facing photos
4. Click **Enroll & Build Embeddings**
5. Go to **Live Detection** — Vikas will now get a GREEN box ✅

Or place photos in `dataset/known_faces/Vikas/` and run:
```bash
.venv\Scripts\python -c "from ugv import face_engine; print(face_engine.rebuild_from_folder())"
```

---

## 📁 Directory Structure

```
UGV Beast Waveshare/
├── .venv/                    # Python virtual environment
├── .streamlit/config.toml    # Dark theme + port 8501
├── Dashboard.py              # Main dashboard (home page)
├── config.py                 # UGV IP, paths, settings
├── requirements.txt          # All dependencies
├── setup_demo.py             # One-time demo setup
├── install.bat               # Windows install script
├── start_dashboard.bat       # Windows launch script
├── pages/
│   ├── 01_📷_Live_Detection.py
│   ├── 02_👤_Enroll_Face.py
│   └── 03_📊_System_Info.py
├── ugv/
│   ├── camera.py             # OpenCV capture thread
│   ├── esp32_comm.py         # HTTP JSON → robot
│   ├── face_engine.py        # Detection + recognition
│   └── telemetry.py          # Battery, IMU, CPU stats
├── dataset/
│   └── known_faces/Vikas/    # Reference photos go here
├── static/captures/          # Auto-saved snapshots
└── database.json             # Person metadata
```

---

## 🌐 UGV Beast Network Info

| Setting | Value |
|---------|-------|
| AP Mode IP | `192.168.50.5` |
| Flask API | `:5000` |
| Streamlit | `:8501` |
| JupyterLab | `:8888` |
| Hotspot | `AccessPopup` / `1234567890` |

---

## ⚙️ Key Configuration (`config.py`)

```python
UGV_IP   = "192.168.50.5"   # Robot IP (AP mode default)
DEMO_MODE = True             # Set False when robot is connected
FACE_MATCH_TOLERANCE = 0.55  # Lower = stricter matching
```

---

## 🔌 ESP32 JSON Command Format

```json
{ "T": 1, "L": 0.5, "R": 0.5 }   // Forward at 50%
{ "T": 0 }                          // Stop
{ "T": 100, "X": 90, "SPD": 0 }   // Pan to 90°
{ "T": 101, "Y": 90, "SPD": 0 }   // Tilt to 90°
{ "T": 132, "IO": 1 }              // LED On
```

---

## 📦 Optional: Full Face Recognition

For true 128-D embedding recognition (not just Haar cascade detection):
```bash
pip install face-recognition dlib
```
> Requires **CMake** and **Visual C++ Build Tools** on Windows.  
> On Raspberry Pi: `pip install face-recognition` (works natively).

---

*Waveshare UGV Beast · Raspberry Pi 4B + ESP32 · 5MP 160° Camera*
