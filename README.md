# 🤖 Waveshare UGV Beast — AI Face Detection & Recognition Dashboard

An enterprise-grade, real-time Computer Vision and Robotic Teleoperation Dashboard designed for the **Waveshare UGV-Beast** (powered by Raspberry Pi 4B + ESP32).

This system provides real-time AI face detection and biometric recognition over wireless network streams, a multi-angle facial enrollment wizard using your workstation webcam, and integrated robot telemetry controls.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Prerequisites & Requirements](#-prerequisites--requirements)
- [Step-by-Step Installation & Setup](#-step-by-step-installation--setup)
- [Connecting to the Physical UGV Beast](#-connecting-to-the-physical-ugv-beast)
- [How to Run the Application](#-how-to-run-the-application)
- [Dashboard Walkthrough & Usage](#-dashboard-walkthrough--usage)
  - [1. Home Dashboard](#1-home-dashboard)
  - [2. Live Detection Page](#2-live-detection-page)
  - [3. Enroll Face Page](#3-enroll-face-page)
  - [4. System Info Page](#4-system-info-page)
- [AI Face Recognition Engine Details](#-ai-face-recognition-engine-details)
- [ESP32 Command Protocol](#-esp32-command-protocol)
- [Project Directory Structure](#-project-directory-structure)
- [Configuration Reference (.env)](#-configuration-reference-env)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Overview

The **Waveshare UGV Beast** is an off-road tracked/wheeled unmanned ground vehicle equipped with a 5MP 160° ultra-wide camera mounted on a 2-DOF Pan/Tilt gimbal. 

This project bridges modern deep-learning computer vision with the UGV Beast hardware:
- **Intelligent Dual-Camera Pipeline**: Surveillance streams directly from the UGV Beast's remote onboard camera over Wi-Fi, while biometric enrollment runs locally on your PC/laptop webcam for optimal image clarity.
- **Deep Metric Learning**: Employs lightweight ONNX models (**YuNet** for ultra-fast face detection and **SFace** for 128-dimensional facial embedding feature extraction) alongside standard `dlib`/`face_recognition` compatibility.
- **Color-Coded Visual Identity**: Instantly highlights **Authorized Personnel (🟢 Green Box)** and **Unknown Visitors (🔴 Red Box)** with confidence metrics and similarity scoring.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                   YOUR WORKSTATION / LAPTOP                 │
│                                                             │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │    Integrated/USB     │       │   Streamlit Web UI    │  │
│  │     Laptop Webcam     │       │   (Port 8501)         │  │
│  └──────────┬────────────┘       └───────────▲───────────┘  │
│             │ (Index 0)                      │              │
│             ▼                                │              │
│  ┌───────────────────────┐       ┌───────────┴───────────┐  │
│  │ Multi-Angle Face      │       │ Real-Time AI Engine   │  │
│  │ Enrollment Wizard     ├──────►│ YuNet (Detection)     │  │
│  │ (Center, Left, Right) │       │ SFace (128-D Embeds)  │  │
│  └───────────────────────┘       └───────────▲───────────┘  │
└──────────────────────────────────────────────┼──────────────┘
                                               │
               Wi-Fi / LAN Network (HTTP MJPEG Stream)
                                               │
┌──────────────────────────────────────────────┴──────────────┐
│                    WAVESHARE UGV BEAST                      │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Raspberry Pi 4B (Brain)                               │  │
│  │  • Flask Video Streaming Server (:5000/video_feed)    │  │
│  │  • 5MP 160° Ultra-Wide Angle Camera                  │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │ UART Serial / I2C             │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │ ESP32 Coprocessor (Actuators & Sensors)               │  │
│  │  • 4WD Motor Driver & Chassis Movement                │  │
│  │  • 2-DOF Pan/Tilt Camera Gimbal Servos                │  │
│  │  • Headlights, Battery Telemetry, IMU                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

1. **Dual-Stream Camera Architecture**:
   - **Enroll Face** automatically binds to your laptop's local webcam (`Index 0`) for crisp, controlled multi-angle enrollment.
   - **Live Detection** streams remotely from the UGV Beast (`http://<UGV_IP>:5000/video_feed`) for real-time mobile surveillance.
   - Clean thread lifecycle management prevents device conflicts when switching between pages.

2. **4-Angle Biometric Enrollment Wizard**:
   - Guided step-by-step capture: **Center Front**, **Turn Left**, **Turn Right**, and **Tilt Up**.
   - Pose landmark estimation verifies head angles before capture.
   - Multi-vector averaging builds a robust 128-dimensional embedding profile stored in `dataset/database_embeddings.json`.

3. **High-Performance Deep Learning Engine**:
   - **YuNet ONNX Detector**: Extremely fast DNN-based face detection running on CPU.
   - **SFace ONNX Embeddings**: Deep metric feature extraction providing reliable cosine similarity matching.
   - Support for `dlib` / `face_recognition` libraries with graceful automatic fallback.

4. **Interactive Dashboard**:
   - Pan/Tilt camera gimbal controls directly from the sidebar.
   - Real-time FPS, inference latency, and detection counts.
   - Snapshot capture and database management.

---

## 📋 Prerequisites & Requirements

### System Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS.
- **Python**: Version **3.10** or **3.11** recommended.
- **Hardware**:
  - A PC/laptop with an integrated webcam or external USB camera.
  - Waveshare UGV-Beast powered on and connected to the same Wi-Fi network.

### Required Software Packages
- `streamlit`
- `opencv-python`
- `numpy`
- `Pillow`
- `requests`
- `psutil`
- `plotly`
- `pandas`
- `python-dotenv`
- *(Optional)* `face-recognition` & `dlib` (Pre-compiled models are already included in `models/` so `dlib` is not mandatory).

---

## 🛠️ Step-by-Step Installation & Setup

### Step 1: Clone or Open the Project
Open PowerShell or your terminal and navigate to the project directory:
```powershell
cd "c:\Users\Dev1\Desktop\UGV Beast Waveshare"
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment ensures dependencies remain isolated:
```powershell
py -3 -m venv .venv
```

Activate the environment:
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Required Dependencies
Run the included installation script:
```cmd
install.bat
```

Or manually install packages using pip:
```powershell
pip install --upgrade pip
pip install streamlit opencv-python numpy Pillow requests psutil plotly pandas python-dotenv
```

> **Note on `dlib` & `face-recognition`:**  
> The dashboard comes bundled with pre-trained ONNX neural network models (`models/face_detection_yunet.onnx` and `models/face_recognition_sface.onnx`). If you wish to also install `dlib`, ensure you have **Visual Studio C++ Build Tools** and **CMake** installed.

### Step 4: Configure the Environment File (`.env`)
The project reads configurations from the `.env` file at the root of the workspace.

Verify or edit your `.env` file:
```ini
# Robot IP Address (Display on UGV's OLED screen)
UGV_IP=192.168.80.154
UGV_PORT=5000

# Operation Mode: 0 = Live Robot Mode, 1 = Demo / Offline Mode
UGV_DEMO=0

# Streamlit Port
DASHBOARD_PORT=8501

# Robot Video Feed
CAMERA_SOURCE=http://192.168.80.154:5000/video_feed
CAMERA_INDEX=http://192.168.80.154:5000/video_feed

# Face recognition strictness (0.0 to 1.0, lower = stricter)
FACE_MATCH_TOLERANCE=0.55
```

### Step 5: Initialize Demo Data (Optional)
To verify the database structure and create starter records:
```powershell
python setup_demo.py
```

---

## 📡 Connecting to the Physical UGV Beast

### 1. Power On the Robot
- Turn on the UGV Beast using its power switch.
- Wait approximately 30–45 seconds for Raspberry Pi OS to finish booting.
- Observe the **0.91" I2C OLED display** on the robot's rear/top chassis:
  - It will display the robot's active IP address and battery voltage.

### 2. Determine Connection Mode

#### Option A: Existing Wi-Fi Router (STA Mode) — *Recommended*
If your robot is configured to join your local Wi-Fi router:
- Note the IP displayed on the OLED (e.g., `192.168.80.154`).
- Ensure your laptop is connected to the same Wi-Fi network.

#### Option B: Robot Hotspot (AP Mode)
If no local router is configured, the robot creates its own Wi-Fi access point:
- **SSID (Wi-Fi Name)**: `AccessPopup`
- **Password**: `1234567890`
- **Default IP Address**: `192.168.50.5`
- Connect your laptop to the `AccessPopup` Wi-Fi network.

### 3. Verify Connectivity
Open PowerShell on your computer and test communication:
```powershell
ping 192.168.80.154
```
*(Replace with your robot's IP address).* You should see active ICMP replies with low latency (under 20ms).

### 4. Verify the Robot Video Stream in Browser
Open Google Chrome or any browser and visit:
```text
http://192.168.80.154:5000/video_feed
```
You should see the live, raw video stream from the UGV Beast's wide-angle camera.

### 5. Update Configuration
Ensure `UGV_IP` in `.env` matches your robot's IP:
```ini
UGV_IP=192.168.80.154
CAMERA_SOURCE=http://192.168.80.154:5000/video_feed
UGV_DEMO=0
```

---

## 🚀 How to Run the Application

### Method 1: Using the Launcher Script (Windows)
Double-click `start_dashboard.bat` or execute in terminal:
```cmd
start_dashboard.bat
```

### Method 2: Manual Terminal Execution
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Start the Streamlit application
streamlit run Dashboard.py --server.port 8501
```

Once started, open your web browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🖥️ Dashboard Walkthrough & Usage

The application features a multi-page interface accessible from the Streamlit sidebar:

```text
Pages:
  ├── 🏠 Dashboard (Home)
  ├── 01_📷_Live_Detection.py
  ├── 02_👤_Enroll_Face.py
  └── 03_📊_System_Info.py
```

---

### 1. Home Dashboard (`Dashboard.py`)
- **System Status Indicator**: Displays whether the physical robot is reachable (`LIVE`) or operating in simulated mode (`DEMO`).
- **Quick Metrics**: Total registered profiles, active camera source, detection model, and server port.
- **Enrolled Personnel Directory**: Displays registered users, roles, access permissions, and registration dates.
- **Hardware Topology**: Outlines the connection between the host PC, Raspberry Pi 4B, and ESP32 micro-controller.

---

### 2. Live Detection Page (`01_📷_Live_Detection.py`)
- **Live Video Streaming**: Streams directly from the UGV Beast's remote camera feed.
- **Real-Time Facial Recognition**:
  - **🟢 Green Bounding Box**: Displayed when a recognized, authorized face is detected. Shows the person's name and similarity confidence percentage.
  - **🔴 Red Bounding Box**: Displayed when an unknown person appears in the camera frame.
- **Pan / Tilt Gimbal Sliders**: Control the physical camera orientation in real time (Pan: 0°–180°, Tilt: 30°–150°).
- **Snapshot Trigger**: Capture and save timestamped evidence snapshots to `static/captures/`.
- **Tolerance Tuning**: Adjust the recognition strictness dynamically via the sidebar slider.

---

### 3. Enroll Face Page (`02_👤_Enroll_Face.py`)
This page is dedicated to enrolling new personnel into the biometric recognition database. **It automatically utilizes your laptop's local webcam (Index 0)** for maximum capture fidelity.

#### Step 1: Enter Profile Details
- **Full Name \***: Enter the individual's full name.
- **Role**: Position or designation (e.g., `Lead Robotics Engineer`, `Security Admin`).
- **Additional Notes**: Optional notes such as badge number or security tier.

#### Step 2: Multi-Angle Capture Wizard
The wizard guides the subject through four distinct facial perspectives to build a robust biometric profile:
1. **Center Front**: Look directly into the camera.
2. **Turn Left**: Slightly turn head to the left (~ -20° yaw).
3. **Turn Right**: Slightly turn head to the right (~ +20° yaw).
4. **Tilt Up**: Slightly tilt head upwards (~ +15° pitch).

*Alternative: You can toggle "Upload Photos" to enroll users via existing images.*

#### Step 3: Complete Enrollment
Click **"Complete Enrollment"**:
- 128-D facial feature vectors are computed for all angles.
- The average normalized vector is stored in `dataset/database_embeddings.json`.
- Metadata is saved in `database.json`.
- The user is immediately recognized on the **Live Detection** page without restarting the server.

---

### 4. System Info Page (`03_📊_System_Info.py`)
- **Live System Telemetry**: Visual gauges for host CPU utilization, RAM usage, and available disk space.
- **UGV Beast Technical Specifications**: Motor specs, encoder resolution, camera sensor parameters, and onboard power distribution.
- **Network Diagnostic Tools**: Ping latency monitor and port status checks.

---

## 🧠 AI Face Recognition Engine Details

The facial recognition pipeline operates in two distinct stages:

```text
Input Frame (640x480)
         │
         ▼
[ Stage 1: Face Detection (YuNet ONNX) ]
  • Predicts face bounding boxes [x, y, w, h]
  • Detects 5 facial landmarks (eyes, nose, mouth corners)
  • Filters out false positives based on score threshold (0.6)
         │
         ▼
[ Stage 2: Feature Extraction (SFace ONNX) ]
  • Aligns facial crop using detected landmarks
  • Computes 128-dimensional L2-normalized feature vector
         │
         ▼
[ Stage 3: Cosine Similarity Matching ]
  • Compares vector against database_embeddings.json
  • Similarity = dot(v_live, v_db)
  • If (1.0 - Similarity) <= Tolerance:
        MATCH → Authorized (🟢 Green Box)
    Else:
        NO MATCH → Unknown (🔴 Red Box)
```

### Tuning `FACE_MATCH_TOLERANCE`
- **Default Value**: `0.55`
- **Lower values (e.g., 0.40 - 0.48)**: Stricter matching; minimizes false acceptances at the expense of requiring good lighting.
- **Higher values (e.g., 0.58 - 0.65)**: More lenient matching; useful in dimly lit or dynamic environments.

---

## 🕹️ ESP32 Command Protocol

The robot executes motion and gimbal adjustments via JSON payloads sent to the robot's HTTP control endpoint (`http://<UGV_IP>:5000/api/ctrl`):

| Command Purpose | JSON Payload | Description |
|-----------------|--------------|-------------|
| **Stop Motors** | `{"T": 0}` | Emergency / standard motor stop |
| **Move Forward** | `{"T": 1, "L": 0.5, "R": 0.5}` | Drives forward at 50% duty cycle |
| **Move Backward** | `{"T": 1, "L": -0.5, "R": -0.5}` | Reverses at 50% duty cycle |
| **Turn Left** | `{"T": 1, "L": -0.4, "R": 0.4}` | Differential left spin |
| **Turn Right** | `{"T": 1, "L": 0.4, "R": -0.4}` | Differential right spin |
| **Pan Gimbal** | `{"T": 100, "X": 90, "SPD": 0}` | Sets horizontal servo angle (0° - 180°) |
| **Tilt Gimbal** | `{"T": 101, "Y": 90, "SPD": 0}` | Sets vertical servo angle (30° - 150°) |
| **LED Headlight** | `{"T": 132, "IO": 1}` | Turns front illumination LED On (`1`) or Off (`0`) |

---

## 📁 Project Directory Structure

```text
UGV Beast Waveshare/
├── .venv/                          # Python virtual environment
├── .streamlit/
│   └── config.toml                 # Streamlit UI theme and server configuration
├── dataset/
│   ├── known_faces/                # Stored enrollment portraits per user
│   ├── database_embeddings.json    # Serialized 128-D embedding vectors
│   └── database.pkl                # Binary cache of embeddings
├── models/
│   ├── face_detection_yunet.onnx   # OpenCV YuNet ONNX face detection model
│   └── face_recognition_sface.onnx # OpenCV SFace ONNX face embedding model
├── pages/
│   ├── 01_📷_Live_Detection.py      # Real-time UGV camera feed & AI recognition
│   ├── 02_👤_Enroll_Face.py         # Multi-angle enrollment wizard (laptop webcam)
│   └── 03_📊_System_Info.py         # System telemetry & hardware diagnostics
├── static/
│   └── captures/                   # Timestamped recognition snapshots
├── ugv/
│   ├── __init__.py
│   ├── camera.py                   # Thread-safe multi-source video capture
│   ├── esp32_comm.py               # HTTP JSON command client for ESP32
│   ├── face_engine.py              # Detection, pose estimation, and matching
│   └── telemetry.py                # System resource & sensor monitoring
├── .env                            # Environment variables & network configuration
├── .gitignore                      # Git exclusion rules
├── config.py                       # Central application configuration loader
├── database.json                   # Human-readable person records & roles
├── Dashboard.py                    # Main dashboard application entrypoint
├── install.bat                     # Windows automated dependency installation
├── README.md                       # Complete technical documentation
├── requirements.txt                # Python package manifest
├── setup_demo.py                   # Initial dataset & sample profile generator
└── start_dashboard.bat             # One-click Windows application launcher
```

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `UGV_IP` | `192.168.80.154` | IP address of the physical UGV Beast robot |
| `UGV_PORT` | `5000` | Port of the robot's onboard Flask streaming server |
| `UGV_DEMO` | `0` | `0` = Live robot hardware mode, `1` = Offline simulated mode |
| `DASHBOARD_PORT` | `8501` | Local web port for the Streamlit dashboard |
| `CAMERA_SOURCE` | `http://<UGV_IP>:5000/video_feed` | Video source for Live Detection page |
| `FACE_MATCH_TOLERANCE` | `0.55` | Biometric distance threshold (lower = stricter) |
| `PAN_CENTER` | `90` | Default center angle for horizontal gimbal (degrees) |
| `TILT_CENTER` | `90` | Default center angle for vertical gimbal (degrees) |
| `DEFAULT_SPEED` | `0.5` | Standard motor drive speed factor (0.1 to 1.0) |

---

## ❓ Troubleshooting & FAQs

### 1. The Live Detection page shows a black screen or connection error
- **Verify IP Address**: Check the OLED screen on the robot to verify its IP address has not changed due to DHCP lease renewal.
- **Check Stream Endpoint**: Open `http://<UGV_IP>:5000/video_feed` in your browser. If it doesn't load, verify that the robot is powered on and that the Flask video service is active.
- **Firewall Restrictions**: Ensure your Windows Defender / local firewall is not blocking incoming/outgoing traffic on port 5000.

### 2. The Enroll Face page cannot access the laptop webcam
- **Camera In Use**: Ensure no other application (e.g., Zoom, Teams, Windows Camera app) is currently using your webcam.
- **Privacy Settings**: In Windows Settings, navigate to **Privacy & Security > Camera** and ensure *Let desktop apps access your camera* is enabled.
- **Multiple Webcams**: If you have multiple cameras, adjust the camera index in `pages/02_👤_Enroll_Face.py` (`st.session_state.camera_source = 1`).

### 3. Enrolled person is recognized as "Unknown"
- **Lighting Differences**: Significant variations in lighting between enrollment and detection can alter facial contours. Use the 4-angle wizard under natural, well-lit conditions.
- **Adjust Tolerance**: Increase `FACE_MATCH_TOLERANCE` slightly (e.g., from `0.55` to `0.58`) using the sidebar slider on the Live Detection page.

### 4. How to clear the registered database and start fresh
Delete or reset the following files:
1. Delete records in `dataset/database_embeddings.json` (`{}`).
2. Delete records in `database.json` (`[]`).
3. Clear portraits from `dataset/known_faces/`.
4. Run `python setup_demo.py` to regenerate a clean default state.

---

## 📜 License & Credits

- Developed for the **Waveshare UGV Beast** Robotics Platform.
- Powered by **Streamlit**, **OpenCV**, **YuNet**, and **SFace**.
- Designed for mobile edge AI surveillance, search-and-rescue prototyping, and autonomous robotics research.
