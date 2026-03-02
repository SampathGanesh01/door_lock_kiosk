# 🔐 Door Lock Kiosk

A **standalone** face-recognition door-access system. No browser, no web server — just a Python process, a camera, and an Arduino-controlled lock.

---

## 🗂️ Project Structure

```
door_lock_kiosk/
├── door_kiosk.py      # Standalone kiosk app (only file you need to run)
├── faces.json         # Face embedding database (populated via registration)
├── start.sh           # Quick launcher script
└── requirements.txt   # Python dependencies
```

---

## ⚙️ How It Works

The kiosk runs a **4-state loop**:

| State | Behaviour |
|---|---|
| **STANDBY** | Idle screen with live camera thumbnail. Checks for a face every 1.5 s using Haar cascade (cheap). |
| **SCANNING** | Face detected → runs one ArcFace 512-D recognition pass. |
| **WELCOME** | Person recognised → greet by name, send `@a,o#` (open) to Arduino, hold 5 s, then close. |
| **DENIED** | Unknown face → display "Not Registered", send `@a,1#` (close), hold 3 s. |

---

## 🚀 Setup & Run

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3)?**
> Edit `requirements.txt` — comment out `tensorflow` and uncomment `tensorflow-macos`, then re-run.

### 3. Launch

**Option A — Quick launcher (auto-detects Arduino)**
```bash
bash start.sh
```

**Option B — Direct Python**
```bash
python3 door_kiosk.py
```

---

## 🎛️ Command-Line Flags

| Flag | Default | Description |
|---|---|---|
| `--port /dev/tty.usbmodem*` | _(none)_ | Arduino serial port for door control |
| `--baud 9600` | `9600` | Serial baud rate |
| `--cam 0` | `0` | Camera device index |
| `--full` | off | Launch in fullscreen mode |

**Example — with Arduino on a specific port, fullscreen:**
```bash
python3 door_kiosk.py --port /dev/tty.usbmodem1101 --baud 9600 --full
```

---

## 🤖 Arduino Serial Protocol

| Command | Meaning |
|---|---|
| `@a,o#` | Unlock / open the door |
| `@a,1#` | Lock / close the door |

The door is **always locked on exit** — even on crash or Ctrl-C.

---

## 🗃️ Face Database (`faces.json`)

Faces are stored as **ArcFace 512-D embedding vectors** — no raw images on disk.  
Each person can have up to 10 stored samples for better accuracy.

To **add faces**, you need to populate `faces.json` using a separate registration tool (e.g., `face_recog_door/app.py` admin panel), then copy the updated `faces.json` into this folder.

---

## ⌨️ Keyboard Shortcuts (kiosk window)

| Key | Action |
|---|---|
| `Q` or `Esc` | Quit the kiosk cleanly |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` | Try `--cam 1` or `--cam 2` |
| ArcFace slow on first run | Downloads ~500 MB model, cached in `~/.deepface/` afterwards |
| Serial port not found | Run `ls /dev/tty.usbmodem*` to find the port |
| TensorFlow install fails on Mac | Use `tensorflow-macos` in `requirements.txt` |
