"""
FaceDoor Kiosk v1
=================
Standalone door-control system — no browser needed.

States
------
  STANDBY   : No person detected. Camera checks every SCAN_INTERVAL seconds.
              Shows a beautiful idle screen (no heavy processing).
  SCANNING  : Face just appeared — running ArcFace recognition once.
  WELCOME   : Person is registered → greet, open door (serial @a,o#), hold 5 s.
  DENIED    : Person not in DB → show "Not Registered", door stays closed.

Door signals (pyserial → Arduino)
----------------------------------
  Open  : b"@a,o#"
  Close : b"@a,1#"

Usage
-----
  python3 door_kiosk.py

  Optional flags:
    --port  /dev/tty.usbmodem111201   Serial port (omit to run without hardware)
    --baud  9600
    --cam   0                         Camera index
    --full                            Fullscreen window
"""

import cv2
import json
import numpy as np
import os
import sys
import time
import math
import signal
import atexit
import threading
import argparse
import warnings
import logging

# ── Suppress TF/Keras noise ────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── Config defaults ────────────────────────────────────────────────────────────
FACES_DB          = "faces.json"
RECOGNITION_MODEL = "ArcFace"
COSINE_THRESHOLD  = 0.40       # lower = stricter
MIN_FACE_PIXELS   = 70         # ignore tiny/distant faces
SCAN_INTERVAL     = 1.5        # seconds between Haar scans in standby
DOOR_OPEN_SECS    = 5          # how long door stays open
DENIED_HOLD_SECS  = 3          # how long to show "denied" screen

WINDOW_NAME = "FaceDoor — Smart Access"

# ── Colours (BGR) ───────────────────────────────────────────────────────────────
C_BG_DARK   = (14, 10, 28)
C_ACCENT    = (240, 150, 80)    # electric indigo-orange
C_GREEN     = (100, 215, 80)
C_RED       = (80, 80, 240)
C_CYAN      = (210, 175, 40)
C_WHITE     = (245, 245, 255)
C_MUTED     = (130, 120, 160)
C_PANEL     = (30, 24, 55)

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI args
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="FaceDoor Kiosk")
    p.add_argument("--port",   default=None,  help="Arduino serial port (omit = no hardware)")
    p.add_argument("--baud",   default=9600,  type=int)
    p.add_argument("--cam",    default=0,     type=int,  help="Camera index")
    p.add_argument("--full",   action="store_true",      help="Fullscreen window")
    return p.parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
#  Serial / Door control
# ═══════════════════════════════════════════════════════════════════════════════
class DoorController:
    def __init__(self, port, baud):
        self.ser      = None
        self.port     = port
        self.baud     = baud
        self._lock    = threading.Lock()
        self._closed  = False
        if port:
            try:
                import serial
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)   # Arduino reset delay
                print(f"\u2705  Serial connected: {port} @ {baud}")
            except Exception as e:
                print(f"\u26a0\ufe0f   Serial not available ({e}). Running in no-hardware mode.")
        else:
            print("\u2139\ufe0f   No serial port specified \u2014 door signals will be printed only.")

    def open(self):
        with self._lock:
            print("\U0001f513  DOOR OPEN  \u2192 @a,o#")
            if self.ser and self.ser.is_open:
                self.ser.write(b"@a,o#")

    def close(self):
        with self._lock:
            print("\U0001f512  DOOR CLOSE \u2192 @a,1#")
            if self.ser and self.ser.is_open:
                self.ser.write(b"@a,1#")

    def cleanup(self):
        """Ensure door is closed and serial port released."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Always lock the door on exit
            try:
                if self.ser and self.ser.is_open:
                    self.ser.write(b"@a,1#")   # close door
                    time.sleep(0.1)
            except Exception:
                pass
            try:
                if self.ser:
                    self.ser.close()
                    print("\U0001f50c  Serial port closed.")
            except Exception:
                pass

    def __del__(self):
        self.cleanup()

# ═══════════════════════════════════════════════════════════════════════════════
#  ArcFace wrapper (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════
class FaceRecognizer:
    def __init__(self):
        self._ready    = False
        self._stop     = threading.Event()   # set this to ask bg thread to exit
        self._lock     = threading.Lock()
        self._DeepFace = None
        self._cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    def _load(self):
        if self._stop.is_set():
            return
        print("\U0001f504  Loading ArcFace model in background\u2026")
        from deepface import DeepFace as DF
        self._DeepFace = DF
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            DF.represent(dummy, model_name=RECOGNITION_MODEL,
                         enforce_detection=False, detector_backend="skip")
        except Exception:
            pass
        if not self._stop.is_set():
            self._ready = True
            print("\u2705  ArcFace ready.")

    def stop(self):
        """Signal background thread to stop and wait briefly."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __del__(self):
        self.stop()

    @property
    def ready(self):
        return self._ready

    # ── Haar cascade: cheap, just boxes ──────────────────────────────────────
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.12, minNeighbors=5,
            minSize=(MIN_FACE_PIXELS, MIN_FACE_PIXELS)
        )
        return list(rects) if len(rects) else []

    # ── ArcFace embedding ─────────────────────────────────────────────────────
    def embed(self, bgr_crop):
        if not self._ready:
            return None
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        try:
            res = self._DeepFace.represent(
                img_path          = rgb,
                model_name        = RECOGNITION_MODEL,
                enforce_detection = False,
                detector_backend  = "skip",
                align             = True,
            )
            if res and "embedding" in res[0]:
                return np.array(res[0]["embedding"], dtype=np.float32)
        except Exception:
            pass
        return None

    # ── Cosine distance ───────────────────────────────────────────────────────
    @staticmethod
    def cosine_dist(a, b):
        a, b = np.array(a, np.float32), np.array(b, np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 2.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    # ── Match against DB ──────────────────────────────────────────────────────
    def identify(self, bgr_crop):
        """Returns (name_or_None, confidence_0_to_100)."""
        emb = self.embed(bgr_crop)
        if emb is None:
            return None, 0

        if not os.path.exists(FACES_DB):
            return None, 0

        with open(FACES_DB) as f:
            db = json.load(f)

        best_name, best_dist = None, float("inf")
        for name, data in db.items():
            for stored in data.get("embeddings", []):
                d = self.cosine_dist(emb, stored)
                if d < best_dist:
                    best_dist, best_name = d, name

        if best_dist <= COSINE_THRESHOLD:
            conf = int((1.0 - best_dist / COSINE_THRESHOLD) * 100)
            return best_name, max(0, min(conf, 100))

        return None, 0

# ═══════════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════
def draw_corner_box(img, x1, y1, x2, y2, color, t=3, L=28):
    """Modern corner-bracket bounding box."""
    # top-left
    cv2.line(img, (x1, y1), (x1 + L, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + L), color, t, cv2.LINE_AA)
    # top-right
    cv2.line(img, (x2, y1), (x2 - L, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + L), color, t, cv2.LINE_AA)
    # bottom-left
    cv2.line(img, (x1, y2), (x1 + L, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - L), color, t, cv2.LINE_AA)
    # bottom-right
    cv2.line(img, (x2, y2), (x2 - L, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - L), color, t, cv2.LINE_AA)

def put_text_centered(img, text, cy, scale, color, thickness=1, shadow=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - tw) // 2
    if shadow:
        cv2.putText(img, text, (x+2, cy+2), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, cy), font, scale, color, thickness, cv2.LINE_AA)

def draw_progress_arc(img, cx, cy, r, pct, color, bg_color, thickness=6):
    """Draw a circular countdown arc (0-100%)."""
    angle = int(360 * pct)
    cv2.ellipse(img, (cx, cy), (r, r), -90, 0, 360, bg_color, thickness, cv2.LINE_AA)
    if angle > 0:
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, angle, color, thickness, cv2.LINE_AA)

def hex_grid_overlay(img, spacing=60, alpha=0.07):
    """Draw a subtle hexagonal grid in the background."""
    h, w = img.shape[:2]
    overlay = img.copy()
    for y in range(0, h + spacing, spacing):
        for x in range(0, w + spacing, spacing):
            pts = []
            for a in range(6):
                ang = math.radians(60 * a)
                pts.append((int(x + 20 * math.cos(ang)), int(y + 20 * math.sin(ang))))
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts_arr], True, C_MUTED, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ═══════════════════════════════════════════════════════════════════════════════
#  Screen rendering
# ═══════════════════════════════════════════════════════════════════════════════
class Renderer:
    def __init__(self, w, h):
        self.W   = w
        self.H   = h
        self._t0 = time.time()

    def _elapsed(self):
        return time.time() - self._t0

    # ── Standby screen ────────────────────────────────────────────────────────
    def standby(self, cam_frame=None):
        """
        Standby: show live camera feed (small, top-right thumbnail)
        + big idle overlay inviting the user to approach.
        """
        canvas = np.full((self.H, self.W, 3), C_BG_DARK, dtype=np.uint8)

        # Hex grid
        hex_grid_overlay(canvas, spacing=55, alpha=0.06)

        t = self._elapsed()

        # Animated vertical scan line
        scan_x = int((math.sin(t * 0.8) * 0.5 + 0.5) * self.W)
        cv2.line(canvas, (scan_x, 0), (scan_x, self.H),
                 (80, 60, 120), 1, cv2.LINE_AA)

        # Central lock icon (circle + lock body outline)
        cx, cy = self.W // 2, self.H // 2 - 60
        pulse  = 0.85 + 0.15 * math.sin(t * 2.0)
        r      = int(70 * pulse)
        glow   = int(30 * pulse)
        # Glow ring
        for dr in range(glow, 0, -4):
            alpha_val = int(40 * (dr / glow))
            col = tuple(min(255, int(c * alpha_val / 255 + C_BG_DARK[i])) for i, c in enumerate(C_ACCENT))
            cv2.circle(canvas, (cx, cy), r + dr, col, 1, cv2.LINE_AA)
        # Main circle
        cv2.circle(canvas, (cx, cy), r, C_ACCENT, 2, cv2.LINE_AA)
        # Lock shackle arc
        cv2.ellipse(canvas, (cx, cy - r // 2), (r // 2, r // 2), 0, 180, 360,
                    C_ACCENT, 2, cv2.LINE_AA)
        # Lock body
        bw, bh = r // 2 + 4, r // 2
        bx, by = cx - bw // 2, cy - r // 4 + 4
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), C_ACCENT, 2)
        # Keyhole
        cv2.circle(canvas, (cx, by + bh // 2), 5, C_ACCENT, -1)
        cv2.rectangle(canvas, (cx - 2, by + bh // 2), (cx + 2, by + bh - 4), C_ACCENT, -1)

        # Title
        put_text_centered(canvas, "FACE ACCESS CONTROL",
                          self.H // 2 + 55, 0.85, C_WHITE, 2)
        put_text_centered(canvas, "Please stand in front of the camera",
                          self.H // 2 + 90, 0.48, C_MUTED, 1, shadow=False)

        # Animated dots
        dots = "." * (int(t * 2) % 4)
        put_text_centered(canvas, f"Scanning{dots}",
                          self.H // 2 + 120, 0.44, C_CYAN, 1, shadow=False)

        # Live camera thumbnail (top-right corner)
        if cam_frame is not None:
            th, tw = self.H // 5, self.W // 4
            thumb  = cv2.resize(cam_frame, (tw, th))
            tx, ty = self.W - tw - 16, 16
            # Border
            cv2.rectangle(canvas, (tx-2, ty-2), (tx+tw+2, ty+th+2), C_ACCENT, 1)
            canvas[ty:ty+th, tx:tx+tw] = thumb
            cv2.putText(canvas, "LIVE", (tx + 6, ty + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_ACCENT, 1, cv2.LINE_AA)

        # Footer
        self._footer(canvas)
        return canvas

    # ── Scanning screen ───────────────────────────────────────────────────────
    def scanning(self, cam_frame, faces):
        canvas = cam_frame.copy()

        # Dim overlay
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (self.W, self.H), (5, 5, 20), -1)
        cv2.addWeighted(ov, 0.35, canvas, 0.65, 0, canvas)

        # Draw scanning box around face
        for (fx, fy, fw, fh) in faces:
            x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
            draw_corner_box(canvas, x1, y1, x2, y2, C_CYAN, t=2)
            put_text_centered(canvas, "Identifying…", y2 + 32, 0.60, C_CYAN, 1)

        put_text_centered(canvas, "SCANNING", 50, 0.80, C_CYAN, 2)
        self._footer(canvas)
        return canvas

    # ── Welcome screen ────────────────────────────────────────────────────────
    def welcome(self, cam_frame, name, conf, time_left, open_secs):
        canvas = cam_frame.copy()

        # Green tint overlay
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (self.W, self.H), (15, 60, 15), -1)
        cv2.addWeighted(ov, 0.30, canvas, 0.70, 0, canvas)

        # Top green banner
        bh = 110
        ov2 = canvas.copy()
        cv2.rectangle(ov2, (0, 0), (self.W, bh), (20, 100, 30), -1)
        cv2.addWeighted(ov2, 0.75, canvas, 0.25, 0, canvas)
        cv2.line(canvas, (0, bh), (self.W, bh), C_GREEN, 2)

        # ✓ icon
        cv2.circle(canvas, (60, bh // 2), 28, C_GREEN, 2, cv2.LINE_AA)
        cv2.putText(canvas, "OK", (44, bh // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GREEN, 2, cv2.LINE_AA)

        # Greeting
        cv2.putText(canvas, f"Hi, {name}!",
                    (105, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_GREEN, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"Confidence: {conf}%   |   Access Granted",
                    (105, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 1, cv2.LINE_AA)

        # Door open indicator
        put_text_centered(canvas, "🚪  DOOR OPEN", self.H // 2 - 20, 1.0, C_GREEN, 2)

        # Countdown arc (bottom-center)
        pct = time_left / open_secs
        arc_cx, arc_cy = self.W // 2, self.H - 90
        draw_progress_arc(canvas, arc_cx, arc_cy, 40, pct, C_GREEN, C_MUTED, 5)
        cv2.putText(canvas, f"{time_left:.1f}s",
                    (arc_cx - 22, arc_cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)

        self._footer(canvas)
        return canvas

    # ── Denied screen ─────────────────────────────────────────────────────────
    def denied(self, cam_frame, faces):
        canvas = cam_frame.copy()

        # Red tint
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (self.W, self.H), (30, 10, 10), -1)
        cv2.addWeighted(ov, 0.35, canvas, 0.65, 0, canvas)

        # Red banner
        bh = 110
        ov2 = canvas.copy()
        cv2.rectangle(ov2, (0, 0), (self.W, bh), (80, 20, 20), -1)
        cv2.addWeighted(ov2, 0.75, canvas, 0.25, 0, canvas)
        cv2.line(canvas, (0, bh), (self.W, bh), C_RED, 2)

        # ✗ icon
        cv2.circle(canvas, (60, bh // 2), 28, C_RED, 2, cv2.LINE_AA)
        cv2.putText(canvas, "X", (48, bh // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, C_RED, 2, cv2.LINE_AA)

        cv2.putText(canvas, "NOT REGISTERED",
                    (105, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.30, C_RED, 3, cv2.LINE_AA)
        cv2.putText(canvas, "Access Denied — Please register via the web admin panel",
                    (105, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WHITE, 1, cv2.LINE_AA)

        # Draw red corner boxes on detected faces
        for (fx, fy, fw, fh) in faces:
            draw_corner_box(canvas, fx, fy, fx + fw, fy + fh, C_RED, t=2)

        put_text_centered(canvas, "🔒  DOOR CLOSED", self.H // 2 + 40, 1.0, C_RED, 2)

        self._footer(canvas)
        return canvas

    # ── Footer strip ──────────────────────────────────────────────────────────
    def _footer(self, canvas):
        fh = 36
        h, w = canvas.shape[:2]
        ov = canvas.copy()
        cv2.rectangle(ov, (0, h - fh), (w, h), C_BG_DARK, -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, h - fh), (w, h - fh), C_PANEL, 1)
        ts = time.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(canvas, f"FaceDoor  |  ArcFace 512-D  |  {ts}",
                    (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, C_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press Q to quit  |  Press R to open web admin (port 5050)",
                    (w - 460, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, C_MUTED, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main kiosk loop
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Camera ──────────────────────────────────────────────────────────────────
    cam = cv2.VideoCapture(args.cam)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cam.isOpened():
        print("\u274c  Could not open camera.")
        sys.exit(1)

    # Window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.full:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    # Objects
    door       = DoorController(args.port, args.baud)
    recognizer = FaceRecognizer()

    # ── Teardown: called on exit no matter what ──────────────────────────────────
    _tore_down = threading.Event()

    def teardown(reason="exit"):
        if _tore_down.is_set():
            return
        _tore_down.set()
        print(f"\n\U0001f9f9  [{reason}] Tearing down FaceDoor kiosk\u2026")
        try:
            door.cleanup()          # send door-close, release serial
        except Exception as e:
            print(f"  \u26a0\ufe0f  door cleanup: {e}")
        try:
            recognizer.stop()       # stop background loader thread
        except Exception as e:
            print(f"  \u26a0\ufe0f  recognizer stop: {e}")
        try:
            cam.release()           # free camera device
            print("  \U0001f4f7  Camera released.")
        except Exception as e:
            print(f"  \u26a0\ufe0f  camera release: {e}")
        try:
            cv2.destroyAllWindows()
            print("  \U0001f5a5\ufe0f   Window destroyed.")
        except Exception:
            pass
        print("\U0001f44b  FaceDoor kiosk stopped cleanly.")

    # Register atexit so teardown runs even on unhandled exceptions
    atexit.register(teardown, "atexit")

    # SIGINT (Ctrl-C) and SIGTERM (kill / systemd stop)
    def _signal_handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        print(f"\n\U0001f6d1  Signal {sig_name} received \u2014 shutting down gracefully\u2026")
        teardown(sig_name)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Get canvas size from camera ──────────────────────────────────────────────
    _, frame0 = cam.read()
    frame0    = cv2.flip(frame0, 1) if frame0 is not None else np.zeros((720, 1280, 3), np.uint8)
    H, W      = frame0.shape[:2]
    renderer  = Renderer(W, H)

    # ── State machine ────────────────────────────────────────────────────────────
    APP_STATE    = "STANDBY"
    last_scan    = 0.0
    welcome_end  = 0.0
    denied_end   = 0.0
    welcome_name = ""
    welcome_conf = 0
    last_frame   = frame0.copy()
    last_faces   = []

    print("\n\U0001f3ac  Kiosk started. Press Q / Esc to quit.\n")

    try:
        while not _tore_down.is_set():
            ret, raw = cam.read()
            if ret:
                live = cv2.flip(raw, 1)
                last_frame = live.copy()
            else:
                live = last_frame

            now = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # Q / Esc
                print("\U0001f6d1  Quit key pressed.")
                break

            # ── STANDBY ───────────────────────────────────────────────────────
            if APP_STATE == "STANDBY":
                canvas = renderer.standby(live)
                if now - last_scan >= SCAN_INTERVAL:
                    last_scan  = now
                    last_faces = recognizer.detect_faces(live)
                    if last_faces:
                        print("\U0001f465  Face detected \u2014 starting recognition\u2026")
                        APP_STATE = "SCANNING"

            # ── SCANNING ──────────────────────────────────────────────────────
            elif APP_STATE == "SCANNING":
                canvas = renderer.scanning(live, last_faces)
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1)

                if not recognizer.ready:
                    time.sleep(0.3)
                    last_faces = recognizer.detect_faces(live)
                    if not last_faces:
                        APP_STATE = "STANDBY"
                    continue

                fx, fy, fw, fh = max(last_faces, key=lambda r: r[2] * r[3])
                pad  = 30
                cx1  = max(fx - pad, 0)
                cy1  = max(fy - pad, 0)
                cx2  = min(fx + fw + pad, W - 1)
                cy2  = min(fy + fh + pad, H - 1)
                crop = live[cy1:cy2, cx1:cx2]

                name, conf = recognizer.identify(crop)

                if name:
                    print(f"\u2705  Recognized: {name}  ({conf}%)")
                    door.open()
                    welcome_name = name
                    welcome_conf = conf
                    welcome_end  = now + DOOR_OPEN_SECS
                    APP_STATE    = "WELCOME"
                else:
                    print("\u274c  Not recognized.")
                    door.close()
                    denied_end = now + DENIED_HOLD_SECS
                    APP_STATE  = "DENIED"

            # ── WELCOME ───────────────────────────────────────────────────────
            elif APP_STATE == "WELCOME":
                time_left = welcome_end - now
                if time_left <= 0:
                    door.close()
                    APP_STATE = "STANDBY"
                    last_scan = 0
                    continue
                canvas = renderer.welcome(live, welcome_name, welcome_conf,
                                          time_left, DOOR_OPEN_SECS)

            # ── DENIED ────────────────────────────────────────────────────────
            elif APP_STATE == "DENIED":
                if now >= denied_end:
                    APP_STATE = "STANDBY"
                    last_scan = 0
                    continue
                canvas = renderer.denied(live, last_faces)

            cv2.imshow(WINDOW_NAME, canvas)

    finally:
        teardown("loop exit")


if __name__ == "__main__":
    main()
