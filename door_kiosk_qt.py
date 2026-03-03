"""
Door Lock Kiosk — Qt Edition v2
=================================
Full-screen PySide6 kiosk. No window chrome, no taskbar, no cursor.
ArcFace 512-D face recognition + Arduino serial door control.

States
------
  STANDBY  → idle camera feed, scan for faces every 1.5 s
  SCANNING → face found, ArcFace running off-thread
  WELCOME  → recognised  → door opens for DOOR_OPEN_SECS, then closes
  DENIED   → unknown     → door locked for DENIED_HOLD_SECS, then standby

Usage
-----
  python3 door_kiosk_qt.py
  python3 door_kiosk_qt.py --port /dev/ttyACM0 --cam 0
"""

from __future__ import annotations
import cv2, json, numpy as np
import os, sys, time, math, signal, atexit, threading, argparse, warnings, logging, queue
import urllib.request, urllib.error
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QSizePolicy,
)
from PySide6.QtCore  import Qt, QTimer, QThread, Signal, QRectF, QPointF
from PySide6.QtGui   import (
    QImage, QPixmap, QFont, QPainter, QColor, QPen,
    QBrush, QLinearGradient,
)
try:
    from PySide6.QtSvg import QSvgWidget, QSvgRenderer
    _HAS_SVG = True
except ImportError:
    _HAS_SVG = False

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asbl_logo.svg")

# ── Config ────────────────────────────────────────────────────────────────────
FACES_DB          = "faces.json"
RECOGNITION_MODEL = "ArcFace"
COSINE_THRESHOLD  = 0.50   # relaxed slightly to account for Haar/RetinaFace alignment margin
MIN_FACE_PIXELS   = 120    # must be large enough for stable recognition without blur
SCAN_INTERVAL     = 1.5
DOOR_OPEN_SECS    = 5
DENIED_HOLD_SECS  = 3
TICK_MS           = 33          # ~30 fps

# ── Area of Interest (AOI) ─────────────────────────────────────────────────────
# Fractions of the full frame that define the centre recognition zone.
# A face whose CENTRE POINT falls inside this box is eligible for ArcFace.
# Adjust these to match your real kiosk camera angle / mounting height.
AOI_X_FRAC   = 0.25   # left edge of AOI  (25 % from left)
AOI_Y_FRAC   = 0.10   # top  edge of AOI  (10 % from top)
AOI_W_FRAC   = 0.50   # width of AOI zone (centre 50 % of frame)
AOI_H_FRAC   = 0.80   # height of AOI zone (centre 80 % of frame)
AOI_MIN_FACE = 90     # minimum face width in pixels  (person too far → skip)

# ── Motion / Presence detection ────────────────────────────────────────────────
MOTION_THRESHOLD  = 25     # MOG2 pixel-diff sensitivity  (lower = more sensitive)
MOTION_MIN_AREA   = 3000   # min contour area (px²) at full resolution
PRESENCE_COOLDOWN = 3.0    # seconds of no motion → hide guide, stay STANDBY

# ── Debug overlay ──────────────────────────────────────────────────────────────
DEBUG_AOI = False   # set True to always draw AOI box for physical alignment tuning

# ── Single-shot Recognition (RPi Optimized) ───────────────────────────────────
# We use a 1.5s window. It succeeds immediately on the first ArcFace match.
# Stable bounding box required for 200ms before embedding to prevent blur.
SCAN_ATTEMPTS     = 3      # maximum ArcFace attempts per scanning session
SCANNING_TIMEOUT  = 1.5    # seconds max in SCANNING window before giving up (→ DENIED)
SCAN_STABILIZE_MS = 0.2    # seconds of stable bounding box required before embedding

# ── Qt colours ────────────────────────────────────────────────────────────────
_BG       = "#0a0a1a"
_HEADER   = "#0c0c24"
_BORDER   = "#1e1e40"
_ACCENT   = "#6366f1"
_TEXT     = "#e2e2f0"
_MUTED    = "#52526e"
_GREEN_BG = "#020e08"
_GREEN    = "#22c55e"
_RED_BG   = "#100404"
_RED      = "#ef4444"
_SCAN_BG  = "#020a12"
_SCAN     = "#38bdf8"

# ── OpenCV BGR for face overlay ───────────────────────────────────────────────
_CV_CYAN  = (190, 155, 20)
_CV_GREEN = (50, 190, 60)
_CV_RED   = (50, 50, 210)

# ── Responsive scaling ─────────────────────────────────────────────────────────
# Set in main() once QApplication + screen info are available.
# Used by _s() to scale every hardcoded pixel value uniformly.
_SCALE: float = 1.0

def _s(n: int | float) -> int:
    """Scale a pixel value by the current screen scale factor."""
    return max(1, int((n * 1.4) * _SCALE))

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI args
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Door Lock Kiosk — Qt Edition")
    p.add_argument("--port", default=None, help="Arduino serial port")
    p.add_argument("--baud", default=9600,  type=int)
    p.add_argument("--cam",  default=0,     type=int, help="Camera index")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Door Controller
# ═══════════════════════════════════════════════════════════════════════════════
class DoorController:
    def __init__(self, port, baud):
        self.ser     = None
        self._lock   = threading.Lock()
        self._closed = False
        if port:
            try:
                import serial
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                print(f"✅  Serial connected: {port} @ {baud}")
            except Exception as e:
                print(f"⚠️   Serial not available ({e}). No-hardware mode.")
        else:
            print("ℹ️   No serial port — door signals printed only.")

    def open(self):
        with self._lock:
            print("🔓  DOOR OPEN  → @a,o#")
            if self.ser and self.ser.is_open:
                self.ser.write(b"@a,o#")

    def close(self):
        with self._lock:
            print("🔒  DOOR CLOSE → @a,1#")
            if self.ser and self.ser.is_open:
                self.ser.write(b"@a,1#")

    def cleanup(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.ser and self.ser.is_open:
                    self.ser.write(b"@a,1#")
                    time.sleep(0.1)
                    self.ser.close()
                    print("🔌  Serial port closed.")
            except Exception:
                pass

    def __del__(self):
        self.cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
#  Face Recognizer
# ═══════════════════════════════════════════════════════════════════════════════
class FaceRecognizer:
    def __init__(self):
        self._ready    = False
        self._stop     = threading.Event()
        self._DeepFace = None
        self._cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # ── In-Memory Database ──
        self._db = {}
        self._emb_matrix = np.array([])
        self._names = []
        self._db_lock = threading.Lock()
        self.reload_db()
        
        threading.Thread(target=self._load, daemon=True).start()

    def reload_db(self):
        """Load faces.json into memory and pre-flatten embeddings for instant vectorized matching."""
        if not os.path.exists(FACES_DB):
            return
            
        with open(FACES_DB, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[DB] Error parsing {FACES_DB}: {e}")
                return
                
        embeddings = []
        names = []
        for name, user_data in data.items():
            for emb in user_data.get("embeddings", []):
                if len(emb) > 0:
                    embeddings.append(emb)
                    names.append(name)
                    
        with self._db_lock:
            self._db = data
            if embeddings:
                self._emb_matrix = np.array(embeddings, dtype=np.float32)
                self._names = names
            else:
                self._emb_matrix = np.array([])
                self._names = []
        print(f"[DB] Loaded {len(names)} total embeddings into memory.")

    def _load(self):
        if self._stop.is_set():
            return
        print("Loading ArcFace + RetinaFace models in background…")
        from deepface import DeepFace as DF
        self._DeepFace = DF
        # ── Warm up ArcFace (recognition model) ────────────────────────────
        try:
            DF.represent(np.zeros((112, 112, 3), np.uint8),
                         model_name=RECOGNITION_MODEL,
                         enforce_detection=False, detector_backend="skip")
        except Exception:
            pass
        # ── Warm up RetinaFace (detection model) ───────────────────────────
        # This prevents a 1-2s cold-start on the first real detect_faces() call.
        try:
            DF.extract_faces(np.zeros((320, 240, 3), np.uint8),
                             detector_backend="retinaface",
                             enforce_detection=False, align=False)
            print("  RetinaFace ready.")
        except Exception as e:
            print(f"  RetinaFace warmup skipped: {e}")
        if not self._stop.is_set():
            self._ready = True
            print("ArcFace ready.")


    def stop(self):
        self._stop.set()

    @property
    def ready(self):
        return self._ready

    def detect_faces(self, frame):
        """Detect faces in a BGR frame using solely a Haar Cascade. Returns (x,y,w,h).
        
        By skipping RetinaFace during detection, we save massive amounts of CPU on the 
        Raspberry Pi. RetinaFace is now strictly reserved for the final Alignment pass
        inside `embed()`.
        """
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(MIN_FACE_PIXELS, MIN_FACE_PIXELS)
        )
        return list(rects) if len(rects) else []


    def embed(self, bgr_crop):
        """Generate a 512-D ArcFace embedding from a BGR face crop.

        Added dynamic alignment + blur check:
        1. Rejects blurry frames instantly using Laplacian variance.
        2. Passes the expanded crop into RetinaFace, allowing DeepFace to
           find the eyes dynamically and enforce server-matching geometry
           via align=True.
        """
        if not self._ready:
            return None

        # ── 1. Blur Check ──────────────────────────────────────────────────────────
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 40.0:
            print(f"  [Embed] Frame blurry ({blur_score:.1f} < 40) — skipping pass")
            return None

        # ── 2. Dynamic Alignment + Embedding ───────────────────────────────────────
        try:
            res = self._DeepFace.represent(
                img_path=cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB),
                model_name=RECOGNITION_MODEL,
                enforce_detection=False,
                detector_backend="retinaface", # Let RetinaFace align the eyes
                align=True,   # CRITICAL: matches server
            )
            if res and "embedding" in res[0]:
                return np.array(res[0]["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"  [Embed error] {e}")
        return None

    @staticmethod
    def cosine_dist(a, b):
        a, b = np.array(a, np.float32), np.array(b, np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 2.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    def identify(self, bgr_crop):
        emb = self.embed(bgr_crop)
        if emb is None:
            return None, 0
            
        with self._db_lock:
            if len(self._emb_matrix) == 0:
                return None, 0
                
            # Vectorized Cosine Similarity
            norms = np.linalg.norm(self._emb_matrix, axis=1) * np.linalg.norm(emb)
            norms[norms == 0] = 1e-9 # Prevent div zero
            similarities = np.dot(self._emb_matrix, emb) / norms
            distances = 1.0 - similarities
            
            idx = np.argmin(distances)
            best_dist = distances[idx]
            best_name = self._names[idx]
            
        if best_dist <= COSINE_THRESHOLD:
            # Convert to 0-100% confidence
            conf = max(0, min(int((1 - float(best_dist) / COSINE_THRESHOLD) * 100), 100))
            return best_name, conf
            
        return None, 0



# ═══════════════════════════════════════════════════════════════════════════════
#  Presence / Motion Detector
# ═══════════════════════════════════════════════════════════════════════════════
class PresenceDetector:
    """Layer-1 cheap trigger: detect ANY human / motion in the entire frame.

    Uses MOG2 background subtraction (built-in OpenCV, no extra model).
    When motion is detected the state machine advances to Layer-2 (AOI check).
    This avoids running Haar cascade + ArcFace on every idle frame.
    """

    def __init__(self):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=MOTION_THRESHOLD,
            detectShadows=False,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> bool:
        """Return True if significant movement / presence found anywhere in frame."""
        # Downscale for speed; motion area is scaled back afterwards
        small  = cv2.resize(frame, (320, 240))
        mask   = self._bg.apply(small)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Scale-up factor so area threshold is resolution-independent
        scale = (frame.shape[1] / 320.0) * (frame.shape[0] / 240.0)
        return any(cv2.contourArea(c) * scale >= MOTION_MIN_AREA for c in cnts)


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenCV overlay helpers
# ═══════════════════════════════════════════════════════════════════════════════
def draw_corner_box(img, x1, y1, x2, y2, color, t=3, L=30):
    for pts in [((x1,y1),(x1+L,y1)), ((x1,y1),(x1,y1+L)),
                ((x2,y1),(x2-L,y1)), ((x2,y1),(x2,y1+L)),
                ((x1,y2),(x1+L,y2)), ((x1,y2),(x1,y2-L)),
                ((x2,y2),(x2-L,y2)), ((x2,y2),(x2,y2-L))]:
        cv2.line(img, pts[0], pts[1], color, t, cv2.LINE_AA)


# ── AOI helpers ───────────────────────────────────────────────────────────────

def _aoi_rect(frame_w: int, frame_h: int) -> tuple:
    """Return (x1, y1, x2, y2) of the AOI rectangle in pixel coords."""
    x1 = int(frame_w * AOI_X_FRAC)
    y1 = int(frame_h * AOI_Y_FRAC)
    x2 = int(frame_w * (AOI_X_FRAC + AOI_W_FRAC))
    y2 = int(frame_h * (AOI_Y_FRAC + AOI_H_FRAC))
    return x1, y1, x2, y2


def is_face_in_aoi(fx: int, fy: int, fw: int, fh: int,
                   frame_w: int, frame_h: int) -> bool:
    """Layer-2 gate: True only when the face centre falls inside the AOI zone
    AND the face is large enough (person close enough to camera).

    Two conditions must both be met:
      1. Face centre (cx, cy) is within the central AOI rectangle
      2. Face width >= AOI_MIN_FACE pixels (not too far away)
    """
    cx = fx + fw // 2
    cy = fy + fh // 2
    x1, y1, x2, y2 = _aoi_rect(frame_w, frame_h)
    in_zone    = x1 <= cx <= x2 and y1 <= cy <= y2
    close_enough = fw >= AOI_MIN_FACE
    return in_zone and close_enough


def draw_aoi_overlay(frame: np.ndarray, state: str = "wait"):
    """Draw the AOI guide box + instruction text on the live camera frame.

    state: 'wait'  → amber box, "Step into the frame"
           'ready' → green box, "Hold still — scanning…"
           'debug' → blue box, used when DEBUG_AOI=True in STANDBY
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = _aoi_rect(W, H)

    # Colour per state
    if state == "ready":
        color = (50, 220, 80)    # green: face in AOI, scanning
        label = "Hold still \u2014 scanning\u2026"
    elif state == "debug":
        color = (200, 120, 20)   # blue tint: debug info only
        label = f"AOI DEBUG  [{x1},{y1}]\u2192[{x2},{y2}]"
    else:
        color = (30, 140, 220)   # amber: guide the user
        label = "Step into the frame"

    # Semi-transparent fill inside AOI
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)  # very subtle tint

    # Corner bracket border
    draw_corner_box(frame, x1, y1, x2, y2, color, t=3, L=40)

    # Centred text below the AOI box
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    tx = max(0, (W - tw) // 2)
    ty = min(H - 10, y2 + th + 14)
    # Drop shadow
    cv2.putText(frame, label, (tx + 1, ty + 1), font, fs, (0, 0, 0), ft + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty),         font, fs, color,      ft,     cv2.LINE_AA)




# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _font(size: int, bold: bool = False) -> QFont:
    # Scale font size to current screen; uses system default sans-serif
    f = QFont()
    f.setPointSize(_s(size))
    f.setBold(bold)
    return f

def _lbl(text: str, size: int, bold: bool, color: str, align=Qt.AlignCenter) -> QLabel:
    w = QLabel(text)
    w.setFont(_font(size, bold))
    w.setStyleSheet(f"color:{color}; background:transparent;")
    w.setAlignment(align)
    w.setWordWrap(True)
    # Expanding so the label fills the full panel width, allowing
    # setAlignment(AlignCenter) to visually center the text.
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return w

def _load_logo_white(w: int = 240, h: int = 74) -> QPixmap | None:
    """
    Renders the ASBL SVG and inverts its colours to white so it's visible
    on any dark background.  Qt stylesheets don't support CSS filters, so
    we do this manually via numpy pixel manipulation.
    """
    if not (_HAS_SVG and os.path.exists(LOGO_PATH)):
        return None
    renderer = QSvgRenderer(LOGO_PATH)
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))          # transparent canvas
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    renderer.render(p)
    p.end()
    # Pixel buffer: Format_ARGB32 is stored as B G R A on little-endian
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()
    mask = arr[:, :, 3] > 10              # non-transparent pixels only
    arr[mask, 0] = 255 - arr[mask, 0]    # invert B
    arr[mask, 1] = 255 - arr[mask, 1]    # invert G
    arr[mask, 2] = 255 - arr[mask, 2]    # invert R
    result = QImage(arr.tobytes(), w, h, w * 4, QImage.Format_ARGB32)
    return QPixmap.fromImage(result)


def _asbl_strip(bg: str = _HEADER) -> QWidget:
    """Small 'Powered by ASBL' branding bar — used at the bottom of every panel."""
    bar = QWidget()
    bar.setFixedHeight(_s(36))
    bar.setStyleSheet(f"background:{bg}; border-top:1px solid rgba(255,255,255,0.06);")
    row = QHBoxLayout(bar)
    row.setContentsMargins(_s(28), 0, _s(28), 0)
    row.addStretch()
    pix = _load_logo_white(_s(70), _s(22))
    if pix:
        logo = QLabel(); logo.setPixmap(pix)
        logo.setStyleSheet("background:transparent;")
        row.addWidget(logo)
        row.addSpacing(_s(8))
    txt = QLabel("Powered by ASBL")
    txt.setFont(_font(9, True))
    txt.setStyleSheet(f"color:{_MUTED}; background:transparent;")
    row.addWidget(txt)
    row.addStretch()
    return bar


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom painted widgets
# ═══════════════════════════════════════════════════════════════════════════════
class PulsingRingWidget(QWidget):
    """Pulsing lock-ring animation for STANDBY."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = time.time()
        self.setMinimumSize(_s(200), _s(200))
        # Single timer: connected AND started on the same instance
        t = QTimer(self); t.timeout.connect(self.update); t.start(33)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        t  = time.time() - self._t0
        cx, cy = self.width() // 2, self.height() // 2
        br = min(cx, cy) - 22

        # Outer glow rings
        for i in range(4):
            alpha = max(0.0, math.sin(t * 1.3 + i * 1.1))
            c = QColor(_ACCENT); c.setAlpha(int(alpha * 30))
            p.setPen(QPen(c, 1.5)); p.setBrush(Qt.NoBrush)
            r = br + 18 + i * 14
            p.drawEllipse(QPointF(cx, cy), r, r)

        # Main ring
        pulse = 0.92 + 0.08 * math.sin(t * 2.4)
        r = int(br * pulse)
        p.setPen(QPen(QColor(_ACCENT), 2.5)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Lock body
        col = QColor(_ACCENT)
        p.setPen(QPen(col, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        bw, bh = int(r * 0.65), int(r * 0.50)
        bx, by = cx - bw // 2, cy - bh // 4
        p.drawRoundedRect(bx, by, bw, bh, 5, 5)
        # Lock shackle
        aw = int(bw * 0.55); ah = int(r * 0.48)
        p.drawArc(cx - aw//2, by - ah + 4, aw, ah, 0, 180 * 16)
        # Keyhole
        p.setBrush(QBrush(col)); p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, by + bh * 0.42), 4, 4)
        p.drawRect(cx - 2, int(by + bh * 0.46), 4, int(bh * 0.34))
        p.end()


class SpinningArcWidget(QWidget):
    """Dual spinning arcs for SCANNING."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = time.time()
        self.setMinimumSize(_s(180), _s(180))
        t = QTimer(self); t.timeout.connect(self.update); t.start(33)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        t  = time.time() - self._t0
        cx, cy = self.width() // 2, self.height() // 2
        r = min(cx, cy) - 14

        # Background track
        p.setPen(QPen(QColor(_BORDER), 7, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Outer spinning arc
        a1 = int(t * 220) % 360
        p.setPen(QPen(QColor(_SCAN), 7, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(cx-r, cy-r, r*2, r*2), (90 - a1) * 16, -110 * 16)

        # Inner counter-spinning arc
        r2 = r - 18
        a2 = int(t * 150) % 360
        p.setPen(QPen(QColor(_ACCENT), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(cx-r2, cy-r2, r2*2, r2*2), (90 + a2) * 16, -70 * 16)

        # Pulsing center dot
        dot = int(7 + 3 * math.sin(t * 5))
        p.setBrush(QBrush(QColor(_SCAN))); p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), dot, dot)
        p.end()


class CountdownBar(QWidget):
    """Horizontal depleting progress bar."""
    def __init__(self, hex_color: str, parent=None):
        super().__init__(parent)
        self._color = QColor(hex_color)
        self._pct   = 1.0
        self.setFixedHeight(_s(10))

    def set_pct(self, pct: float):
        self._pct = max(0.0, min(1.0, pct))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h, r = self.width(), self.height(), 5
        p.setBrush(QBrush(QColor(_BORDER))); p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, w, h, r, r)
        fw = int(w * self._pct)
        if fw > 4:
            g = QLinearGradient(0, 0, fw, 0)
            g.setColorAt(0.0, self._color)
            g.setColorAt(1.0, self._color.lighter(140))
            p.setBrush(QBrush(g))
            p.drawRoundedRect(0, 0, fw, h, r, r)
        p.end()


# ═══════════════════════════════════════════════════════════════════════════════
#  State Panels
# ═══════════════════════════════════════════════════════════════════════════════
class StartupPanel(QWidget):
    """Initial loading screen shown while deepface/retinaface loads in background."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{_BG};")
        lay = QVBoxLayout(self)
        
        lay.addStretch(3)
        
        # Render the large SVG ASBL Logo in the centre
        logo = QLabel()
        pix = _load_logo_white(_s(240), _s(70))
        if pix:
            logo.setPixmap(pix)
            logo.setAlignment(Qt.AlignCenter)
        else:
            logo.setText("A S B L")
            logo.setFont(_font(48, True))
            logo.setAlignment(Qt.AlignCenter)
            logo.setStyleSheet("color: white;")
        lay.addWidget(logo)

        lay.addSpacing(_s(30))

        # Welcome Subtext
        welcome = QLabel("Welcome to ASBL Gym")
        welcome.setFont(_font(22, True))
        welcome.setStyleSheet("color: white; background:transparent;")
        welcome.setAlignment(Qt.AlignCenter)
        lay.addWidget(welcome)
        
        lay.addStretch(2)
        
        # Loading Indicator Text (Moved lower, made larger)
        self._status = QLabel("Loading AI Models...")
        self._status.setFont(_font(18, True))
        self._status.setStyleSheet(f"color: {_MUTED}; background:transparent;")
        self._status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._status)

        lay.addSpacing(_s(40))


class StandbyPanel(QWidget):
    """Branded idle screen — full screen, camera hidden, live clock."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{_BG};")

        outer = QVBoxLayout(self)
        outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)

        # ── Live clock ────────────────────────────────────────────────────────
        self._time_lbl = _lbl("", 60, True, _TEXT)
        outer.addWidget(self._time_lbl)
        self._date_lbl = _lbl("", 14, False, _MUTED)
        outer.addWidget(self._date_lbl)

        outer.addSpacing(_s(28))

        # Thin divider
        div = QWidget(); div.setFixedSize(_s(160), 1)
        div.setStyleSheet(f"background:{_BORDER};")
        outer.addWidget(div, 0, Qt.AlignCenter)

        outer.addSpacing(_s(24))

        # ── ASBL Logo (white) ───────────────────────────────────────────────
        pix = _load_logo_white(_s(280), _s(86))
        if pix:
            logo_lbl = QLabel()
            logo_lbl.setPixmap(pix)
            logo_lbl.setAlignment(Qt.AlignCenter)
            logo_lbl.setStyleSheet("background:transparent;")
            outer.addWidget(logo_lbl)
        else:
            outer.addWidget(_lbl("ASBL", 40, True, _TEXT))

        outer.addSpacing(_s(20))

        # ── Title + invitation ───────────────────────────────────────────────
        outer.addWidget(_lbl("Welcome to ASBL GYM", 26, True, _TEXT))
        outer.addSpacing(_s(6))
        outer.addWidget(_lbl("Please stand in front of the camera", 13, False, _MUTED))
        outer.addSpacing(_s(12))
        self._dots = _lbl("●  ○  ○", 11, False, _ACCENT)
        outer.addWidget(self._dots)

        outer.addStretch(1)

        # Start timers
        t = QTimer(self); t.timeout.connect(self._tick_dots); t.start(600)
        self._di = 0
        tc = QTimer(self); tc.timeout.connect(self._tick_clock); tc.start(1000)
        self._tick_clock()

    def _tick_clock(self):
        self._time_lbl.setText(time.strftime("%H:%M"))
        self._date_lbl.setText(time.strftime("%A, %d %B %Y"))

    def _tick_dots(self):
        patterns = ["●  ○  ○", "●  ●  ○", "●  ●  ●", "○  ●  ●", "○  ○  ●", "○  ○  ○"]
        self._di = (self._di + 1) % len(patterns)
        self._dots.setText(patterns[self._di])


class ScanningPanel(QWidget):
    def __init__(self, cam_view: QWidget, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{_SCAN_BG};")
        
        self.hlay = QHBoxLayout(self)
        self.hlay.setSpacing(0); self.hlay.setContentsMargins(0,0,0,0)
        
        self.hlay.addWidget(cam_view, 65)
        
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background:{_BORDER};")
        self.hlay.addWidget(sep)
        
        right_container = QWidget()
        lay = QVBoxLayout(right_container)
        lay.setSpacing(10); lay.setContentsMargins(24, 28, 24, 20)

        lay.addStretch(1)
        self._arc = SpinningArcWidget()
        self._arc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._arc, 4, Qt.AlignCenter)
        lay.addStretch(1)

        lay.addWidget(_lbl("IDENTIFYING", 18, True, _SCAN))
        lay.addWidget(_lbl("Please hold still…", 11, False, _MUTED))
        lay.addSpacing(16)
        
        self.hlay.addWidget(right_container, 35)


class WelcomePanel(QWidget):
    """Full-screen access-granted panel — just a clean welcome greeting."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{_GREEN_BG};")

        outer = QVBoxLayout(self)
        outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(2)

        centre = QWidget()
        centre.setStyleSheet("background:transparent;")
        # Must expand horizontally so child labels fill full width
        centre.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(centre)
        lay.setSpacing(_s(10)); lay.setContentsMargins(_s(60), 0, _s(60), 0)
        lay.setAlignment(Qt.AlignCenter)

        lay.addWidget(_lbl("✓", 110, True, _GREEN))

        badge = _lbl("ACCESS GRANTED", 13, True, _GREEN_BG)
        badge.setStyleSheet(
            f"color:{_GREEN_BG}; background:{_GREEN};"
            f"border-radius:6px; padding: 4px 20px;"
        )
        badge.setFixedHeight(_s(34))
        lay.addWidget(badge, 0, Qt.AlignCenter)
        lay.addSpacing(_s(24))

        lay.addWidget(_lbl("Welcome,", 18, False, _MUTED))
        self._name = _lbl("", 68, True, _GREEN)
        lay.addWidget(self._name)

        outer.addWidget(centre)
        outer.addStretch(3)

    def set_info(self, name: str, conf: int):
        self._name.setText(name.upper())

    def set_time_left(self, time_left: float, total: float):
        pass   # door closes silently in background


class DeniedPanel(QWidget):
    """Full-screen access-denied panel — camera is hidden while this shows."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{_RED_BG};")

        outer = QVBoxLayout(self)
        outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(2)

        centre = QWidget()
        centre.setStyleSheet("background:transparent;")
        # Must expand horizontally so child labels fill full width
        centre.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(centre)
        lay.setSpacing(_s(8)); lay.setContentsMargins(_s(60), 0, _s(60), 0)
        lay.setAlignment(Qt.AlignCenter)

        lay.addWidget(_lbl("✕", 110, True, _RED))

        badge = _lbl("ACCESS DENIED", 13, True, _RED_BG)
        badge.setStyleSheet(
            f"color:{_RED_BG}; background:{_RED};"
            f"border-radius:6px; padding: 4px 20px;"
        )
        badge.setFixedHeight(_s(34))
        lay.addWidget(badge, 0, Qt.AlignCenter)
        lay.addSpacing(_s(16))

        lay.addWidget(_lbl("Face Not Registered", 22, True, _RED))
        lay.addWidget(_lbl("Please contact an administrator to register.", 13, False, _MUTED))

        outer.addWidget(centre)
        outer.addStretch(2)

        bottom = QWidget()
        bottom.setStyleSheet(f"background:#180404; border-top:1px solid #2a0808;")
        bot_lay = QVBoxLayout(bottom)
        bot_lay.setContentsMargins(48, 14, 48, 18); bot_lay.setSpacing(6)

        bot_lay.addWidget(_lbl("Returning to standby in", 10, False, _MUTED))
        self._bar = CountdownBar(_RED)
        bot_lay.addWidget(self._bar)
        self._secs = _lbl("", 15, True, _RED)
        bot_lay.addWidget(self._secs)

        outer.addWidget(bottom)

    def set_time_left(self, time_left: float, total: float):
        self._bar.set_pct(max(0.0, time_left / total))
        self._secs.setText(f"{max(0.0, time_left):.1f} s")


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera Thread
# ═══════════════════════════════════════════════════════════════════════════════
class CameraThread(QThread):
    frame_ready = Signal(object)   # emits np.ndarray

    def __init__(self, cam_index: int = 0):
        super().__init__()
        self._cam_index = cam_index
        self._running   = True

    def run(self):
        cap = cv2.VideoCapture(self._cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self._running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(cv2.flip(frame, 1))
            else:
                time.sleep(0.01)
        cap.release()

    def stop(self):
        self._running = False
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera View
# ═══════════════════════════════════════════════════════════════════════════════
class CameraView(QLabel):
    """Scales and displays the live camera frame."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:{_BG};")

    def set_frame(self, bgr: np.ndarray):
        h, w, ch = bgr.shape
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        self.setPixmap(pix)


# ═══════════════════════════════════════════════════════════════════════════════
#  Background API Sync
# ═══════════════════════════════════════════════════════════════════════════════
class ApiSyncWorker(QThread):
    """Background thread that syncs face data from the cloud server.

    Key fix (v2): This worker now hits /api/faces.json which returns the FULL
    512-D ArcFace embeddings generated on the server (with align=True).
    Previously it hit /api/faces (summary only) and re-computed embeddings
    from the 80×80 thumbnail using align=False — causing two problems:
      1. Low-quality embeddings from a tiny blurry thumbnail
      2. Alignment mismatch → high cosine distance → everyone denied

    Now: server embeddings are copied directly to local faces.json with no
    re-computation needed. Multiple embeddings per person are preserved.
    """
    sync_completed = Signal(str)

    # /api/faces.json returns full embeddings; /api/faces is summary-only
    FACES_JSON_URL = "https://testasbl.sarvamsync.com/api/faces.json"

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        while self._running:
            try:
                print("🔄 Syncing from API (full embeddings)…")
                req = urllib.request.Request(
                    self.FACES_JSON_URL,
                    headers={'accept': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    raw = response.read()

                # /api/faces.json returns:
                # { name: { user_id, embeddings: [[...512-D...], ...], face_b64, samples } }
                server_db: dict = json.loads(raw)

                # Load current local faces.json
                local_data: dict = {}
                if os.path.exists(FACES_DB):
                    with open(FACES_DB, 'r') as f:
                        local_data = json.load(f)

                # Load deleted faces archive
                deleted_data: dict = {}
                if os.path.exists("deleted_faces.json"):
                    with open("deleted_faces.json", 'r') as f:
                        deleted_data = json.load(f)

                changed = False

                # ── Build uid-keyed lookups for stable comparison ───────────
                server_by_uid: dict = {
                    str(data.get("user_id", "")): (name, data)
                    for name, data in server_db.items()
                    if data.get("user_id")
                }
                local_by_uid: dict = {
                    str(info.get("user_id", "")): lname
                    for lname, info in local_data.items()
                    if info.get("user_id")
                }

                # ── Additions / updates ─────────────────────────────────────
                for uid, (api_name, api_data) in server_by_uid.items():
                    server_embs    = api_data.get("embeddings", [])
                    server_samples = len(server_embs)
                    face_b64       = api_data.get("face_b64", "")

                    if uid not in local_by_uid:
                        # Brand-new person — copy embeddings directly from server
                        # (server already ran ArcFace with align=True, no re-compute needed)
                        print(f"➕ New: {api_name} (uid={uid}) — {server_samples} embedding(s) from server")
                        local_data[api_name] = {
                            "user_id":    uid,
                            "embeddings": server_embs,
                            "face_b64":   face_b64,
                            "samples":    server_samples,
                        }
                        local_by_uid[uid] = api_name
                        changed = True

                    else:
                        local_name = local_by_uid[uid]
                        local_info = local_data[local_name]
                        local_embs = local_info.get("embeddings", [])

                        # Sync when server has more embeddings or local has none
                        need_sync = (server_samples > len(local_embs)) or (len(local_embs) == 0)

                        if need_sync:
                            # Rename if needed
                            if local_name != api_name:
                                local_data[api_name] = local_data.pop(local_name)
                                local_by_uid[uid] = api_name
                                local_name = api_name
                                print(f"✏️  Name: {local_name} → {api_name}")

                            # Replace with full server embedding set directly
                            local_data[local_name]["embeddings"] = server_embs
                            local_data[local_name]["samples"]    = server_samples
                            if face_b64:
                                local_data[local_name]["face_b64"] = face_b64
                            print(f"🔄 Updated {api_name}: {len(local_embs)} → {server_samples} embeddings")
                            changed = True

                        elif local_name != api_name:
                            # Name changed only
                            print(f"✏️  Name: {local_name} → {api_name}")
                            local_data[api_name] = local_data.pop(local_name)
                            local_by_uid[uid] = api_name
                            changed = True

                # ── Deletions ───────────────────────────────────────────────
                for uid in [u for u in local_by_uid if u not in server_by_uid]:
                    lname = local_by_uid[uid]
                    print(f"🗑️  Removed: {lname} (uid={uid})")
                    deleted_data[lname] = local_data.pop(lname)
                    changed = True

                if changed:
                    total_embs = sum(len(v.get("embeddings", [])) for v in local_data.values())
                    print(f"💾 Saved faces.json — {len(local_data)} persons, {total_embs} total embeddings")
                    with open(FACES_DB, 'w') as f:
                        json.dump(local_data, f, indent=2)
                    with open("deleted_faces.json", 'w') as f:
                        json.dump(deleted_data, f, indent=2)
                else:
                    total_embs = sum(len(v.get("embeddings", [])) for v in local_data.values())
                    print(f"✅ Up-to-date: {len(local_data)} persons, {total_embs} total embeddings")

                now = datetime.now().strftime("%I:%M %p")
                self.sync_completed.emit(f"Last Sync: {now}")

            except Exception as e:
                import traceback
                print(f"⚠️ Sync failed: {e}")
                traceback.print_exc()

            for _ in range(10):
                if not self._running:
                    break
                time.sleep(0.5)

    def stop(self):
        self._running = False
        self.wait(3000)

# ═══════════════════════════════════════════════════════════════════════════════
#  Header + Footer
# ═══════════════════════════════════════════════════════════════════════════════
class HeaderBar(QWidget):
    manual_sync_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(_s(58))
        self.setStyleSheet(
            f"background:{_HEADER}; border-bottom: 1px solid {_BORDER};"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(_s(22), 0, _s(22), 0)

        logo = QLabel("🔐  DOOR ACCESS SYSTEM")
        logo.setFont(_font(14, True))
        logo.setStyleSheet(f"color:{_TEXT}; background:transparent;")
        row.addWidget(logo)
        row.addStretch()

        self._clock = QLabel()
        self._clock.setFont(_font(13))
        self._clock.setStyleSheet(f"color:{_MUTED}; background:transparent;")
        self._clock.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self._clock)
        
        row.addSpacing(_s(15))

        self.sync_btn = QPushButton("Sync Now")
        self.sync_btn.setFont(_font(12, True))
        self.sync_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_ACCENT};
                color: white;
                border-radius: {_s(4)}px;
                padding: {_s(6)}px {_s(12)}px;
            }}
            QPushButton:pressed {{
                background: {_SCAN};
            }}
        """)
        self.sync_btn.clicked.connect(self.manual_sync_requested.emit)
        row.addWidget(self.sync_btn)

        self._tick()
        t = QTimer(self); t.timeout.connect(self._tick); t.start(1000)

    def _tick(self):
        self._clock.setText(time.strftime("%H:%M:%S   %a %d %b %Y"))


class FooterBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(_s(40))
        self.setStyleSheet(
            f"background:{_HEADER}; border-top: 1px solid {_BORDER};"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(_s(22), 0, _s(22), 0)

        self._sync_status = QLabel("Last Sync: --")
        self._sync_status.setFont(_font(11, False))
        self._sync_status.setStyleSheet(f"color:{_MUTED}; background:transparent;")
        row.addWidget(self._sync_status)

        # Centre: stretch + logo + label + stretch
        row.addStretch()
        pix = _load_logo_white(_s(80), _s(24))
        if pix:
            logo = QLabel(); logo.setPixmap(pix)
            logo.setStyleSheet("background:transparent;")
            row.addWidget(logo)
            row.addSpacing(_s(10))

        powered = QLabel("Powered by ASBL")
        powered.setFont(_font(10, True))
        powered.setStyleSheet(f"color:{_TEXT}; background:transparent;")
        row.addWidget(powered)
        row.addStretch()

    def set_sync_status(self, text: str):
        self._sync_status.setText(text)


# ===============================================================================
#  Align Panel
# ===============================================================================
class AlignPanel(QWidget):
    """Fifth state: shows live camera + AOI guide overlay.

    Appears when motion is detected anywhere but face is not yet in the central
    AOI zone. Guides the user to step into position without triggering ArcFace.
    """
    def __init__(self, cam_view: QWidget, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{_BG};")
        lay = QVBoxLayout(self)
        lay.setSpacing(0); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(cam_view, 1)   # full-screen camera fill


# ===============================================================================
#  Main Kiosk Window
# ===============================================================================
PANEL_IDX = {"STARTUP": 0, "STANDBY": 1, "SCANNING": 2, "WELCOME": 3, "DENIED": 4, "ALIGN": 5}


class KioskWindow(QMainWindow):
    def __init__(self, door: DoorController, recognizer: FaceRecognizer, cam_index: int):
        super().__init__()
        self._door       = door
        self._recognizer = recognizer
        self._presence   = PresenceDetector()        # Layer-1: motion / presence trigger
        self._state      = "STARTUP"
        self._last_frame : np.ndarray | None = None
        self._last_faces : list = []
        self._last_scan  = 0.0
        self._last_presence = 0.0                    # timestamp of last detected motion
        
        # ── Single-shot recognition state ────────────────────────────────────
        self._scan_count : int   = 0     # ArcFace attempts this session
        self._scan_start : float = 0.0   # when SCANNING state was entered

        self._state_end  = 0.0
        self._reco_busy  = False
        self._result_q   : queue.Queue = queue.Queue()

        # ── Background detection state ───────────────────────────────────────
        self._detect_busy = False
        self._detect_q    : queue.Queue = queue.Queue()
        self._current_faces: list = []

        # -- Window flags: frameless + always on top ----------------------------
        self.setWindowFlags(
            Qt.Window |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )
        self.showFullScreen()
        QApplication.setOverrideCursor(Qt.BlankCursor)

        # -- Central widget + layout --------------------------------------------
        root = QWidget(); self.setCentralWidget(root)
        root.setStyleSheet(f"background:{_BG};")
        vlay = QVBoxLayout(root); vlay.setSpacing(0); vlay.setContentsMargins(0,0,0,0)

        self._header = HeaderBar()
        self._header.manual_sync_requested.connect(self._force_sync)
        vlay.addWidget(self._header)

        self._cam_view = CameraView()

        # State panel stack fills the screen completely between header and footer
        self._stack = QStackedWidget()
        self._panels = {
            "STARTUP":  StartupPanel(),
            "STANDBY":  StandbyPanel(),
            "SCANNING": ScanningPanel(self._cam_view),
            "WELCOME":  WelcomePanel(),
            "DENIED":   DeniedPanel(),
            "ALIGN":    AlignPanel(self._cam_view),  # 5th state: guided alignment
        }
        for key, panel in self._panels.items():
            self._stack.addWidget(panel)

            
        vlay.addWidget(self._stack, 1)

        self._footer = FooterBar()
        vlay.addWidget(self._footer)

        # ── API Sync thread ──────────────────────────────────────────────────
        self._api_worker = ApiSyncWorker()
        self._api_worker.sync_completed.connect(self._on_sync_completed)
        self._api_worker.start()

        # ── Camera thread ────────────────────────────────────────────────────
        self._cam_thread = CameraThread(cam_index)
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.start()

        # ── Main tick timer ──────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

    # ── Frame receiver ────────────────────────────────────────────────────────
    def _on_frame(self, frame: np.ndarray):
        self._last_frame = frame

    def _on_sync_completed(self, status: str):
        self._footer.set_sync_status(status)

    def _force_sync(self):
        # The worker is already running every 5 seconds, but to make the button
        # feel responsive immediately without managing threading locks tightly,
        self._header.sync_btn.setText("Syncing...")
        self._header.sync_btn.setEnabled(False)
        self._api_worker.stop()
        self._api_worker = ApiSyncWorker()
        self._api_worker.sync_completed.connect(self._on_sync_completed)
        self._api_worker.start()
        
        # Reload memory database instantly after a sync completes
        self._recognizer.reload_db()
        
        QTimer.singleShot(1000, lambda: self._header.sync_btn.setText("Sync Now"))
        QTimer.singleShot(1000, lambda: self._header.sync_btn.setEnabled(True))

    # ── Main state-machine tick ───────────────────────────────────────────────
    def _tick(self):
        frame = self._last_frame
        if frame is None:
            return

        # Check for recognition result from background thread
        try:
            name, conf = self._result_q.get_nowait()
            self._reco_busy = False
            self._on_recognition_result(name, conf)
        except queue.Empty:
            pass

        # Check for detection result from background thread
        try:
            faces = self._detect_q.get_nowait()
            self._detect_busy = False
            self._current_faces = faces
        except queue.Empty:
            pass

        now = time.time()

        # ── STARTUP ────────────────────────────────────────────────────────────
        if self._state == "STARTUP":
            if self._recognizer.ready:
                print("Models loaded, moving to STANDBY")
                self._enter("STANDBY")
            else:
                # Discard frames while waiting for AI models to warm up in background
                return

        # ── STANDBY (Cooldown wait) ────────────────────────────────────────────
        elif self._state == "STANDBY":
            # If we are in a forced cooldown (e.g silent deny), ignore camera until it expires
            if now < self._state_end:
                return

            if now - self._last_scan >= SCAN_INTERVAL:
                self._last_scan = now

                # Layer 1 — cheap motion/presence anywhere in frame
                if self._presence.detect(frame):
                    self._last_presence = now
                    H, W = frame.shape[:2]

                    if not self._detect_busy:
                        self._run_detection(frame)

                    # Layer 2 — AOI gate: only face centres inside the zone
                    faces = self._current_faces
                    aoi_faces = [
                        f for f in faces
                        if is_face_in_aoi(f[0], f[1], f[2], f[3], W, H)
                    ]

                    if aoi_faces:
                        # Layer 3 — ArcFace recognition
                        self._last_faces = aoi_faces
                        print(f"[AOI] {len(aoi_faces)} face(s) in zone — scanning…")
                        self._enter("SCANNING")
                        self._run_recognition(frame)
                    else:
                        # Person present but not in AOI — show guide
                        if faces:
                            print(f"[AOI] {len(faces)} face(s) detected outside AOI zone")
                        self._show_guidance_overlay(frame, faces)

                elif (now - self._last_presence) < PRESENCE_COOLDOWN:
                    # Recent motion, MOG2 settled — keep guide briefly
                    self._show_guidance_overlay(frame, [])

                elif DEBUG_AOI:
                    disp = frame.copy()
                    draw_aoi_overlay(disp, state="debug")
                    self._cam_view.set_frame(disp)
            return

        # ── ALIGN ──────────────────────────────────────────────────────────────
        elif self._state == "ALIGN":
            if now - self._last_scan >= SCAN_INTERVAL:
                self._last_scan = now
                H, W = frame.shape[:2]

                if self._presence.detect(frame):
                    self._last_presence = now
                    
                    if not self._detect_busy:
                        self._run_detection(frame)
                    
                    faces = self._current_faces
                    aoi_faces = [
                        f for f in faces
                        if is_face_in_aoi(f[0], f[1], f[2], f[3], W, H)
                    ]
                    if aoi_faces:
                        self._last_faces = aoi_faces
                        print("[AOI] Face moved into zone — scanning…")
                        self._enter("SCANNING")
                        self._run_recognition(frame)
                        return
                    else:
                        self._show_guidance_overlay(frame, faces)
                else:
                    if (now - self._last_presence) >= PRESENCE_COOLDOWN:
                        print("[AOI] Presence gone — returning to STANDBY")
                        self._enter("STANDBY")
                    else:
                        self._show_guidance_overlay(frame, [])
            return

        # ── SCANNING ───────────────────────────────────────────────────────────
        elif self._state == "SCANNING":
            # Safety: timeout — if stuck too long, go to DENIED automatically
            if now - self._scan_start > SCANNING_TIMEOUT:
                if self._scan_count >= 3:
                    print(f"[SCAN] Timeout + 3 fails — HARD DENIED")
                    self._door.close()
                    self._state_end = now + DENIED_HOLD_SECS
                    self._enter("DENIED")
                else:
                    print(f"[SCAN] Timeout — SILENT RETRY (1.5s cooldown)")
                    self._state_end = now + 1.5
                    self._enter("STANDBY")
                return

            # Fire the next recognition pass if we still have attempts left
            if not self._reco_busy and self._scan_count < SCAN_ATTEMPTS:
                self._run_recognition(frame)

            # Live display: face box + AOI overlay + attempt progress text
            display = frame.copy()
            for (fx, fy, fw, fh) in self._last_faces:
                draw_corner_box(display, fx, fy, fx+fw, fy+fh, _CV_CYAN, t=3)
            draw_aoi_overlay(display, state="ready")

            # Progress bar drawn at bottom-left
            H, W = display.shape[:2]
            bar_w = int(W * 0.35)
            progress = int(bar_w * (self._scan_count / max(SCAN_ATTEMPTS, 1)))
            cv2.rectangle(display, (20, H - 38), (20 + bar_w, H - 18), (60, 60, 60), -1)
            cv2.rectangle(display, (20, H - 38), (20 + progress, H - 18), (50, 220, 80), -1)
            
            # Show time left before timeout DENIED
            tl = max(0.0, SCANNING_TIMEOUT - (now - self._scan_start))
            label = f"Scanning... {tl:.1f}s"
            cv2.putText(display, label, (22, H - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            self._cam_view.set_frame(display)


        # ── WELCOME ────────────────────────────────────────────────────────────
        elif self._state == "WELCOME":
            tl = self._state_end - now
            self._panels["WELCOME"].set_time_left(tl, DOOR_OPEN_SECS)
            if tl <= 0:
                self._door.close()
                self._enter("STANDBY")
                return

        # ── DENIED ─────────────────────────────────────────────────────────────
        elif self._state == "DENIED":
            tl = self._state_end - now
            self._panels["DENIED"].set_time_left(tl, DENIED_HOLD_SECS)
            if tl <= 0:
                self._enter("STANDBY")
                return

    # ── Guidance overlay ───────────────────────────────────────────────────────
    def _show_guidance_overlay(self, frame: np.ndarray, all_faces: list):
        """Render live camera + amber AOI box; transition STANDBY -> ALIGN."""
        display = frame.copy()
        # Draw off-centre faces in dim red so user sees they are out of zone
        for (fx, fy, fw, fh) in all_faces:
            draw_corner_box(display, fx, fy, fx+fw, fy+fh, _CV_RED, t=2, L=20)
        draw_aoi_overlay(display, state="wait")    # amber box + "Step into the frame"
        self._cam_view.set_frame(display)
        if self._state == "STANDBY":
            self._enter("ALIGN")



    # ── Detection in background thread ───────────────────────────────────────
    def _run_detection(self, frame: np.ndarray):
        if self._detect_busy:
            return
        self._detect_busy = True
        
        q = self._detect_q
        recognizer = self._recognizer
        # pass a copy of the frame to prevent it from mutating while thread runs
        frame_copy = frame.copy()

        def work():
            faces = recognizer.detect_faces(frame_copy)
            q.put(faces)

        threading.Thread(target=work, daemon=True).start()

    # ── Recognition in background thread ─────────────────────────────────────
    def _run_recognition(self, frame: np.ndarray):
        if self._reco_busy or not self._last_faces:
            return
        self._reco_busy = True

        fx, fy, fw, fh = max(self._last_faces, key=lambda r: r[2] * r[3])
        
        # ── Dynamic Tracking Pad ───────────────────────────────────────────────────
        # In SCANNING state, self._last_faces is frozen from the ALIGN trigger.
        # By passing a generous 100px padded crop to RetinaFace down the line,
        # RetinaFace can easily find the person's new location if they swayed or 
        # stepped forward during the 1-2 second multi-frame scanning process.
        pad = 100
        H, W = frame.shape[:2]
        crop = frame[max(fy-pad,0):min(fy+fh+pad,H), max(fx-pad,0):min(fx+fw+pad,W)]


        q = self._result_q
        recognizer = self._recognizer

        def work():
            # Wait for stabilization to prevent blurry frames matching
            time.sleep(SCAN_STABILIZE_MS)
            
            if not recognizer.ready:
                # Wait for model to load
                for _ in range(60):
                    if recognizer.ready:
                        break
                    time.sleep(0.5)
            name, conf = recognizer.identify(crop)
            q.put((name, conf))

        threading.Thread(target=work, daemon=True).start()

    def _on_recognition_result(self, name: str | None, conf: float):
        """Process a single ArcFace recognition pass.
        
        Succeeds immediately on the first match. If no match, it increments scan attempt
        counter. If attempts run out or timeout expires (handled in _tick), goes to DENIED.
        """
        self._scan_count += 1
        
        if name:
            # Immediate Success Branch
            print(f"[RECOGNITION] WELCOME: {name} ({conf}%) on attempt {self._scan_count}")
            self._door.open()
            self._panels["WELCOME"].set_info(name, int(conf))
            self._state_end = time.time() + DOOR_OPEN_SECS
            self._enter("WELCOME")
            return
            
        else:
            print(f"[RECOGNITION] Attempt {self._scan_count}/{SCAN_ATTEMPTS}: no match")

        # Not all passes done yet — keep scanning
        if self._scan_count < SCAN_ATTEMPTS:
            return

        # Out of attempts without a match
        if self._scan_count >= 3:
            print(f"DENIED: 0 matches after {SCAN_ATTEMPTS} attempts")
            self._door.close()
            self._state_end = time.time() + DENIED_HOLD_SECS
            self._enter("DENIED")
        else:
            print(f"SILENT DENIED: {self._scan_count} attempts. 1.5s cooldown.")
            self._state_end = time.time() + 1.5
            self._enter("STANDBY")



    # ── State transition ──────────────────────────────────────────────────────
    def _enter(self, state: str):
        self._state = state
        self._stack.setCurrentIndex(PANEL_IDX[state])

        if state == "STANDBY":
            self._last_scan     = 0.0
            self._last_presence = 0.0
        elif state == "ALIGN":
            self._last_scan = 0.0   # immediately re-check on next tick
        elif state == "SCANNING":
            self._scan_count = 0
            self._scan_start = time.time()



    # ── Keyboard ──────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()

    # ── Teardown ──────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        print("\n🧹  Shutting down kiosk…")
        self._timer.stop()
        self._cam_thread.stop()
        self._api_worker.stop()
        self._door.cleanup()
        self._recognizer.stop()
        QApplication.restoreOverrideCursor()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    app    = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── Responsive scale factor ────────────────────────────────────────────────
    # Derive a scale relative to a 1920x1080 baseline so the UI looks right
    # on every screen: small 7" kiosk (1024x600) up to 4K displays.
    global _SCALE
    screen = app.primaryScreen().geometry()
    _SCALE = min(screen.width() / 1920.0, screen.height() / 1080.0)
    _SCALE = max(0.45, min(_SCALE, 2.0))   # clamp: 45 % … 200 %
    print(f"🖥️   Screen: {screen.width()}x{screen.height()}  →  scale factor: {_SCALE:.2f}")

    # Dark palette so Qt fills any unset areas correctly
    from PySide6.QtGui import QPalette
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor(_BG))
    pal.setColor(QPalette.WindowText, QColor(_TEXT))
    app.setPalette(pal)

    door       = DoorController(args.port, args.baud)
    recognizer = FaceRecognizer()

    # Ensure door is closed on hard exit
    atexit.register(door.cleanup)

    win = KioskWindow(door, recognizer, args.cam)

    def _sig(signum, _):
        print(f"\n🛑  Signal {signal.Signals(signum).name} — exiting…")
        win.close()
        app.quit()

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
