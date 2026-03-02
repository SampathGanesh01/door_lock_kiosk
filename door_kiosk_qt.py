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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
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
COSINE_THRESHOLD  = 0.40
MIN_FACE_PIXELS   = 70
SCAN_INTERVAL     = 1.5
DOOR_OPEN_SECS    = 5
DENIED_HOLD_SECS  = 3
TICK_MS           = 33          # ~30 fps

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
    return max(1, int(n * _SCALE))


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
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if self._stop.is_set():
            return
        print("🔄  Loading ArcFace model in background…")
        from deepface import DeepFace as DF
        self._DeepFace = DF
        try:
            DF.represent(np.zeros((112, 112, 3), np.uint8),
                         model_name=RECOGNITION_MODEL,
                         enforce_detection=False, detector_backend="skip")
        except Exception:
            pass
        if not self._stop.is_set():
            self._ready = True
            print("✅  ArcFace ready.")

    def stop(self):
        self._stop.set()

    @property
    def ready(self):
        return self._ready

    def detect_faces(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.12, minNeighbors=5,
            minSize=(MIN_FACE_PIXELS, MIN_FACE_PIXELS)
        )
        return list(rects) if len(rects) else []

    def embed(self, bgr_crop):
        if not self._ready:
            return None
        try:
            res = self._DeepFace.represent(
                img_path=cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB),
                model_name=RECOGNITION_MODEL,
                enforce_detection=False, detector_backend="skip", align=True,
            )
            if res and "embedding" in res[0]:
                return np.array(res[0]["embedding"], dtype=np.float32)
        except Exception:
            pass
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
        if emb is None or not os.path.exists(FACES_DB):
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
            return best_name, max(0, min(int((1 - best_dist / COSINE_THRESHOLD) * 100), 100))
        return None, 0


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenCV overlay helpers
# ═══════════════════════════════════════════════════════════════════════════════
def draw_corner_box(img, x1, y1, x2, y2, color, t=3, L=30):
    for pts in [((x1,y1),(x1+L,y1)), ((x1,y1),(x1,y1+L)),
                ((x2,y1),(x2-L,y1)), ((x2,y1),(x2,y1+L)),
                ((x1,y2),(x1+L,y2)), ((x1,y2),(x1,y2-L)),
                ((x2,y2),(x2-L,y2)), ((x2,y2),(x2,y2-L))]:
        cv2.line(img, pts[0], pts[1], color, t, cv2.LINE_AA)


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
        outer.addWidget(_lbl("Welcome to ASBL Homes GYM", 26, True, _TEXT))
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{_SCAN_BG};")
        lay = QVBoxLayout(self)
        lay.setSpacing(10); lay.setContentsMargins(24, 28, 24, 20)

        lay.addStretch(1)
        self._arc = SpinningArcWidget()
        self._arc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._arc, 4, Qt.AlignCenter)
        lay.addStretch(1)

        lay.addWidget(_lbl("IDENTIFYING", 18, True, _SCAN))
        lay.addWidget(_lbl("Please hold still…", 11, False, _MUTED))
        lay.addSpacing(16)


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
#  Header + Footer
# ═══════════════════════════════════════════════════════════════════════════════
class HeaderBar(QWidget):
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Kiosk Window
# ═══════════════════════════════════════════════════════════════════════════════
PANEL_IDX = {"STANDBY": 0, "SCANNING": 1, "WELCOME": 2, "DENIED": 3}

class KioskWindow(QMainWindow):
    def __init__(self, door: DoorController, recognizer: FaceRecognizer, cam_index: int):
        super().__init__()
        self._door       = door
        self._recognizer = recognizer
        self._state      = "STANDBY"
        self._last_frame : np.ndarray | None = None
        self._last_faces : list = []
        self._last_scan  = 0.0
        self._state_end  = 0.0
        self._reco_busy  = False
        self._result_q   : queue.Queue = queue.Queue()

        # ── Window flags: frameless + always on top ──────────────────────────
        self.setWindowFlags(
            Qt.Window |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )
        self.showFullScreen()
        QApplication.setOverrideCursor(Qt.BlankCursor)

        # ── Central widget + layout ──────────────────────────────────────────
        root = QWidget(); self.setCentralWidget(root)
        root.setStyleSheet(f"background:{_BG};")
        vlay = QVBoxLayout(root); vlay.setSpacing(0); vlay.setContentsMargins(0,0,0,0)

        vlay.addWidget(HeaderBar())

        # Content row — horizontal split: camera | separator | state-panel
        content = QWidget()
        self._hlay = QHBoxLayout(content)
        self._hlay.setSpacing(0); self._hlay.setContentsMargins(0,0,0,0)

        # Camera view (65%) — hidden during STANDBY / WELCOME / DENIED
        self._cam_view = CameraView()
        self._hlay.addWidget(self._cam_view, 65)

        # Vertical separator — also hidden when camera is hidden
        self._sep = QWidget(); self._sep.setFixedWidth(1)
        self._sep.setStyleSheet(f"background:{_BORDER};")
        self._hlay.addWidget(self._sep)

        # State panel stack
        self._stack = QStackedWidget()
        self._panels = {
            "STANDBY":  StandbyPanel(),
            "SCANNING": ScanningPanel(),
            "WELCOME":  WelcomePanel(),
            "DENIED":   DeniedPanel(),
        }
        for key, panel in self._panels.items():
            self._stack.addWidget(panel)
        self._hlay.addWidget(self._stack, 35)

        vlay.addWidget(content, 1)
        vlay.addWidget(FooterBar())

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

        now = time.time()

        if self._state == "STANDBY":
            if now - self._last_scan >= SCAN_INTERVAL:
                self._last_scan = now
                faces = self._recognizer.detect_faces(frame)
                if faces:
                    self._last_faces = faces
                    print("👥  Face detected — starting recognition…")
                    self._enter("SCANNING")
                    self._run_recognition(frame)
            # Camera hidden in STANDBY — no frame rendering needed
            return

        elif self._state == "SCANNING":
            display = frame.copy()
            for (fx, fy, fw, fh) in self._last_faces:
                draw_corner_box(display, fx, fy, fx+fw, fy+fh, _CV_CYAN, t=3)
            self._cam_view.set_frame(display)

        elif self._state == "WELCOME":
            tl = self._state_end - now
            self._panels["WELCOME"].set_time_left(tl, DOOR_OPEN_SECS)
            if tl <= 0:
                self._door.close()
                self._enter("STANDBY")
                return

        elif self._state == "DENIED":
            tl = self._state_end - now
            self._panels["DENIED"].set_time_left(tl, DENIED_HOLD_SECS)
            if tl <= 0:
                self._enter("STANDBY")
                return

    # ── Recognition in background thread ─────────────────────────────────────
    def _run_recognition(self, frame: np.ndarray):
        if self._reco_busy or not self._last_faces:
            return
        self._reco_busy = True

        fx, fy, fw, fh = max(self._last_faces, key=lambda r: r[2] * r[3])
        pad = 30
        H, W = frame.shape[:2]
        crop = frame[max(fy-pad,0):min(fy+fh+pad,H), max(fx-pad,0):min(fx+fw+pad,W)]

        q = self._result_q
        recognizer = self._recognizer

        def work():
            if not recognizer.ready:
                # Wait for model to load
                for _ in range(60):
                    if recognizer.ready:
                        break
                    time.sleep(0.5)
            name, conf = recognizer.identify(crop)
            q.put((name, conf))

        threading.Thread(target=work, daemon=True).start()

    def _on_recognition_result(self, name, conf):
        if name:
            print(f"✅  Recognised: {name} ({conf}%)")
            self._door.open()
            self._panels["WELCOME"].set_info(name, conf)
            self._state_end = time.time() + DOOR_OPEN_SECS
            self._enter("WELCOME")
        else:
            print("❌  Not recognised.")
            self._door.close()
            self._state_end = time.time() + DENIED_HOLD_SECS
            self._enter("DENIED")

    # ── State transition ──────────────────────────────────────────────────────
    def _enter(self, state: str):
        self._state = state
        self._stack.setCurrentIndex(PANEL_IDX[state])

        show_cam = (state == "SCANNING")
        self._cam_view.setVisible(show_cam)
        self._sep.setVisible(show_cam)

        # ★ Critical: Qt keeps the stretch-factor allocation (65/35) even when
        #   the camera widget is hidden, leaving the stack in only 35% width.
        #   Explicitly set stretch to 0/100 so the stack fills the full screen.
        if show_cam:
            self._hlay.setStretchFactor(self._cam_view, 65)
            self._hlay.setStretchFactor(self._stack, 35)
        else:
            self._hlay.setStretchFactor(self._cam_view, 0)
            self._hlay.setStretchFactor(self._stack, 100)

        if state == "STANDBY":
            self._last_scan = 0.0

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()

    # ── Teardown ──────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        print("\n🧹  Shutting down kiosk…")
        self._timer.stop()
        self._cam_thread.stop()
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
    pal = app.palette()
    pal.setColor(pal.Window, QColor(_BG))
    pal.setColor(pal.WindowText, QColor(_TEXT))
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
