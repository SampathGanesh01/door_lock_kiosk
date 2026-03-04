from __future__ import annotations
import cv2
import numpy as np
import os
import time
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPainterPath, QBrush
from PySide6.QtSvg import QSvgRenderer
import threading
import serial
import warnings

# --- Config ---
import importlib.util
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# --- Door Controller ---
class DoorController:
    def __init__(self, port=None, baud=None, open_cmd=None, close_cmd=None):
        self.ser = None
        self._lock = threading.Lock()
        self._closed = True
        self.open_cmd = open_cmd if open_cmd is not None else config.DOOR_OPEN_CMD.encode()
        self.close_cmd = close_cmd if close_cmd is not None else config.DOOR_CLOSE_CMD.encode()
        port = port if port is not None else config.SERIAL_PORT
        baud = baud if baud is not None else config.SERIAL_BAUD
        if port:
            try:
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)  # Wait for Arduino to reset
                self._closed = True
            except Exception as e:
                warnings.warn(f"[Door] Serial error: {e}")
                self.ser = None

    def open(self):
        with self._lock:
            print(f"🔓  DOOR OPEN  → {self.open_cmd.decode()}")
            if self.ser and self.ser.is_open:
                self.ser.write(self.open_cmd)
                self._closed = False

    def close(self):
        with self._lock:
            print(f"🔒  DOOR CLOSE → {self.close_cmd.decode()}")
            if self.ser and self.ser.is_open:
                self.ser.write(self.close_cmd)
                self._closed = True

    def cleanup(self):
        with self._lock:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                self._closed = True

    def __del__(self):
        self.cleanup()
from insightface.app import FaceAnalysis

# --- Qt colours ---
_BG       = "#0a0a1a"
_TEXT     = "#e2e2f0"
_MUTED    = "#52526e"
_GREEN_BG = "#020e08"
_GREEN    = "#22c55e"
_ACCENT   = "#6366f1"
_BORDER   = "#1e1e40"

def _font(size: int, bold: bool = False) -> QFont:
    f = QFont()
    f.setPointSize(size)
    f.setBold(bold)
    return f

def _load_logo_white(logo_path: str, w: int = 240, h: int = 74):
    if not os.path.exists(logo_path):
        return None
    renderer = QSvgRenderer(logo_path)
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))          # transparent canvas
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    renderer.render(p)
    p.end()
    
    # Safely load the byte array across different PySide6 versions
    ptr = img.constBits()
    arr = np.array(ptr).reshape((h, w, 4)).copy()
    
    mask = arr[:, :, 3] > 10             # non-transparent pixels only
    arr[mask, 0] = 255 - arr[mask, 0]    # invert B
    arr[mask, 1] = 255 - arr[mask, 1]    # invert G
    arr[mask, 2] = 255 - arr[mask, 2]    # invert R
    
    result = QImage(arr.tobytes(), w, h, w * 4, QImage.Format_ARGB32)
    return QPixmap.fromImage(result)


# --- Setup ---
app_insight = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(320, 320))

def load_authorized():
    known = {}
    auth_dir = "authorized_users"
    if not os.path.exists(auth_dir):
        os.makedirs(auth_dir)
    for file in os.listdir(auth_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file)[0]
            img = cv2.imread(os.path.join(auth_dir, file))
            if img is not None:
                faces = app_insight.get(img)
                if faces:
                    known[name] = faces[0].normed_embedding
    return known

# --- UI Elements ---
class SimpleKiosk(QMainWindow):
    def __init__(self, door=None):
        super().__init__()
        self.setWindowTitle("Door Access (Clean UI)")

        # Make full screen without borders or window controls
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.showFullScreen()

        self.setStyleSheet(f"background-color: {_BG}; color: {_TEXT};")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header / Logo Area
        self.header = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asbl_logo.svg")
        self.logo_pixmap = _load_logo_white(logo_path, 280, 86)

        if self.logo_pixmap:
            self.header.setPixmap(self.logo_pixmap)
        else:
            self.header.setText("ASBL GYM")
            self.header.setFont(_font(28, True))

        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(f"color: {_TEXT}; margin-top: 80px;")

        self.status_lbl = QLabel("Please stand in front of the camera")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setFont(_font(14, False))
        self.status_lbl.setStyleSheet(f"color: {_MUTED}; margin-top: 10px;")

        # Thumbnail Circle
        self.thumb_lbl = QLabel()
        self.thumb_lbl.setFixedSize(240, 240)
        self.thumb_lbl.setStyleSheet(f"""
            border: 2px solid {_BORDER}; 
            border-radius: 120px; 
            background-color: #0c0c24;
        """)
        self.thumb_lbl.setAlignment(Qt.AlignCenter)

        # Footer
        self.footer = QLabel("Powered by ASBL")
        self.footer.setFont(_font(10, True))
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setStyleSheet(f"color: {_MUTED}; margin-bottom: 30px;")

        layout.addWidget(self.header)
        layout.addWidget(self.status_lbl)
        layout.addStretch()
        layout.addWidget(self.thumb_lbl, alignment=Qt.AlignCenter)
        layout.addStretch()
        layout.addWidget(self.footer)

        self.known_faces = load_authorized()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Camera is not available. Exiting.")
            if door:
                door.cleanup()
            sys.exit(1)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # State: 0 = Searching, 1 = Detected (Paused)
        self.state = 0

        self.door = door

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(30) # 30ms loop

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        # Release camera and door resources
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.door:
                self.door.cleanup()
        except Exception:
            pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def update_loop(self):
        # 1. ALWAYS read a frame to drain the camera buffer so old frames don't pile up!
        ret, frame = self.cap.read()
        if not ret: return
        
        # 2. If we are currently showing a success screen, ignore the frame
        if self.state == 1:
            return
        
        # 3. Very fast detection on the absolute latest frame
        faces = app_insight.get(frame)
        
        if faces:
            face = faces[0]
            current_emb = face.normed_embedding
            
            best_name = "Unknown"
            best_sim = 0
            
            for name, saved_emb in self.known_faces.items():
                sim = np.dot(current_emb, saved_emb)
                if sim > 0.45:
                    best_name = name
                    best_sim = sim
                    break
            
            if best_name != "Unknown":
                self.show_success(best_name, frame, face.bbox.astype(int))

    def show_success(self, name, frame, bbox):
        self.state = 1

        # Open the door if available
        if self.door:
            self.door.open()

        # Update colors to success state
        self.setStyleSheet(f"background-color: {_GREEN_BG}; color: {_TEXT};")
        self.header.setText("✓ ACCESS GRANTED")
        self.header.setStyleSheet(f"color: {_GREEN}; margin-top: 60px;")

        self.status_lbl.setText(f"Welcome, {name.capitalize()}")
        self.status_lbl.setFont(_font(16, True))
        self.status_lbl.setStyleSheet(f"color: {_GREEN}; margin-top: 10px;")

        self.thumb_lbl.setStyleSheet(f"""
            border: 4px solid {_GREEN}; 
            border-radius: 120px; 
            background-color: #0c0c24;
        """)

        # Crop thumbnail
        x1, y1, x2, y2 = bbox

        # Expand crop slightly for better look
        pad = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size != 0:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            height, width, channel = crop.shape
            bytes_per_line = 3 * width
            q_img = QImage(crop.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img).scaled(240, 240, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

            circular_pix = self.make_circular_pixmap(pix)
            self.thumb_lbl.setPixmap(circular_pix)

        # Wait config.DOOR_OPEN_SECS seconds, then reset and close door
        QTimer.singleShot(int(config.DOOR_OPEN_SECS * 1000), self._access_cleanup)

    def _access_cleanup(self):
        if self.door:
            self.door.close()
        self.reset_to_standby()

    def make_circular_pixmap(self, source_pixmap):
        size = min(source_pixmap.width(), source_pixmap.height())
        target = QPixmap(size, size)
        target.fill(Qt.transparent)
        
        painter = QPainter(target)
        painter.setRenderHint(QPainter.Antialiasing)
        
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)
        
        # Center the source pixmap
        x = (source_pixmap.width() - size) // 2
        y = (source_pixmap.height() - size) // 2
        painter.drawPixmap(0, 0, source_pixmap, x, y, size, size)
        painter.end()
        return target

    def reset_to_standby(self):
        self.state = 0
        self.setStyleSheet(f"background-color: {_BG}; color: {_TEXT};")
        
        if hasattr(self, 'logo_pixmap') and self.logo_pixmap:
            self.header.setPixmap(self.logo_pixmap)
        else:
            self.header.setText("ASBL GYM")
            self.header.setFont(_font(28, True))
            
        self.header.setStyleSheet(f"color: {_TEXT}; margin-top: 80px;")
        
        self.status_lbl.setText("Please stand in front of the camera")
        self.status_lbl.setFont(_font(14, False))
        self.status_lbl.setStyleSheet(f"color: {_MUTED}; margin-top: 10px;")
        
        self.thumb_lbl.clear()
        self.thumb_lbl.setStyleSheet(f"""
            border: 2px solid {_BORDER}; 
            border-radius: 120px; 
            background-color: #0c0c24;
        """)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Face Kiosk")
    parser.add_argument("--port", default=None, help="Arduino serial port (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud", default=None, type=int, help="Serial baud rate")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    door = DoorController(
        port=args.port if args.port is not None else config.SERIAL_PORT,
        baud=args.baud if args.baud is not None else config.SERIAL_BAUD,
        open_cmd=config.DOOR_OPEN_CMD.encode(),
        close_cmd=config.DOOR_CLOSE_CMD.encode()
    ) if (args.port or config.SERIAL_PORT) else None

    kiosk = SimpleKiosk(door=door)
    kiosk.show()
    sys.exit(app.exec())
