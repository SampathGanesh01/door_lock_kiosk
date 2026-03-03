"""
face_test.py
============
Simple face detection + recognition test using:
  - OpenCV  → camera + Haar face detection
  - DeepFace (ArcFace) → 512-D embeddings
  - faces.json → local face database

Press Q to quit.
"""

import cv2
import json
import os
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
FACES_DB         = "faces.json"
RECOGNITION_MODEL = "ArcFace"
COSINE_THRESHOLD  = 0.45        # unified with kiosk and server (0.50 was too loose, 0.40 too strict)
MIN_FACE_PIXELS   = 70
CAM_INDEX         = 0

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (50,  220, 50)
RED    = (50,  50,  220)
YELLOW = (50,  200, 200)
WHITE  = (240, 240, 240)
BLACK  = (0,   0,   0)

# ── Load ArcFace model once ───────────────────────────────────────────────────
print("🔄 Loading ArcFace model … (first run downloads ~500 MB)")
import warnings, logging, os as _os
_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from deepface import DeepFace

# Warm up
try:
    DeepFace.represent(np.zeros((112, 112, 3), np.uint8),
                       model_name=RECOGNITION_MODEL,
                       enforce_detection=False, detector_backend="skip")
except Exception:
    pass
print("✅ ArcFace ready.\n")


# ── Load faces.json ───────────────────────────────────────────────────────────
def load_db():
    if not os.path.exists(FACES_DB):
        print(f"⚠️  {FACES_DB} not found — running in detection-only mode.")
        return {}
    with open(FACES_DB) as f:
        db = json.load(f)
    total_emb = sum(len(v.get("embeddings", [])) for v in db.values())
    print(f"📂 Loaded {len(db)} persons, {total_emb} embeddings from {FACES_DB}")
    return db


# ── Cosine distance ───────────────────────────────────────────────────────────
def cosine_dist(a, b):
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 2.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


# ── Identify a face crop ──────────────────────────────────────────────────────
def identify(bgr_crop, db):
    """Returns (name, confidence_pct) or (None, 0) if unknown."""
    if not db:
        return None, 0
    try:
        res = DeepFace.represent(
            img_path=cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB),
            model_name=RECOGNITION_MODEL,
            enforce_detection=False,
            detector_backend="skip",
            align=True,   # MUST match server (align=True); mismatch causes high cosine dist
        )
        if not res or "embedding" not in res[0]:
            return None, 0
        emb = np.array(res[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"  embed error: {e}")
        return None, 0

    best_name, best_dist = None, float("inf")
    for name, data in db.items():
        for stored in data.get("embeddings", []):
            d = cosine_dist(emb, stored)
            if d < best_dist:
                best_dist, best_name = d, name

    if best_dist <= COSINE_THRESHOLD:
        conf = int((1 - best_dist / COSINE_THRESHOLD) * 100)
        conf = max(0, min(conf, 100))
        return best_name, conf

    # Debug: show why it failed
    print(f"     best={best_name} dist={best_dist:.3f} (threshold={COSINE_THRESHOLD})")
    return None, 0


# ── Draw a nice bounding box with label ──────────────────────────────────────
def draw_box(frame, x, y, w, h, name, conf, color):
    x2, y2 = x + w, y + h
    # Corner bracket style
    L = max(20, w // 5)
    t = 2
    for p in [((x,y),(x+L,y)),((x,y),(x,y+L)),
               ((x2,y),(x2-L,y)),((x2,y),(x2,y+L)),
               ((x,y2),(x+L,y2)),((x,y2),(x,y2-L)),
               ((x2,y2),(x2-L,y2)),((x2,y2),(x2,y2-L))]:
        cv2.line(frame, p[0], p[1], color, t, cv2.LINE_AA)

    if name:
        label = f"{name}  {conf}%"
    else:
        label = "Unknown"

    fs   = 0.55
    ft   = 1
    pad  = 6
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    bx1, by1 = x, y - th - pad * 2
    bx2, by2 = x + tw + pad * 2, y

    # Background pill
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(frame, label,
                (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, fs, BLACK, ft, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    db = load_db()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("📷 Camera started. Press Q to quit.\n")

    # We run identify only when the frame counter hits the interval
    # (ArcFace is slow — don't run every frame)
    RECO_EVERY_N = 10        # run recognition every N frames
    frame_count  = 0
    last_results = []        # list of (x, y, w, h, name, conf, color)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1

        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = cascade.detectMultiScale(
            gray, scaleFactor=1.12, minNeighbors=5,
            minSize=(MIN_FACE_PIXELS, MIN_FACE_PIXELS)
        )

        if frame_count % RECO_EVERY_N == 0 and len(rects) > 0:
            last_results = []
            H, W = frame.shape[:2]
            for (fx, fy, fw, fh) in rects:
                pad  = 20
                crop = frame[max(fy-pad,0):min(fy+fh+pad,H),
                             max(fx-pad,0):min(fx+fw+pad,W)]

                name, conf = identify(crop, db)
                color = GREEN if name else RED
                last_results.append((fx, fy, fw, fh, name, conf, color))
                status = f"✅ {name} ({conf}%)" if name else "❌ Unknown"
                print(f"  [{frame_count:05d}] {status}")

        # Draw last known results every frame
        for (fx, fy, fw, fh, name, conf, color) in last_results:
            draw_box(frame, fx, fy, fw, fh, name, conf, color)

        # Show face count
        n = len(rects)
        cv2.putText(frame, f"Faces: {n}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit",
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)

        cv2.imshow("Face Recognition Test  [faces.json]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Done.")


if __name__ == "__main__":
    main()
