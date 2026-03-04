import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# 1. Initialize InsightFace with the lightweight 'buffalo_s' model
# This uses MobileFaceNet which is optimized for CPU/ARM devices
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def get_embedding(img_path):
    """Loads an image and returns its facial embedding vector."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return None
    
    faces = app.get(img)
    if len(faces) > 0:
        # Return the normalized embedding of the first face found
        return faces[0].normed_embedding
    else:
        print(f"No face detected in {img_path}")
        return None

# 2. Automatically load all authorized users from the folder
authorized_dir = "authorized_users"
known_faces = {} # Dictionary to store {Name: Embedding}

print("--- Loading Authorized Users ---")
for filename in os.listdir(authorized_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        path = os.path.join(authorized_dir, filename)
        
        embedding = get_embedding(path)
        if embedding is not None:
            known_faces[name] = embedding
            print(f"Successfully loaded: {name}")

if not known_faces:
    print("Warning: No authorized users loaded. Check your 'authorized_users' folder.")

# 3. Start the Real-Time Stream
cap = cv2.VideoCapture(0)
print("\nStarting Camera Stream... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Resize for even faster processing on Pi
    # display_frame = cv2.resize(frame, (640, 480)) 

    # Extract faces from the current frame
    faces = app.get(frame)

    for face in faces:
        # Draw bounding box
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Recognition Logic: Compare live face against all known faces
        current_embedding = face.normed_embedding
        best_match_name = "Unknown"
        max_sim = 0

        for name, saved_embedding in known_faces.items():
            # Calculate Cosine Similarity
            sim = np.dot(current_embedding, saved_embedding)
            if sim > max_sim:
                max_sim = sim
                if sim > 0.45: # Threshold: > 0.45 is usually the same person
                    best_match_name = name

        # Display Name and Similarity Score
        label = f"{best_match_name} ({max_sim:.2f})"
        color = (0, 255, 0) if best_match_name != "Unknown" else (0, 0, 255)
        
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Trigger Action (e.g., Door Open)
        if best_match_name != "Unknown":
            # print(f"ACCESS GRANTED for {best_match_name}")
            pass

    # Show the video
    cv2.imshow("Pi 4 Access Control - InsightFace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
