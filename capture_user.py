import cv2
import os

# Create directory if it doesn't exist
folder = 'authorized_users'
if not os.path.exists(folder):
    os.makedirs(folder)

cam = cv2.VideoCapture(0)
print("Press 's' to save photo, 'q' to quit.")

while True:
    ret, frame = cam.read()
    cv2.imshow("Capture Face", frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        name = input("Enter name for this person: ")
        img_name = os.path.join(folder, f"{name}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
    elif key & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
