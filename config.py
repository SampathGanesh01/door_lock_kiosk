# simple_face_qt configuration file
# You can tune these values for your hardware and environment

SERIAL_PORT = None           # e.g. '/dev/ttyACM0'
SERIAL_BAUD = 9600           # e.g. 9600
DOOR_OPEN_CMD = '@a,o#'      # Serial command to open door
DOOR_CLOSE_CMD = '@a,1#'     # Serial command to close door
DOOR_OPEN_SECS = 8           # Seconds to keep door open after access granted
CAM_INDEX = 0                # Camera index for cv2.VideoCapture

# Add more config variables as needed
