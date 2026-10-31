import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)

node = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1) # mirror effect on camera

    current_time = datetime.now().strftime("%H:%M:%S") # time in corner
    cv2.putText(frame, current_time, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
        1, (255, 255, 255), 2)

    if node == 0:
        output = frame  # normal camera

    elif node == 1:
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale

    elif node == 2:
        output = cv2.GaussianBlur(frame, (15, 15), 0)   # blur - higher number, more blur

    elif node == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # edge detection effect
        output = cv2.Canny(gray, 50, 150)

    cv2.imshow("Gesture Filter Project", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("0"):
        node = 0
    elif key == ord("1"):
        node = 1
    elif key == ord("2"):
        node = 2
    elif key == ord("3"):
        node = 3
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()