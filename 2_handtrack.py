"""
Fingertip landmark id
0   wrist
4   thumb tip!!
8   index finger tip!!
12  middle finger tip!!
16  ring finger tip!!
20  pinky tip!!
"""
import cv2
import mediapipe as mp
import os
import time
from datetime import datetime

# model file
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)

# mediapipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

# fingertip landmark id
FINGER_TIPS = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # flip camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert to mediapipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # timestamp
    timestamp_ms = time.monotonic_ns() // 1_000_000

    # detect hand
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.hand_landmarks:
        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            # recognize left or right hand
            # (swap required for camera flip)
            label = result.handedness[hand_index][0].category_name

            if label == "Left":
                label = "Right"
            else:
                label = "Left"

            # draw hand connections
            for connection in (
                mp.tasks.vision
                .HandLandmarksConnections
                .HAND_CONNECTIONS
            ):
                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                x1 = int(start.x * width)
                y1 = int(start.y * height)

                x2 = int(end.x * width)
                y2 = int(end.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

            # draw all landmark
            for landmark in hand_landmarks:
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (0, 255, 0),
                    -1
                )

            # highlight fingertips
            for tip_id in FINGER_TIPS:
                tip = hand_landmarks[tip_id]

                cx = int(tip.x * width)
                cy = int(tip.y * height)

                cv2.circle(
                    frame,
                    (cx, cy),
                    10,
                    (0, 0, 255),
                    -1
                )

            # show hand label
            wrist = hand_landmarks[0]

            wrist_x = int(wrist.x * width)
            wrist_y = int(wrist.y * height)

            cv2.putText(
                frame,
                label,
                (wrist_x, wrist_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # index fingertip
            index_tip = hand_landmarks[8]

            index_x = int(index_tip.x * width)
            index_y = int(index_tip.y * height)

            # show index coordinates
            coordinate_text = (
                f"({index_tip.x:.2f}, "
                f"{index_tip.y:.2f})"
            )

            cv2.putText(
                frame,
                coordinate_text,
                (index_x + 10, index_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

    # current time
    current_time = datetime.now().strftime("%H:%M:%S")

    cv2.putText(
        frame,
        current_time,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # show result
    cv2.imshow("MediaPipe Hand Tracking", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# clean up
cap.release()
landmarker.close()
cv2.destroyAllWindows()