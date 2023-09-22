import cv2
import mediapipe as mp
import math

def drawing(img, landmarks):
    for landmark in landmarks:
        x, y, _ = landmark
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

def classify_pose(landmarks):
    t_pose = False
    lea = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    rea = calculate_angle(landmarks[16], landmarks[14], landmarks[12])
    lsa = calculate_angle(landmarks[13], landmarks[11], landmarks[23])
    rsa = calculate_angle(landmarks[24], landmarks[12], landmarks[14])

    if (165 < lea <= 180) and (165 < rea <= 180) and (75 < rsa <= 105) and (75 < lsa <= 105):
        t_pose = True

    return t_pose

def process_video():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)  # 0 represents the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []

        if results.pose_landmarks:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * frame.shape[1]) for lm in results.pose_landmarks.landmark]

            t_pose = classify_pose(landmarks)

            if t_pose:
                result_text = "t-Pose: True"
            else:
                result_text = "t-Pose: False"
            cv2.putText(frame, result_text, (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            drawing(frame, landmarks)

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video stream
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()


