# Computer_Vision-Project
import cv2
import mediapipe as mp
import math


def drawing(img, landmarks):
    for landmark in landmarks:
        x, y, _ = landmark
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)


def detection(img_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    landmarks = []
    if results.pose_landmarks:
        landmarks = [(lm.x * img.shape[1], lm.y * img.shape[0], lm.z * img.shape[1]) for lm in results.pose_landmarks.landmark]
        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]]
        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]]
        lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]]
        relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]]
        rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]]
        lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]]

        drawing(img, lshoulder)
        drawing(img, rshoulder)
        drawing(img, lelbow)
        drawing(img, relbow)
        drawing(img, rwrist)
        drawing(img, lwrist)
    return img, landmarks


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
    lea = calculate_angle(landmarks[0], landmarks[1], landmarks[2])
    rea = calculate_angle(landmarks[3], landmarks[4], landmarks[5])
    lsa = calculate_angle(landmarks[2], landmarks[6], landmarks[8])
    rsa = calculate_angle(landmarks[12], landmarks[6], landmarks[4])

    if (165 < lea <= 180) and (165 < rea <= 180) and (75 < rsa <= 90) and (75 < lsa <= 90):
        t_pose = True

    return t_pose


def process_img(img_path):
    output_img, landmarks = detection(img_path)

    if landmarks:
        t_pose = classify_pose(landmarks)

        if t_pose:
            result_text = "t-Pose: True"
        else:
            result_text = "t-Pose: False"
        cv2.putText(output_img, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Classified Image", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #image_path = "path/to/your_T-pose_image.jpg"
    img_path = r"C:\Users\pranathi\Pictures\Saved Pictures\download.jpg"
    process_img(img_path)
