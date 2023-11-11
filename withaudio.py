import cv2
import mediapipe as mp
import math
import pygame
import os
pygame.mixer.init()

ideal_angles = {
    "lea": 180.0,
    "rea": 180.0,
    "lsa": 90.0,
    "rsa": 90.0,
    "bal": 90.0,
    "sar": 180.0,
    "bar": 180.0,
    "sal": 90.0,
}

def drawing(img, landmarks):
    # Extract only (x, y) coordinates from landmarks
    landmarks_xy = [(int(x), int(y)) for x, y, z in landmarks]

    # Draw a line using the modified landmarks
    cv2.line(img, landmarks_xy[12], landmarks_xy[11], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[23], landmarks_xy[11], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[24], landmarks_xy[23], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[12], landmarks_xy[24], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[26], landmarks_xy[24], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[26], landmarks_xy[28], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[23], landmarks_xy[25], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[25], landmarks_xy[27], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[12], landmarks_xy[14], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[14], landmarks_xy[16], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[12], landmarks_xy[24], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[11], landmarks_xy[13], (0, 225, 0), 2)
    cv2.line(img, landmarks_xy[13], landmarks_xy[15], (0, 225, 0), 2)



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
    march_pose = False
    lea = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    rea = calculate_angle(landmarks[16], landmarks[14], landmarks[12])
    lsa = calculate_angle(landmarks[13], landmarks[11], landmarks[23])
    rsa = calculate_angle(landmarks[24], landmarks[12], landmarks[14])
    bal = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
    sar = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
    bar = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    sal = calculate_angle(landmarks[11], landmarks[23], landmarks[25])


    if (150 < lea <= 220) and (150 < rea <= 220) and (75 < rsa <= 115) and (75 < lsa <= 115):
        t_pose = True
    if ((75<=bal<=110) and (165<=sar<=195))  or ((75<=bar<=110) and (165<=sal<=195)):
        march_pose = True
    if(t_pose):
        pose_r = 1
    elif(march_pose):
        pose_r = 2
    else:
        pose_r = 0

    return pose_r

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

            pose_r = classify_pose(landmarks)

            if (pose_r==1):
                result_text = "Exercise:Lateral Raise"
                lea = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                rea = calculate_angle(landmarks[16], landmarks[14], landmarks[12])
                rsa = calculate_angle(landmarks[24], landmarks[12], landmarks[14])
                lsa = calculate_angle(landmarks[13], landmarks[11], landmarks[23])

                total_error = abs(ideal_angles["lea"] - lea) + abs(ideal_angles["rea"] - rea) + abs(
                    ideal_angles["rsa"] - rsa) + abs(ideal_angles["lsa"] - lsa)
                lateral_raise_sound_path = r"C:\Users\Snehas\Desktop\lateral raise.mp3"

                pygame.mixer.music.load(lateral_raise_sound_path)
                pygame.mixer.music.play()
            elif(pose_r==2):

                result_text = "Exercise:Marching"
                bal = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                sar = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
                bar = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                sal = calculate_angle(landmarks[11], landmarks[23], landmarks[25])


                total_error = abs(ideal_angles["bal"] - bal) + abs(ideal_angles["sar"] - sar) + abs(
                    ideal_angles["bar"] - bar) + abs(ideal_angles["sal"] - sal)
                marching_sound_path = r"C:\Users\Snehas\Desktop\marching.mp3"
                pygame.mixer.music.load(marching_sound_path)
                pygame.mixer.music.play()
            else:
                result_text = "Pose: Unknown"
                total_error= 0
            cv2.putText(frame, result_text, (10,16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Error: {total_error:.2f}", (10,461), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            drawing(frame, landmarks)

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video stream
            break

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    process_video()
