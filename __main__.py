import cv2
import mediapipe

drawingModule = mediapipe.solutions.drawing_utils
poseModule = mediapipe.solutions.pose

capture = cv2.VideoCapture(0)

with poseModule.Pose(static_image_mode=False) as pose:
    while (True):

        ret, frame = capture.read()
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.namedWindow("Test pose", cv2.WINDOW_NORMAL)


        if result.pose_landmarks:
            drawingModule.draw_landmarks(frame, result.pose_landmarks, poseModule.POSE_CONNECTIONS,
                            drawingModule.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            drawingModule.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            
            right_index = result.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_INDEX]
            right_index_coords = (int(right_index.x * frame.shape[1]), int(right_index.y * frame.shape[0]))
            print(f"Right Index Position: {right_index_coords}")

            # Draw a circle on the right index position
            cv2.circle(frame, right_index_coords, 20, (255, 0, 0), -1)


        cv2.imshow('Test pose', frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
capture.release()
