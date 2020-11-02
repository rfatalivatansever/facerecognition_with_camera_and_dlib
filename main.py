import cv2
import dlib

cam     = cv2.VideoCapture(0)
detect  = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1,p2) :
    return  int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)


while True:
    ret, frame = cam.read()
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect(grayScale)

    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        landmarks = predict(grayScale, face)

        # left eye
        left_eye_points_1 = (landmarks.part(36).x, landmarks.part(36).y)
        left_eye_points_2 = (landmarks.part(39).x, landmarks.part(39).y)
        left_eye_center_top = midpoint(landmarks.part(37), landmarks.part(38))
        left_eye_center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        h_line_1 = cv2.line(frame, left_eye_points_1, left_eye_points_2, (0, 255, 255), 1)
        v_line_1 = cv2.line(frame, left_eye_center_top, left_eye_center_bottom, (0, 0, 255), 1)

        # right_eye
        right_eye_points_1 = (landmarks.part(42).x, landmarks.part(42).y)
        right_eye_points_2 = (landmarks.part(45).x, landmarks.part(45).y)
        right_eye_center_top = midpoint(landmarks.part(43), landmarks.part(44))
        right_eye_center_bottom = midpoint(landmarks.part(47), landmarks.part(46))

        h_line_2 = cv2.line(frame, right_eye_points_1, right_eye_points_2, (0, 255, 255), 1)
        v_line_2 = cv2.line(frame, right_eye_center_top, right_eye_center_bottom, (0, 0, 255), 1)

        # nose
        nose_1 = (landmarks.part(27).x, landmarks.part(27).y)
        nose_2 = (landmarks.part(30).x, landmarks.part(30).y)

        h_line_3 = cv2.line(frame, nose_1, nose_2, (255, 255, 0), 2)

        # mouth
        mouth_points_1 = (landmarks.part(48).x, landmarks.part(48).y)
        mouth_points_2 = (landmarks.part(54).x, landmarks.part(54).y)

        h_line = cv2.line(frame, mouth_points_1, mouth_points_2, (0, 255, 255), 1)

        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            cv2.imshow("Face", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
