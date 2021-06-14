import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_hight, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            # Uncomment to see whole hand
            # for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for hand_landmarks in results.multi_hand_landmarks:
                # Print index finger tip coordinates.
                                
                # Center coordinates
                x_coordinate = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                y_coordinate = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight)
                center_coordinates = (x_coordinate,y_coordinate)
                
                # Radius of circle
                radius = 20
                
                # Blue color in BGR
                color = (0, 0, 255)
                
                # Line thickness of 2 px
                thickness = 2
                image = cv2.circle(image, center_coordinates, radius, color, thickness)

                # Display Coordinates
                text = "("+str(x_coordinate)+" , "+str(y_coordinate)+")"
                coordinates = (100,100)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0,0,255)
                thickness = 2
                image = cv2.putText(image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

                print(
                    f'Index finger tip coordinate: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                )
        
        cv2.imshow('Hand Position Test', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
