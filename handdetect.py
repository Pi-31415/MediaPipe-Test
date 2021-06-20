import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Module Class
class Module:
    def __init__(self, id, is_vibrating,radius,x,y):
        self.id = id
        self.is_vibrating = is_vibrating
        self.radius = radius
        self.x = x
        self.y = y
        self.tolerance = 50

    def show(self,IMG,finger_x,finger_y):
        if(int(abs(self.x-finger_x)) <= self.tolerance and int(abs(self.y-finger_y)) <= self.tolerance):
            cv2.circle(IMG, (self.x,self.y), self.radius-2, (0, 0, 255), -1)
            cv2.circle(IMG, (self.x,self.y), self.radius, (255, 255, 255), 3) 
        else:
            cv2.circle(IMG, (self.x,self.y), self.radius, (255, 255, 255), 3)
        




# For webcam input:
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

with mp_hands.Hands(
        max_num_hands=2,
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
                x_coordinate = int(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                y_coordinate = int(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight)
                center_coordinates = (x_coordinate, y_coordinate)

                # Radius of circle
                radius = 20

                # Blue color in BGR
                color = (0, 0, 255)

                # Line thickness of 2 px
                thickness = 2
                # Display finger Position
                # image = cv2.circle(image, center_coordinates,radius, color, thickness)

                # Display Coordinates
                text = "Fingertip : ("+str(x_coordinate)+" , "+str(y_coordinate)+")"
                coordinates = (30, 50)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0, 0, 0)
                thickness = 2
                # image = cv2.putText(image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
                '''
                print(
                    f'Index finger tip coordinate: (',
                    f'{x_coordinate}, '
                    f'{y_coordinate})'
                )
                '''

                module = []

                for x in range(9):
                    index = x+1
                    module.append(Module(index,True,30,index*100,200))
                    module[x].show(image,x_coordinate,y_coordinate)
                
        cv2.imshow('Multiple Point Detection Test', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
