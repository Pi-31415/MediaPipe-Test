import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Location of the Modules, and Jacket Outline
modules_x = [400,500,600,700,800,350,500,600,700,850,300,500,600,700,900,250,500,600,700,950]
modules_y = [200,200,200,200,200,300,300,300,300,300,400,400,400,400,400,500,500,500,500,500]
jacket_shape_start = [(150,550),(1050,550),(750,250),(450,250),(150,550),(900,550),(450,250),(750,250),(450,550),(350,150)]
jacket_shape_end = [(350,150),(850,150),(900,550),(300,550),(300,550),(1050,550),(450,550),(750,550),(750,550),(850,150)]

# Module Class
class Module:
    def __init__(self, id, is_vibrating, radius, x, y):
        self.id = id
        self.is_vibrating = is_vibrating
        self.radius = radius
        self.x = x
        self.y = y
        self.tolerance = 50

    def show(self, IMG, finger_x, finger_y): #Visualizes the Haptic Module on Camera
        if(int(abs(self.x-finger_x)) <= self.tolerance and int(abs(self.y-finger_y)) <= self.tolerance):
            cv2.circle(IMG, (self.x, self.y), self.radius-2,
                       (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(IMG, (self.x, self.y), self.radius,
                       (255, 255, 255), 3, lineType=cv2.LINE_AA)
        else:
            cv2.circle(IMG, (self.x, self.y), self.radius,
                       (255, 255, 255), 3, lineType=cv2.LINE_AA)


# For webcam input:
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        #For opacity
        overlay = image.copy()
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
                radius = 10

                # Blue color in BGR
                color = (0, 0, 255)

                # Line thickness of 2 px
                thickness = -1
                # Display finger Position
                image = cv2.circle(image, center_coordinates,radius, color, thickness, cv2.LINE_AA)

                # Display Coordinates for single finger
                # text = "Fingertip : ("+str(x_coordinate) + " , "+str(y_coordinate)+")"
                text = "Finger Detected"
                coordinates = (30, 50)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0, 0, 0)
                thickness = 2
                image = cv2.putText(image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
                '''
                print(
                    f'Index finger tip coordinate: (',
                    f'{x_coordinate}, '
                    f'{y_coordinate})'
                )
                '''

                #Render Haptic Modules
                module = []

                for x in range(len(modules_x)):
                    index = x+1
                    module.append(Module(index, True, 30, modules_x[x], modules_y[x]))
                    module[x].show(image, x_coordinate, y_coordinate)
                
                #Draw Lines
                for x in range(len(jacket_shape_start)):
                    image = cv2.line(image, jacket_shape_start[x], jacket_shape_end[x], color, thickness,cv2.LINE_AA)

        else:
            text = "No finger Detected"
            coordinates = (30, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 0)
            thickness = 2
            image = cv2.putText(image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

       
        cv2.imshow('Multiple Point Detection Test', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
