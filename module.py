import cv2


class Module:
    def __init__(self, id, is_vibrating,diameter):
        self.id = id
        self.is_vibrating = is_vibrating
        self.diameter = diameter


x = Module(1,True,2)

print(x.diameter)

cap = cv2.VideoCapture(0)