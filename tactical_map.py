import cv2
import numpy as np

class TacticalMap:
    def __init__(self):
        self.width = 300
        self.height = 500

    def draw_pitch(self):
        pitch = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pitch[:] = (30, 120, 30)
        cv2.rectangle(pitch, (10, 10), (290, 490), (255,255,255), 2)
        return pitch

    def draw_entity(self, pitch, x, y, color):
        cv2.circle(pitch, (x, y), 5, color, -1)
