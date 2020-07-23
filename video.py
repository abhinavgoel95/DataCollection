import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import math

class InputVideo:
    def __init__(self, height, width, FPS, length, video_name):
        self.height = height
        self.width = width
        self.FPS = FPS
        self.length = length
        self.video_name = video_name

    def _create_image(self):
        rgb = np.random.randint(255, size=(self.height,self.width, 3),dtype=np.uint8)
        return rgb

    def create_video(self):
        video = cv2.VideoWriter(self.video_name, 0, 1, (self.width, self.height))
        frame_array = []
        for frame_number in range(math.ceil(self.FPS * self.length)):  
            #reading each files
            img = self._create_image()
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)

        out = cv2.VideoWriter(self.video_name, cv2.VideoWriter_fourcc(*'DIVX'), self.FPS, size)

        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        return out


def test():
    vid = InputVideo(32,32,24,5)
    vid.create_video()
    cap = cv2.VideoCapture('generated_video.avi')

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            print("hi")
        else: 
            break

    cap.release()