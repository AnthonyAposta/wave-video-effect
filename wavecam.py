import cv2 
from time  import sleep 
import numpy as np
from numba import jit
from PIL import Image,ImageDraw, ImageFilter


def wave(frames, L, A, masks):

    new_frame = frames.pop(0)

    for i in range(len(frames)):

        new_frame = Image.composite(framelist[i], new_frame, masks[i])

    return new_frame, frames

cap = cv2.VideoCapture(2)
ret, frame = cap.read()
scale_percent = 40
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

frame = cv2.resize(frame, dim)

L = frame.shape[1]
A = frame.shape[0]
step = 2
blur = 2
N = int(L/step/2)

masks = []
for i in range(N):
    
    #R = ((A/2 - i*N)**2)/(A/2)
    R = (L/2 - i*step)
    print(R)
    mask = Image.new("L", (L,A), 0)
    draw = ImageDraw.Draw(mask)

    draw.ellipse((L/2-R, A/2-R, L/2+R, A/2+R), fill=255)
    mask_blur = mask.filter(ImageFilter.GaussianBlur(blur))
    masks.append(mask_blur)


#cap = cv2.VideoCapture(0)


framelist = []

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, dim)
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    
    if len(framelist) < N:
        framelist.append(Image.fromarray(frame))
    else:

        new_frame, framelist = wave(framelist, L, A, masks)
        cv2.imshow('frame', np.array(new_frame))
        framelist.append(Image.fromarray(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
