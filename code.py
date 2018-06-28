
# coding: utf-8

# In[ ]:


import cv2
import glob
import os
import time
import imutils
import argparse
from imutils.object_detection import non_max_suppression

subject_label = 1
font = cv2.FONT_HERSHEY_SIMPLEX
list_of_videos = []
cascade_path = "face_cascades/haarcascade_profileface.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
recognizer = cv2.face.LBPHFaceRecognizer_create()
count=0


# In[ ]:


face_cascade = cv2.CascadeClassifier('/home/spartan/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')


# In[ ]:


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h),
                      (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)

def find_people(img):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        img = frame
        if img is None:
            return None
        #  print('Failed to load image file:', fn)
        #  continue
        # except:
        #  print('loading error')
        #  continue

        found, w = hog.detectMultiScale(
            img, winStride=(10, 10), padding=(32, 32), scale=1.05)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        found_filtered = []
        draw_detections(img,faces)
        draw_detections(img,found)
        
        cv2.putText(img,('%d body found %d faces found' % (len(found), len(faces))),(150,150), cv2.FONT_HERSHEY_SIMPLEX,1, (200,255,155), 2, cv2.LINE_AA)
        return img


# In[ ]:



import math

videoFile = "faces.mp4"
vidcap = cv2.VideoCapture(videoFile)
success,image = vidcap.read()



seconds = 1
fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
print(fps)
multiplier = round(fps * seconds)
print(multiplier)


while success:
    frameId = int(round(vidcap.get(1))) 
    success,image = vidcap.read()
    print(success)
    print(frameId)
    if frameId % multiplier == 0:
        print('yeah')
        frame=image
        imag=find_people(frame)
        cv2.imshow('ip',imag)
        cv2.waitKey(30)
        cv2.imwrite("/home/spartan/nn/frame%d.jpg" % frameId, imag)

vidcap.release()
print ("Complete")

