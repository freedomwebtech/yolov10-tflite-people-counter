import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from tracker import Tracker

model = YOLO("best_float32.tflite")  
 



cap=cv2.VideoCapture('pcount.mp4')
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker=Tracker()
count=0

cy1=340
cy2=372
offset=11
goingup={}
personup=[]
goingdown={}
persondown=[]
while True:
    ret,frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
       break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame,imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        list.append([x1,y1,x2,y2,])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
        if cy1<(cy+offset) and cy1>(cy-offset):
           goingup[id]=(cx,cy)
        if id in goingup:
            if cy2<(cy+offset) and cy2>(cy-offset):
               cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
               cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
               if personup.count(id)==0:
                  personup.append(id)
##########################################################################
        if cy2<(cy+offset) and cy2>(cy-offset):
           goingdown[id]=(cx,cy)
        if id in goingdown:
            if cy1<(cy+offset) and cy1>(cy-offset):
               cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
               cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,0),2)
               if persondown.count(id)==0:
                  persondown.append(id)          
                  
    cv2.line(frame,(1,340),(1019,340),(0,0,255),2)
    cv2.line(frame,(1,372),(1019,372),(255,0,255),2)
    pup=len(personup)
    pdown=len(persondown)
    cvzone.putTextRect(frame,f'pup:-{pup}',(50,60),1,1)
    cvzone.putTextRect(frame,f'pdown:-{pdown}',(50,160),1,1)


    cv2.imshow("frame", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()