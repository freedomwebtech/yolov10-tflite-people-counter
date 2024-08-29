import cv2
import time
cpt = 0
maxFrames = 240 # if you want 5 frames only.

cap=cv2.VideoCapture('pcount.mov')

while cpt < maxFrames:
    ret, frame = cap.read()
    if not ret:
       break
    frame=cv2.resize(frame,(1020,600))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"C:\Users\freed\Downloads\yolov10.tflite-main\yolov10.tflite-main\imges\img_%d.jpg" %cpt, frame)
    cpt += 1
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
