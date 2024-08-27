import cv2
import numpy as np
import os

weightspath = r"C:\Users\HP\Desktop\yolo\yolov3.weights"
cfgpath = r"C:\Users\HP\Desktop\yolo\yolov3.cfg"
namespath = r"C:\Users\HP\Desktop\yolo\coco.names"

yolo = cv2.dnn.readNet(weightspath, cfgpath)

classes = []
with open(namespath, "r") as f:
    for line in f:
        classes.append(line.strip())

def processingofyolo(frame):
    height, width = frame.shape[:2]
    imgresized = cv2.resize(frame, (750, 750))
    rgbimg = cv2.cvtColor(imgresized, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(rgbimg, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo.setInput(blob)
    outputlayers = yolo.getUnconnectedOutLayersNames()
    layersoutput = yolo.forward(outputlayers)
    
    boxes = []
    confidences = []
    classids = []

    for output in layersoutput:
        for detection in output:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > 0.7:
                centerx = int(detection[0] * width)
                centery = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x1 = int(centerx - w / 2)
                y1 = int(centery - h / 2)
                x2 = int(centerx + w / 2)
                y2 = int(centery + h / 2)
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                classids.append(classid)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    
    for i in indices:
        box = boxes[i[0]]
        x1, y1, x2, y2 = box
        label = classes[classids[i[0]]] + ": " + str(int(confidences[i[0]] * 100)) + "%"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 5)

    return frame

userchoice = input(f"Do you want to use the camera (c) or upload an image (i)? Enter 'c' or 'i': ")

if userchoice == 'c':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processedframe = processingofyolo(frame)
        
        cv2.imshow('YOLO Object Detection', processedframe)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
elif userchoice == 'i':
    imgpath = input(f"Please enter the full path to the image: ")
    if os.path.exists(imgpath):
        img = cv2.imread(imgpath)
        processedimg = processingofyolo(img)
        
        cv2.imshow('YOLO Object Detection', processedimg)
        cv2.waitKey(0)
    else:
        print("Wont Work . Experiencing Error")

cv2.destroyAllWindows()
