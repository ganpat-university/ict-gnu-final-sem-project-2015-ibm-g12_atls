import numpy as np
import cv2
from datetime import datetime as dt
import boto3
import time

thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
cap = cv2.VideoCapture('Road_traffic_video2.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,280) #width 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120) #height 
cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness 


#cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)

classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()
#print(classNames)

font = cv2.FONT_HERSHEY_PLAIN
#font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    if len(classIds) != 0:
        data_list=[]
        file=open("Datavalue.txt","w+")
        for i in indices:
            #i = i[0]
            box = bbox[i]
            confidence = str(round(confs[i],2))
            #print(classIds[i])
            color = Colors[classIds[i]-1]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
            cv2.putText(img, classNames[classIds[i]-1]+" "+confidence,(x+10,y+20),
                        font,1,color,2)
#             cv2.putText(img,str(round(confidence,2)),(box[0]+100,box[1]+30),
#                         font,1,colors[classId-1],2)
            
            data_dict={"d_id":str(dt.now()),"Vehicles Detected":classNames[classIds[i]-1],"Accuracy":confidence}
            db = boto3.resource('dynamodb')
            print(data_dict)
            table = db.Table('data_atls') # give here your dynamo db table's name
            table.put_item(Item=data_dict)
            time.sleep(2)
            print(classNames[classIds[i]-1]+" "+confidence,dt.now())
            data_list.append({"Date Time":dt.now(),"Vehicles Detected":classIds,"Accuracy":confidence})
            
        
        
            
    cv2.imshow("Output",img)
    cv2.waitKey(1)