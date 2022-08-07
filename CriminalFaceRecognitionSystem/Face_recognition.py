#import all the necessary packages

import face_recognition
import cv2
import numpy as np 
import os
from datetime import datetime

#add the images path
path='images'

#create a list for all the images
images = []

#write the names of all the images
DisplayNames = []

# to grab the list of images from the folder
myList =  os.listdir(path)

#print the list of images
print(myList)

for imgs in myList:
    img = cv2.imread(f'{path}/{imgs}')
    images.append(img)
    DisplayNames.append(os.path.splitext(imgs)[0])

print(DisplayNames)

#create a def for performing the encoding of the images
def findEncodings (images):
    encodeList = [ ] 
    for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            encode = face_recognition.face_encodings (img)[0]
            encodeList.append(encode) 
    return encodeList

def marktime(name):
    with open(r'C:\opencv\source-code-face-recognition\source code\Time.csv', 'r+') as f:
         myDataList = f.readlines() 
         nameList = []
         for line in myDataList:

             entry =line.split(',') 
             nameList.append(entry[0])

         if name not in nameList: 
            now = datetime.now()

            dtString =now.strftime('%H:%M:%S') 
            f.writelines(f'\n{name}, {dtString}')



encodeListKnown = findEncodings(images)
print("Images have been Encoded")

#capture from camera
video_capture = cv2.VideoCapture(0)

while True:
    ret,frame=video_capture.read()
    imgSmall = cv2.resize(frame,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodeFrame = face_recognition.face_encodings (imgSmall,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        face_names = []
               
        if faceDis!=encodeListKnown:
            name="UNKNOWN"
            y1,x2,y2, x1 = faceLoc 
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1) 
            cv2.rectangle(frame,(x1,y2-35), (x2,y2),(0,0,255), cv2.FILLED) 
            cv2.putText(frame, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255))

        if matches[matchIndex]:
            name=DisplayNames[matchIndex].upper()
            if name=="CRIMINAL":
                print(name)

                y1,x2,y2, x1 = faceLoc 
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1) 
                cv2.rectangle(frame,(x1,y2-35), (x2,y2),(0,0,255), cv2.FILLED) 
                cv2.putText(frame, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255))
                marktime('CRIMINAL')

            else:
                    print(name)
                    y1,x2,y2, x1 = faceLoc 
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1) 
                    cv2.rectangle(frame,(x1,y2-35), (x2,y2),(0,255,0), cv2.FILLED) 
                    cv2.putText(frame, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255))
                    marktime(name)
                    


    cv2.imshow("Video",frame)
    wait_key=cv2.waitKey(1)
    if wait_key==27:
        break


video_capture.release()
cv2.destroyAllWindows()