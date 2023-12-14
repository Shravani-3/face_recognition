import cv2
import numpy as np
import face_recognition

imghemant=face_recognition.load_image_file("images face/Shravani.jpg")
imghemant=cv2.cvtColor(imghemant,cv2.COLOR_BGR2RGB)

imgTest=face_recognition.load_image_file("images face/hemant.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)



faceLoc=face_recognition.face_locations(imghemant)[0]
encodehemant=face_recognition.face_encodings(imghemant)[0]
# cv2.rectangle(imghemant,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLoctest=face_recognition.face_locations(imgTest)[0]
encodetest=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


results=face_recognition.compare_faces([encodehemant],encodetest)
faceDis=face_recognition.face_distance([encodehemant],encodetest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Rohit",imghemant)
cv2.imshow("hemant",imgTest)
cv2.waitKey(0)
