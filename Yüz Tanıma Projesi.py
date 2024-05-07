import cv2
import matplotlib.pyplot as plt

#ice aktar eindtein
einstein = cv2.imread("einstein.jpg",0)
plt.figure(), plt.imshow(einstein,cmap= "gray"), plt.axis("off")

#sınıflandırıcı yüz olup olmadığı
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y), (x+w, y+h),(255,255,255),10)

plt.figure(), plt.imshow(einstein,cmap= "gray"), plt.axis("off")























