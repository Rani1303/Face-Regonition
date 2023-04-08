import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Combobox
import os
import cv2
import sys
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PIL import ImageTk, Image
import cv2
import numpy as np
import pickle


root=Tk()
root.title("Facial Detection!")
root.geometry("450x700")
root.resizable(False,False)
root.configure(bg="#305065")

#icon
#image_icon=PhotoImage(file="")
#root.iconphoto(False,image_icon) 

def callback():
    face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
    # eye_cascade=cv2.CascadeClassifier('data/haarcascade_eye.xml')

    recognizer=cv2.face.LBPHFaceRecognizer_create()
    labels=recognizer.read("trainner.yml")

    labels={"person_name": 1}
    with open("labels.pickle",'rb') as f:
        og_labels=pickle.load(f)
        labels={v:k for k,v in og_labels.items()}


    cap=cv2.VideoCapture(0)

    while(True):
        ret,frame=cap.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        for(x,y,w,h) in faces:
            # print(x,y,w,h)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]
            id_,conf=recognizer.predict(roi_gray)

            if conf>=45 and conf<=85:
                print(id_)
                print(labels[id_])
                font=cv2.FONT_HERSHEY_COMPLEX
                name=labels[id_]
                color=(255,255,255)
                stroke=2
                cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

            img_item="7.png"
            cv2.imwrite(img_item,roi_color)

            color=(255,0,0)
            stroke=2
            end_cord_x=x+w
            end_cord_y=y+h
            cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
            # for (ex,ey,eh,ew) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#Top Frame
Top_frame=Frame(root,bg="white",width=900,height=100)
Top_frame.place(x=0,y=0)

Logo=PhotoImage(file="dsf.jpg")
Label(Top_frame,image=Logo,bg="white").place(x=10,y=5)
Label(Top_frame,text="Facial Recognition",font='arial 20 bold',bg='white',fg="black").place(x=100,y=30)
frame=Frame(root,width=440,height=500)
frame.pack()
frame.place(anchor='center',relx=0.5,rely=0.5)
img=ImageTk.PhotoImage(Image.open("ping.jpg"))
label=Label(frame,image=img)
label.pack()
btn = Button(root, text = 'Start Recognizing! ',height=5,width=15,font='sans 16 bold',
                command = callback)
 
# Set the position of button on the top of window
btn.place(x=125,y=500)    
root.mainloop()