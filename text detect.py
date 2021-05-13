# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:44:40 2021

@author: JK
"""
import os
import pytesseract
from pathlib import Path
from matplotlib import pyplot as plt   ## library for plot
import cv2
import numpy as np

## Here you set the path of tesseract-ocr engine
pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'
print("Please provide the full path of the image file or name if in the same folder like img.png :")
path=str(input())
imgpath=Path(path)
img=cv2.imread(imgpath.as_posix())

## To plot fig flag make this one(TRUE)
plot_fig=1


def components(img):
    
    thresh = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    [m,n]=img.shape
    win=max(m,n)//50
    armax=win**2
    if armax<1000:
        armax=1000
    new_img=img.copy()
    mask = np.zeros(img.shape, dtype="uint8")
    for i in range(1,num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y=stats[i, cv2.CC_STAT_TOP] 
        w=stats[i, cv2.CC_STAT_WIDTH]
        h= stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        new_img=cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if area>70 and area<armax and (w>3 and w<50+win) and (h>5 and h<54+win):
            componentMask = (labels == i).astype("uint8") * 255
            mask=cv2.bitwise_or(mask, componentMask)
  
    return mask


def compare(text,data):
    text=list(text) 
    pmax=0.4
    out=" "
    for d in data:
        d=list(d)
        c=0;
        stack=text.copy()
        for ele in d:
            for t in stack:
                if ele==t:
                    stack.remove(t)
                    c+=1
                    break
        p=c/max(len(d),len(text))
        if p>pmax:
            pmax=p
            out=d
    return pmax,"".join(out)     
                
                
    

def word(text):
    temp1=0
    text=text.lower()
    pos={"bedroom":1,"hall":1,"sovrum":1,"sovrum1":1,"sovrum2":1,"sovrum3":1,"sovrum4":1,"sovrum5":1,"kladvard":1,"matplats/vardagsrum":1,"kok//vardagsrum":1,
         "vardagsrum":1,"kok":1,"dm":1,"klk":1,"tm":1,"uteplats":1,"matplats":1,"g/g/g":1,"v-rum/kok":1,"badr":1,"sov":1,"brostningshojd":1,"grovtatt":1,
         "inglasadbalkong" :1,"balkong":1,"klk1":1,"klk2":1,"hall":1,"entre":1,"allrum":1,"bad":1,"badrum":1,"badrum1":1,"badrum2":1,"badrum3":1,
         "garage":1,"skarmvgg":1,"ftv":1,"bad/wc":1,"rum":1,"tambur":1,"sallskapsrum":1,"balk":1,"vinkyl":1,"alkov":1,"kokso":1,"arbetsrum":1,"gastrum":1,
         "arbetsrum/gastrum":1,"walk-in-closet":1,"tvatt":1,"franskbalkong":1,"altan":1,"trapphus":1,"terras":1,"forrad":1,"matk":1,"arbrum":1,"u/m":1,
         "wc/d":1,"st":1,"wc":1,"wc/dusch":1,"ks":1,"tt":1,"ftx":1,"kf":1,"wc/":1,"k/f":1,"tm/tt":1,"vp":1,"clc":1,"ht":1,"f/sk":1,"k/sk":1,"kh":1,
         "ku":1,"hs":1,"kpr":1,"sop":1,"ggg":1,"pass":1,"kapprum":1 
             }

    try:
        temp1=pos[text]
    except:
        data=sorted(pos.keys())
        temp1=2
        [p,txt]=compare(text,data)
        if p>0.6:
            return 1,txt
        else:
            return 0,text
    if int(temp1)==1:
        return 1,text

    
## Preprocessing image for better results
def preprocess(img):
    img= cv2.resize(img, (0, 0), fx=2, fy=2)
    thresh = cv2. cvtColor (img, cv2.COLOR_BGR2GRAY)

    [m,n]=thresh.shape
    win=max(m,n)//45
    if win%2==0:
        win=win+1
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, win, 5)
    ## Global thresholding
    ret3,thresh = cv2.threshold(thresh,0,255,cv2.THRESH_OTSU)
    kernel=np.ones((2,2)).astype("uint8")

    thresh=components(thresh)
    ret3,thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY_INV)
    thresh=cv2.dilate(thresh,kernel,iterations =0)
    #cv2.imwrite("New1.jpeg",thresh)
    return thresh
        
def detect(img):  
    custom_config = r"--oem 3 --psm 11  -l swe+eng  -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxy0123456789/ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    q=pytesseract.image_to_data(img, config=custom_config,output_type='dict')
    left=q['left']
    top=q['top']
    width=q['width']
    height=q['height']
    text=q['text']
    center=[]

    for i in range(len(text)):
        if len(text[i])>1 and len(text[i])<20:
            [tr,ret_text]=word(text[i])
            if tr==1:
                center.append([ret_text,(left[i]//2+width[i]//4),(top[i]//2+height[i]//4)])
                
    return center
    
def mainL(img):
    m=img.shape[0]
    n=img.shape[1]
    if m*n<2500:
        print("Low Resolution")
    else:
        print("Detected center pixels")
        overlay=img.copy()
        img1=preprocess(img)
        center0=detect(img1)
        for c in center0:
            print(c[0],end=":")
            print("left:{}px top:{}px".format(c[1],c[2]))
            cv2.circle(overlay,(c[1],c[2]),35, (0, 255,0),-1)
            
        ## After rotation +90
        img90=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        img90=preprocess(img90)
        center90=detect(img90)
        for c in center90:
            print(c[0],end=":")
            print("left:{}px top:{}px".format(n-c[2],c[1]))
            cv2.circle(overlay,(n-c[2],c[1]),35, (0, 255,0),-1)
        
        ## After Rotation -90
        img270=cv2.rotate(img,cv2.cv2.ROTATE_90_CLOCKWISE)
        img270=preprocess(img270)
        center270=detect(img270)
        for c in center270:
            print(c[0],end=":")
            print("left:{}px top:{}px".format(c[2],m-c[1]))
            cv2.circle(overlay,(c[2],m-c[1]),35, (0, 255,0),-1)
        
        cv2.addWeighted(overlay, 0.5,img, 1 -0.5,0, img)
        if plot_fig==1:
            plt.figure()
            plt.imshow(img,cmap="gray")
            cv2.imwrite("new.jpeg",img)
  
mainL(img)
