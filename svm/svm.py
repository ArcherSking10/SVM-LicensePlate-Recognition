import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import time
import random
import scipy
import cv2

from PIL import Image
from sklearn import datasets, linear_model, preprocessing
from sklearn import svm
from sklearn.externals import joblib
from scipy import signal

X_LENGTH = 1280
SVCs = {}
TRAINED = 0
CLASS_NUM = {'letter':24, 'province':6, 'char':34}
POSITION = ['province','letter','char','char','char','char','char']

#-----------------------------------------------Scan Folders-----------------------------------------------
def getFilesName(nclass, TYPE, s):
    
    input_count = 0
    filenames = []
    types = []
    for i in range(nclass):
        dir = './%s/%s/%s/' % (s,TYPE,i)
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1
                filenames.append(dir+filename)
                types.append(i)
    return input_count,filenames, types

#----------------------------------------------Read Single Image---------------------------------------------
def getImage(fn):
    global X_LENGTH
    img = Image.open(fn)
    v = np.zeros((1, X_LENGTH))
    width = img.size[0]
    height = img.size[1]
    for h in range(0, height):
        for w in range(0, width):
            #Threshold into binary value
            if img.getpixel((w, h)) < 127:
                v[0,w+h*width] = 0
            else:
                v[0,w+h*width] = 1
    return v

#---------------------------------------------Training Single Word---------------------------------------------
def training(s):
    
    if os.path.exists("./%s.m"%s):
        svc = joblib.load("%s.m"%s)
        print("Loaded previous model.")
        return svc
    
    global X_LENGTH, CLASS_NUM
    input_count, filenames, types= getFilesName(CLASS_NUM[s],s,"train")
    X = np.zeros((input_count, X_LENGTH))
    y = np.zeros(input_count)

    for i in range(input_count):
        filename = filenames[i]
        X[i,:] = getImage(filename)
        y[i] = types[i]

    svc = svm.SVC(probability=True,  kernel="rbf", C=2.8, gamma=.0073,verbose=10)
    svc.fit(X, y)
    y_hat = svc.predict(X)
    acc = np.mean(y_hat == y)
    print("\n\nTraining Accuracy for %s: %.2f\n"%(s, acc))
    joblib.dump(svc, "%s.m"%s)
    return svc

#---------------------------------------------Testing Single Word---------------------------------------------
def testing(s):
    
    global X_LENGTH, CLASS_NUM
    input_count, filenames, types = getFilesName(CLASS_NUM[s],s,"test")
    Xtest = np.zeros((input_count, X_LENGTH))
    ytest = np.zeros(input_count)
    
    for i in range(input_count):
        filename = filenames[i]
        Xtest[i,:] = getImage(filename)
        ytest[i] = types[i]
 
    ytest_hat = SVCs[s].predict(Xtest)
    print("\nTesting Accuracy for %s: %.2f\n"%(s, np.mean(ytest_hat == ytest)))

#----------------------------------------------ClassNum to Char------------------------------------------------
def getChar(x, s):
    if s == "province":
        return chr(x+97)
    elif s == "letter":
        if x <= 7:
            return chr(x+65)
        elif x <= 12:
            return chr(x+66)
        else:
            return chr(x+67)
    else:
        if (x >= 10):
            return getChar(x-10, "letter")
        else:
            return chr(x+48)


#----------------------------------------------Predicting Single Word---------------------------------------------
def predict(X, s): #X is 1x1280
    global SVCs
    y = SVCs[s].predict(X)
    return getChar(int(y[0]), s)

#------------------------------------------------Predicting A Plate----------------------------------------------
def detect(s):
 
    global POSITION
    
    #Binary-valuing the plate:
    img_gray = cv2.imread("./%s"%s,0)
    #Reduce the noise:
    img_gray = cv2.GaussianBlur(img_gray,(3,3),0.1)

    img_thre = img_gray
    mid = (img_gray.min()+img_gray.max())/2
    cv2.threshold(img_gray, mid, 255, cv2.THRESH_BINARY_INV, img_thre)
    img = img_thre
    
    h = img.shape[0]
    w = img.shape[1]

    #find blank columns:
    white_num = []
    white_max = 0
    for i in range(w):
        white = 0
        for j in range(h):
            if img[j,i] == 255:
                white += 1
        white_num.append(white)
        white_max = max(white_max, white)
    blank = []
    for i in range(w):
        if (white_num[i]  > 0.95 * white_max):
            blank.append(True)
        else:
            blank.append(False)

    #split index:
    i = 0
    num = 0
    l = 0
    x,y,d = [],[],[]
    while (i < w):
        if blank[i]:
            i += 1
        else:
            j = i
            while (j<w)and(  (not blank[j])or
                             (j-i<3)  ):
                j += 1
            x.append(i)
            y.append(j)
            d.append(j-i)
            l += 1
            i = j
    d = np.array(d)
    while (l > 7):
        i = np.argmin(d)
        l1 = d[i-1] if i>0 else 100
        l2 = d[i+1] if i<l-1 else 100
        if l1 > l2:
            x[i+1] = x[i]
        else:
            if (i-1>=0)and(i<l):
                y[i-1] = y[i]
        if (i>=0)and(i<l):
            x.pop(i)
        if (i>=0)and(i<l):
            y.pop(i)
        np.delete(d,[i])
        l -= 1

    #predict plate:
    stri = ""
    Xtest = np.zeros((1,X_LENGTH))
    img = Image.fromarray(255-img)
    width = img.size[0]
    height = img.size[1]
    for i in range(l):
        sub_img = img.crop((x[i],0,y[i]-1,40))
        w0 = sub_img.size[0]
        h0 = sub_img.size[1]
        if w0 < 32:
            bg = Image.new("L",(32,40),0)
            for h in range(h0):
                for w in range(w0):
                    bg.putpixel((int(w+(32-w0)/2),h),sub_img.getpixel((w,h)))
            sub_img = bg
        sub_img = sub_img.resize((32,40),Image.ANTIALIAS)
        #sub_img.save("%d.png"%i,"png")
        w0 = sub_img.size[0]
        h0 = sub_img.size[1]
        for h in range(0, h0):
            for w in range(0, w0):
                if sub_img.getpixel((w, h)) < mid:
                    Xtest[0,w+h*w0] = 0
                else:
                    Xtest[0,w+h*w0] = 1
        stri += predict(Xtest, POSITION[i])

    return stri

#--------------------------------------------Predicting All Plates---------------------------------------------
def test_plate():
    
    input_count = 0
    acc = 0
    filenames = []
    pred = []
    dir = './test_plate/'
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            if filename[-4:] != ".png":
                continue
            input_count += 1
            ans = detect(dir+filename)
            print(filename+" "+ans)
            filenames.append(filename)
            pred.append(ans)
            if ans == filename[9:16]:
                acc += 1

    print("\nTesting Accuracy for plates: %.2f"%(acc/input_count))

#-----------------------------------------------------Menu------------------------------------------------------
def train_menu():
    global SVCs
    global TRAINED
    a = input(" _________________________ \n"+
              "|_1_|_train_char_data_____|\n"+
              "|_2_|_train_letter_data___|\n"+
              "|_3_|_train_province_data_|\n"+
              "|_4_|_train_all_data______|\n"+
              "\nEnter:")
    if not(a in ["1","2","3","4"]):
        print("Illegal Input!")
        return
    
    a, b = int(a)-1, [1,2,4,7]
    if b[a]&1 != 0:
        SVCs["char"] = training("char")
        TRAINED |= 1
    if b[a]&2 != 0:
        SVCs["letter"] = training("letter")
        TRAINED |= 2
    if b[a]&4 != 0:
        SVCs["province"] = training("province")
        TRAINED |= 4


def testsingle_menu():
    global SVCs
    global TRAINED
    a = input(" ________________________ \n"+
              "|_1_|_test_char_data_____|\n"+
              "|_2_|_test_letter_data___|\n"+
              "|_3_|_test_province_data_|\n"+
              "|_4_|_test_all_data______|\n"+
              "\nEnter:")
    if not(a in ["1","2","3","4"]):
        print("Illegal Input!")
        return
    
    a, b = int(a)-1, [1,2,4,7]
    if TRAINED&b[a] != b[a]:
        print("Haven't train data!")
        return
    if b[a]&1 != 0:
        testing("char")
    if b[a]&2 != 0:
        testing("letter")
    if b[a]&4 != 0:
        testing("province")


#-----------------------------------------------------Main------------------------------------------------------
a = ""
while (a != "5"):
    a = input(" _________________________________ \n"+
              "|_1_|_train_data__________________|\n"+
              "|_2_|_test_single_character_datas_|\n"+
              "|_3_|_test_plate_datas____________|\n"+
              "|_4_|_show________________________|\n"+
              "|_5_|_exit________________________|\n"+
              "\nEnter:")
    if   a == "1":
        train_menu()
    elif a == "2":
        testsingle_menu()
    elif a == "3":
        if TRAINED != 7:
            print("Haven't train data!")
            continue
        test_plate()
    elif a == "4":
        if TRAINED != 7:
            print("Haven't train data!")
            continue
        s = input("Enter picture name:")
        ans = detect(s)
        print(ans)
