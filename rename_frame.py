import os
import cv2
import numpy as numpy
import sys
import glob

flist = glob.glob("**/*.png", recursive = True)
index = 0
path=os.getcwd()

for x in flist:
    oldname = x
    newname=path+'\\test_%03d' % (index)+'.png'
    os.rename(oldname,newname)
    print(oldname,'=>',newname)
    index+=1