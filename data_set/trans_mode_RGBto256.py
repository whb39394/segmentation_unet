# -*- coding: utf-8 -*-
import os
import cv2
import sys
import numpy as np
from PIL import Image
import sys
import glob

flist = glob.glob("**/*.png", recursive = True)
index = 0
path=os.getcwd()

for x in flist:
    path_png = x
    image = Image.open(path_png)
    #p_img = image.convert("P")
    p_img  = image.quantize(colors=256, method=2)
    p_img.save(path_png)
    print(path_png,'=> 8 bit')
    index+=1





