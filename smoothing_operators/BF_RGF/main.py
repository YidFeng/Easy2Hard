import os
import cv2
from matplotlib.pyplot import *
output_dir = '../results/'
for file in os.listdir('../../realGTfrom'):
    img = cv2.imread('../../realGTfrom/'+file)
    bf_filtered = cv2.bilateralFilter(src=img, d=45, sigmaColor=200,sigmaSpace=1000)
    rgf_filtered = cv2.ximgproc.rollingGuidanceFilter(src=img, d=45, sigmaColor=150, sigmaSpace=1000, numOfIter=4)
    cv2.imwrite(output_dir+'BF/'+file, bf_filtered)
    cv2.imwrite(output_dir+'RGF/'+file, rgf_filtered)
