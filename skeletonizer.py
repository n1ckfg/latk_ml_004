import sys, os, glob
import cv2
import random
import latk
import kinect_converter as kc
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

inputLinesPath = argv[0]
inputRgbPath = argv[1]
inputDepthPath = argv[2]
lineThreshold = int(argv[3])
useSwig = bool(int(argv[4]))
doInpainting = bool(int(argv[5]))
csize = 10
maxIter = 999

if (useSwig == True):
    sys.path.append("skeleton-tracing/swig") # C library
else:
    sys.path.append("skeleton-tracing/py") # pure python
from trace_skeleton import *

print("")
print("Reading lines from : " + inputLinesPath)
print("Reading rgb from : " + inputRgbPath)
print("Reading depth from : " + inputDepthPath)
print("")

lineFilesList = os.listdir(inputLinesPath)
rgbFilesList = os.listdir(inputRgbPath)
depthFilesList = os.listdir(inputDepthPath)

'''
if (len(lineFilesList) != len(rgbFilesList) or (rgbFilesList) != len(depthFilesList)):
    print("*** Error: file lists are different lengths. ***")
    return
'''

la = latk.Latk()
la.layers.append(latk.LatkLayer())

for i in range(0, len(lineFilesList)):
    lineFileName = lineFilesList[i]
    rgbFileName = rgbFilesList[i]
    depthFileName = depthFilesList[i]

    print(str(i+1) + " / " + str(len(lineFilesList)))
    print("Reading line file : " + lineFileName)
    print("Reading rgb file : " + rgbFileName)
    print("Reading depth file : " + depthFileName)
    print("")

    im0 = cv2.imread(os.path.join(inputLinesPath, lineFileName))
    im0 = cv2.bitwise_not(im0) # invert
    imWidth = len(im0[0])
    imHeight = len(im0)
    im = (im0[:,:,0] > lineThreshold).astype(np.uint8)
    
    if (useSwig == True):
        im = skeletonize(im).astype(np.uint8)
    else:
        im = thinning(im)

    imRgb = cv2.imread(os.path.join(inputRgbPath, rgbFileName))
    imRgb = cv2.resize(imRgb, [imWidth, imHeight])
    
    imDepth = cv2.imread(os.path.join(inputDepthPath, depthFileName))
    imDepth = cv2.resize(imDepth, [imWidth, imHeight])

    if (doInpainting == True):
        mask = cv2.cvtColor(imDepth, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 16, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        imDepth = cv2.inpaint(imDepth, mask, 3, cv2.INPAINT_TELEA) # source, mask, radius, method (TELEA or NS)
        #out = Image.fromarray(imDepth)
        #out.save("test.jpg") 

    rects = []
    # im, x, y, w, h, csize, maxIter, rects
    if (useSwig == True):
        polys = from_numpy(im, csize, maxIter)
    else:
        polys = traceSkeleton(im, 0, 0, im.shape[1], im.shape[0], csize, maxIter, rects)

    frame = latk.LatkFrame(frame_number=i)

    for stroke in polys:
        lPoints = []
        for point in stroke:
            #x = point[0] / imWidth
            #y = 1.0 - (point[1] / imHeight)

            depthPixel = imDepth[point[1]][point[0]]
            #z = 1.0 - (depthPixel[0] / 255)
            
            rgbPixel = imRgb[point[1]][point[0]]
            rgbPixel2 = (rgbPixel[2] / 255, rgbPixel[1] / 255, rgbPixel[0] / 255, 1)

            co = kc.uvd_to_xyz(u=point[0], v=point[1], d=abs(255 - depthPixel[0]), scale=0.1)
            co2 = (co[0], co[2], co[1])
            lPoint = latk.LatkPoint(co2)
            lPoint.vertex_color = rgbPixel2
            lPoints.append(lPoint)

        if (len(lPoints) > 1):
            frame.strokes.append(latk.LatkStroke(lPoints))

    la.layers[0].frames.append(frame)

outputFile = "output.latk"
print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
print("Writing " + outputFile)
la.write(outputFile)

