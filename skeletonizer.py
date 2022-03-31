import sys, os, glob
sys.path.append("skeleton-tracing/py")
#sys.path.append("skeleton-tracing/swig") # C

from trace_skeleton import *
import cv2
import random
import latk

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] # get all args after "--"

    inputLinesPath = argv[0]
    inputRgbPath = argv[1]
    inputDepthPath = argv[2]
    threshold = int(argv[3])

    print("")
    print("Reading lines from : " + inputLinesPath)
    print("Reading rgb from : " + inputRgbPath)
    print("Reading depth from : " + inputDepthPath)
    print("")

    lineFilesList = os.listdir(inputLinesPath)
    rgbFilesList = os.listdir(inputRgbPath)
    depthFilesList = os.listdir(inputDepthPath)

    if not (len(lineFilesList) == len(rgbFilesList) == len(depthFilesList)):
        print("*** Error: file lists are different lengths. ***")
        return

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
        im = (im0[:,:,0] > threshold).astype(np.uint8)
        im = thinning(im)

        imRgb = cv2.imread(os.path.join(inputRgbPath, rgbFileName))
        imRgb = cv2.resize(imRgb, [imWidth, imHeight])
        imDepth = cv2.imread(os.path.join(inputDepthPath, depthFileName))
        imDepth = cv2.resize(imDepth, [imWidth, imHeight])

        rects = []
        polys = traceSkeleton(im,0,0,im.shape[1],im.shape[0],10,999,rects)

        frame = latk.LatkFrame(frame_number=i)

        for stroke in polys:
            lPoints = []
            for point in stroke:
                x = point[0] / imWidth
                y = 1.0 - (point[1] / imHeight)

                depthPixel = imDepth[point[1]][point[0]]
                z = 1.0 - (depthPixel[0] / 255)
                
                rgbPixel = imRgb[point[1]][point[0]]
                rgbPixel2 = (rgbPixel[2] / 255, rgbPixel[1] / 255, rgbPixel[0] / 255, 1)

                co = (x, z, y)
                lPoint = latk.LatkPoint(co)
                lPoint.vertex_color = rgbPixel2
                lPoints.append(lPoint)

            frame.strokes.append(latk.LatkStroke(lPoints))

        la.layers[0].frames.append(frame)

    outputFile = "output.latk"
    print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
    print("Writing " + outputFile)
    la.write(outputFile)

main()