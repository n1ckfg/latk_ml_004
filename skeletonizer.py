import sys
sys.path.append("skeleton-tracing/py")
#sys.path.append("skeleton-tracing/swig")

from trace_skeleton import *
import cv2
import random
import latk

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] # get all args after "--"

    inputPath = argv[0]
    threshold = int(argv[1])

    print("Reading from : " + inputPath)

    url = ""
    outputPathArray = inputPath.split(".")
    for i in range(0, len(outputPathArray)-1):
        url += outputPathArray[i]
    url += ".latk"
  
    im0 = cv2.imread(inputPath)
    im0 = cv2.bitwise_not(im0) # invert
    imWidth = len(im0[0])
    imHeight = len(im0)

    im = (im0[:,:,0] > threshold).astype(np.uint8)
    im = thinning(im)

    rects = []
    polys = traceSkeleton(im,0,0,im.shape[1],im.shape[0],10,999,rects)

    la = latk.Latk()
    la.layers.append(latk.LatkLayer())
    frame = latk.LatkFrame(frame_number=0)
    la.layers[0].frames.append(frame)

    for stroke in polys:
        lPoints = []
        for point in stroke:
            point[0] /= imWidth
            point[1] /= imHeight
            
            lPoint = (point[0], point[1], 0)
            lPoints.append(latk.LatkPoint(lPoint))
        la.layers[0].frames[0].strokes.append(latk.LatkStroke(lPoints))

    la.write("output.latk")

main()