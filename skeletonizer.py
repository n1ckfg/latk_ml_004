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
    inputDepthPath = argv[1]
    inputRgbPath = argv[2]
    threshold = int(argv[3])

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

    imDepth = cv2.imread(inputDepthPath)
    imRgb = cv2.imread(inputRgbPath)

    rects = []
    polys = traceSkeleton(im,0,0,im.shape[1],im.shape[0],10,999,rects)

    la = latk.Latk()
    la.layers.append(latk.LatkLayer())
    frame = latk.LatkFrame(frame_number=0)
    la.layers[0].frames.append(frame)

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

        la.layers[0].frames[0].strokes.append(latk.LatkStroke(lPoints))

    la.write("output.latk")

main()