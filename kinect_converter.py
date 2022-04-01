import numpy as np

# https://docs.opencv.org/4.x/dc/d3a/classcv_1_1viz_1_1Camera.html#a753809aff611cdcc5a8a377126156b05
# Generic
cx = 320
cy = 240
focalx = 525
focaly = 525

'''
# Azure Kinect (4K mode)
# https://forum.open3d.org/t/azure-kinect-intrinsic-structure/121
cx = 2044.2911376953125
cy = 1565.8837890625
focalx = 1959.37890625
focaly = 1958.880126953125

# Kinect 2
# https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
cx = 254.878
cy = 205.395
focalx = 365.456
focaly = 365.456
'''

def uvd_to_xyz(u, v, d, scale=1):
    d *= scale
    x_over_z = (cx - u) / focalx
    y_over_z = (cy - v) / focaly

    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    #print((x, y, z))

    return (x, y, z)