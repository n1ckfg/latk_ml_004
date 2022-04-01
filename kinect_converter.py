# https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f

import numpy as np

# Kinect 2
cx = 254.878
cy = 205.395
focalx = 365.456
focaly = 365.456

def uvd_to_xyz(u, v, d, scale=1):
    d *= scale
    x_over_z = (cx - u) / focalx
    y_over_z = (cy - v) / focaly

    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    #print((x, y, z))

    return (x, y, z)